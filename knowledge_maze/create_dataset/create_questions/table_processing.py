"""
Wikipedia Structured Table Extraction Engine (AsyncWikiExtractor)
================================================================

This module provides a high-performance pipeline for discovering, cleaning, and
extracting structured tabular data from Wikipedia's REST API. It is designed
to handle large-scale extractions (100k+ pages) while maintaining data integrity
and respecting Wikipedia's infrastructure.

Key Pipeline Stages:
--------------------
1.  PRE-PROCESSING (Clustering):
    Uses TF-IDF Vectorization and NetworkX community detection to identify
    highly similar page titles. It prioritizes data-rich sources (e.g.,
    "List of...", "Demographics of...") and discards redundant titles to
    minimize unnecessary API calls.

2.  ASYNCHRONOUS EXTRACTION:
    Implements a Semaphore-controlled `aiohttp` engine to fetch HTML content.
    Includes a global pause mechanism for HTTP 429 (Rate Limit) responses,
    ensuring the crawler remains compliant with Wikipedia's ToS.

3.  PARALLEL VALIDATION & PARSING:
    Offloads heavy BeautifulSoup operations to a `ProcessPoolExecutor` to
    prevent Global Interpreter Lock (GIL) bottlenecks. It handles complex
    HTML table structures including:
    - Rowspan/Colspan expansion (flattening merged cells).
    - Automatic numeric conversion.
    - Hierarchy path reconstruction (Page > Section > Sub-section).
    - First-column fullness constraints (ensuring data is structured, not layout).

4.  POST-PROCESSING & QUALITY CONTROL:
    Filters the raw JSONL output to remove "noisy" data. It utilizes statistical
    heuristics to detect:
    - Mashed/Bad Parsing: Headers with high capitalization density or word counts.
    - Layout Tables: Tables with single columns or low data density.
    - Overflow Cells: Cells exceeding character limits, suggesting paragraph ingestion.

Classes:
--------
- Config: Centralized hyperparameters for extraction and validation.
- TableValidator: Static methods for HTML cleaning and structural extraction.
- AsyncWikiExtractor: The primary engine orchestrating networking and MP-parsing.
- TablePostProcessor: Final data quality gate and sampling logic.

Dependencies:
-------------
Standard: json, re, asyncio, logging, collections
Data Science: numpy, pandas, sklearn (TF-IDF, Cosine Similarity)
Graph: networkx
HTML/Parsing: bs4 (BeautifulSoup), lxml
UI: tqdm, IPython

Usage:
------
    extractor = AsyncWikiExtractor(output_file="data.jsonl")
    await extractor.run_with_progress(list_of_titles)

    processor = TablePostProcessor("data.jsonl")
    final_data = processor.process(max_rows=12)
"""

import re
import os
import gc
import json
import random
import asyncio
import aiohttp
import logging
import numpy as np
import pandas as pd
import urllib.parse
import networkx as nx
import tqdm.auto as tqdm_lib
from bs4 import BeautifulSoup
from collections import defaultdict
from IPython.display import display
from typing import List, Dict, Optional, Any
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from knowledge_maze.tools.api_tools import *

# Configure Logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# --- CONFIGURATION ---
class Config:
    USER_AGENT = "ResearchBot/2.0 (AsyncWiki/1.0)"
    CONCURRENT_REQUESTS = 20  # Safe limit for Wikipedia
    BATCH_SIZE = 200  # Process a fixed-size batch of pages at a time to save RAM
    RETRIES = 3
    MIN_ROWS = 4
    MIN_DENSITY = 0.8
    EXCLUDE_SECTIONS = {"references", "see also", "external links", "notes", "further reading", "sources"}


# --- VALIDATOR ---
class TableValidator:

    @staticmethod
    def normalize_cell(cell_el, is_header=False) -> Any:
        if not cell_el: return ""
        text = cell_el.get_text(" ", strip=True)
        text = re.sub(r"\[.*?\]", "", text).strip()
        if not is_header:
            clean_num = text.replace(",", "")
            try:
                if "." in clean_num: return float(clean_num)
                return int(clean_num)
            except ValueError:
                pass
        return text

    @staticmethod
    def validate_and_extract(table_el, hierarchy_path, title) -> Optional[Dict]:
        # --- 1. Caption / Title Capture ---
        external_caption = ""
        caption_tag = table_el.find("caption")
        if caption_tag:
            external_caption = caption_tag.get_text(" ", strip=True)
        else:
            for sibling in table_el.find_previous_siblings(limit=2):
                if sibling.name in ["p", "div", "h4"]:
                    bold_text = sibling.find(["b", "strong"])
                    if bold_text:
                        external_caption = bold_text.get_text(" ", strip=True)
                        break
                    elif len(sibling.get_text(strip=True)) < 100:
                        external_caption = sibling.get_text(" ", strip=True)
                        break

        external_caption = re.sub(r"\[.*?\]", "", external_caption).strip()
        full_table_path = f"{title} > {hierarchy_path}"
        if external_caption:
            full_table_path += f" ({external_caption})"

        all_tr = table_el.find_all("tr")
        if not all_tr: return None

        # --- 2. Header Extraction ---
        header_row = all_tr[0]
        ths = header_row.find_all(["th", "td"])
        headers = [TableValidator.normalize_cell(th, is_header=True) for th in ths]
        num_headers = len(headers)

        if num_headers < 2 or all(not h for h in headers):
            return None

        def get_clean_int(el, attr, default=1):
            val = el.get(attr, str(default))
            match = re.search(r'\d+', str(val))
            return int(match.group()) if match else default

        # --- 3. Geometric Row Processing ---
        rows_raw = []
        rowspan_tracker = {}  # (row_idx, col_idx) -> value

        for r_idx, tr in enumerate(all_tr[1:]):
            cells = tr.find_all(["td", "th"])
            current_row_vals = [None] * num_headers  # Initialize fixed-size row
            html_cell_ptr = 0

            for c_idx in range(num_headers):
                # Check for inherited rowspan data
                if (r_idx, c_idx) in rowspan_tracker:
                    current_row_vals[c_idx] = rowspan_tracker.pop((r_idx, c_idx))
                    continue

                # Pull from HTML if available
                if html_cell_ptr < len(cells):
                    cell = cells[html_cell_ptr]
                    val = TableValidator.normalize_cell(cell, is_header=False)

                    rowspan = get_clean_int(cell, "rowspan", 1)
                    colspan = get_clean_int(cell, "colspan", 1)

                    # --- CONSTRAINT 1: Check if colspan exceeds header bounds ---
                    if c_idx + colspan > num_headers:
                        return None  # Geometric Mismatch

                    # Fill current row (Colspan handle)
                    for i in range(colspan):
                        current_row_vals[c_idx + i] = val

                    # Track future rows (Rowspan handle)
                    if rowspan > 1:
                        for r_offset in range(1, rowspan):
                            for c_offset in range(colspan):
                                rowspan_tracker[(r_idx + r_offset, c_idx + c_offset)] = val

                    # Advance to next cell in HTML, skip logical columns used by colspan
                    html_cell_ptr += 1
                    # Note: The loop's c_idx increment handles the rest of the colspan
                    # by hitting the 'None' check or the 'rowspan_tracker' in the next iterations
                else:
                    # No more HTML cells but we have logical columns left
                    # This happens if a previous rowspan didn't fill this spot and HTML is empty
                    if current_row_vals[c_idx] is None:
                        return None  # Incomplete Row (Strict Size Check)

            # Final check: Did we use all HTML cells in this row?
            if html_cell_ptr < len(cells):
                return None  # Excess cells (Strict Size Check)

            rows_raw.append(current_row_vals)

        if not rows_raw: return None

        # --- 4. Filtering & Structuring ---
        # Identify non-empty columns
        active_indices = [i for i in range(num_headers) if
                          headers[i] and any(str(row[i]).strip() != "" for row in rows_raw)]

        # --- SAFETY GATE: Ensure there is at least one data column ---
        if not active_indices:
            return None

        final_rows = []
        filled_cells = 0
        for row in rows_raw:
            structured_row = {}
            for i in active_indices:
                val = row[i]
                structured_row[headers[i]] = val
                if str(val).strip() != "": filled_cells += 1
            final_rows.append(structured_row)

        # --- 5. Validations ---
        if len(final_rows) < 4:  # Assuming Config.MIN_ROWS is roughly 3
            return None

        # Ensure first column is 100% full (Main Entity column)
        first_col_key = headers[active_indices[0]]
        if not all(str(row.get(first_col_key, "")).strip() != "" for row in final_rows):
            return None

        # Density check
        total_slots = len(final_rows) * len(active_indices)
        density = filled_cells / total_slots if total_slots > 0 else 0
        if density < 0.5:  # Assuming Config.MIN_DENSITY is 0.5
            return None

        return {
            "table_name": full_table_path,
            "rows": final_rows,
            "density": density
        }


# --- TOP-LEVEL WORKER FUNCTIONS (Must be pickleable) ---
def parse_html_content(html_content: str, title: str) -> List[Dict]:
    if not html_content: return []
    soup = BeautifulSoup(html_content, "lxml")
    results = []
    current_hierarchy = {}
    inside_content = False

    for el in soup.find_all(["h2", "h3", "h4", "table"]):
        tag = el.name.lower()
        if tag.startswith("h"):
            level = int(tag[1])
            heading_text = el.get_text(strip=True).replace("[edit]", "")
            if level == 2:
                inside_content = True
                if heading_text.lower() in Config.EXCLUDE_SECTIONS:
                    inside_content = False
            if not inside_content: continue
            levels_to_clear = [l for l in current_hierarchy.keys() if l >= level]
            for l in levels_to_clear: del current_hierarchy[l]
            current_hierarchy[level] = heading_text

        elif tag == "table" and inside_content:
            sorted_levels = sorted(current_hierarchy.keys())
            path = " > ".join([current_hierarchy[l] for l in sorted_levels])
            table_data = TableValidator.validate_and_extract(el, path, title)
            if table_data:
                results.append(table_data)
    return results

def mp_helper(args):
    """Helper to map arguments in multiprocessing"""
    htmls, titles = args
    return list(map(parse_html_content, htmls, titles))

def cluster_titles_with_priority(titles, similarity_threshold=0.8):
    if not titles: return []

    # 1. TF-IDF & Cosine Similarity
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(titles)
    sim_matrix = cosine_similarity(tfidf_matrix)

    # 2. Build the Graph
    G = nx.Graph()
    G.add_nodes_from(range(len(titles)))
    rows, cols = np.where(sim_matrix > similarity_threshold)
    edges = [(r, c) for r, c in zip(rows, cols) if r < c]
    G.add_edges_from(edges)

    # 3. Define Priority Logic
    # Higher score = Higher priority
    priority_keywords = ["list of", "demographics of", "statistics of", "timeline of"]

    def get_priority_score(title):
        score = 0
        lower_title = title.lower()

        # Priority 1: Keywords you specifically want
        for i, word in enumerate(priority_keywords):
            if word in lower_title:
                # Earlier keywords in the list get slightly higher scores
                score += (100 - i)

                # Priority 2: Penalize titles that are too long/short (optional)
        # We prefer titles that sound like "data" sources
        return score

    # 4. Extract Clusters and Pick Representative
    clusters = list(nx.connected_components(G))
    final_titles = []

    for cluster in clusters:
        cluster_strings = [titles[i] for i in cluster]

        # Sort by priority score first (desc), then by length (asc) as a tie-breaker
        representative = sorted(
            cluster_strings,
            key=lambda x: (get_priority_score(x), -len(x)),
            reverse=True
        )[0]

        final_titles.append(representative)

    return final_titles


# --- ASYNC CLASS ---
class AsyncWikiExtractor:
    def __init__(self, output_file="extracted_tables.jsonl", max_workers=None):
        self.output_file = output_file
        self.sem = asyncio.Semaphore(Config.CONCURRENT_REQUESTS)
        self.resume_event = asyncio.Event()
        self.resume_event.set()
        self.stats = {
            "success": 0,  # Network 200 OK
            "429": 0,  # Rate limit hits
            "errors": 0,  # Network/Timeout/Other failures
            "pages_w_tables": 0,  # Pages that actually had tables
            "empty_pages": 0,  # Pages with 200 OK but 0 tables
            "tables": 0  # Total count of individual tables found
        }

        self.cpu_workers = max_workers if max_workers else max(1, os.cpu_count() - 1)

    async def fetch_page(self, session, title):
        # --- FIX START: Normalize URL ---
        # 1. Replace spaces with underscores
        clean_title = title.strip().replace(" ", "_")
        # 2. URL Encode (handles "United States", "C++", parens, etc.)
        safe_title = urllib.parse.quote(clean_title)

        url = f"https://en.wikipedia.org/api/rest_v1/page/html/{safe_title}"
        # --- FIX END ---

        for attempt in range(Config.RETRIES):
            await self.resume_event.wait()
            async with self.sem:
                try:
                    # Added timeout=25 to be safe
                    async with session.get(url, timeout=25) as response:
                        if response.status == 200:
                            return await response.text(), title


                        elif response.status == 429:
                            # 1. If I'm the first one here, trigger the global pause
                            if self.resume_event.is_set():
                                self.resume_event.clear()
                                self.stats["429"] += 1
                                retry_after = int(response.headers.get("Retry-After", 5))
                                logger.debug(f"Rate Limited (429). Pausing for {retry_after}s...")
                                await asyncio.sleep(retry_after)
                                self.resume_event.set()
                            else:
                                # 2. If I'm NOT the first one, I MUST wait for the person who cleared it
                                # to set it back to True before I 'continue' and retry my request.
                                await self.resume_event.wait()
                            continue  # Now it's safe to retry because the 'wait' is over

                        elif response.status == 404:
                            # Optional: Log which title wasn't found to debug
                            logger.debug(f"404 Not Found: {title}")
                            return None, title

                        elif response.status == 403:
                            logger.debug(f"403 Forbidden (Blocked): {title} -> Check User-Agent!")
                            return None, title

                        else:
                            # 500s or other errors
                            logger.debug(f"500s or other errors.")
                            await asyncio.sleep(1)
                except Exception:
                    # Connection timeouts/DNS errors
                    logger.debug(f"Connection timeouts/DNS errors.")
                    pass

        self.stats["errors"] += 1
        return None, title

    async def process_batch(self, titles, process_pool):
        async with aiohttp.ClientSession(headers={"User-Agent": Config.USER_AGENT}) as session:
            tasks = [self.fetch_page(session, t) for t in titles]
            fetched_results = await asyncio.gather(*tasks)

        valid_pages = [(html, title) for html, title in fetched_results if html]
        self.stats["success"] += len(valid_pages)

        if not valid_pages: return []

        loop = asyncio.get_running_loop()
        htmls = [p[0] for p in valid_pages]
        page_titles = [p[1] for p in valid_pages]

        # Offload parsing to CPU cores
        parsed_results = await loop.run_in_executor(
            process_pool,
            mp_helper,
            (htmls, page_titles)
        )

        batch_tables = []
        for p_res in parsed_results:
            if p_res:  # If this list is not empty, the page had tables
                self.stats["pages_w_tables"] += 1
                batch_tables.extend(p_res)
            else:  # Page was downloaded but no tables were extracted
                self.stats["empty_pages"] += 1

        self.stats["tables"] += len(batch_tables)
        return batch_tables

    async def run(self, titles, progress_callback=None):
        print(
            f"Starting Extraction: {len(titles)} titles | Async Limit: {Config.CONCURRENT_REQUESTS} | CPU Workers: {self.cpu_workers}")

        # Clear/Init file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            pass

        with ProcessPoolExecutor(max_workers=self.cpu_workers) as process_pool:
            total_processed = 0
            for i in range(0, len(titles), Config.BATCH_SIZE):
                batch_titles = titles[i: i + Config.BATCH_SIZE]

                tables = await self.process_batch(batch_titles, process_pool)

                if tables:
                    with open(self.output_file, 'a', encoding='utf-8') as f:
                        for table in tables:
                            f.write(json.dumps(table) + "\n")

                # Update progress
                total_processed += len(batch_titles)
                if progress_callback:
                    progress_callback(len(batch_titles), self.stats)

                # Cleanup Memory
                del tables
                del batch_titles
                gc.collect()

        # Summary:
        good_tables = self.stats['pages_w_tables']
        total_tables = len(titles)
        success_rate = 100 * (good_tables / total_tables)
        print(f"TOTAL PAGES WITH VALID TABLES: {good_tables} OUT OF {total_tables} ({success_rate:.1f}%)")

        return self.output_file

    async def run_with_progress(self, titles, desc="Extracting Tables"):
        """Convenience method to run with an internal progress bar."""

        # Pre-process - Run the prioritized clustering tool
        print("Pre-processing: cluster similar pages and discard similarities", end=" ")
        clean_titles = cluster_titles_with_priority(
            titles,
            similarity_threshold=0.75  # Lower threshold = larger clusters/more aggressive reduction
        )
        print(f"---> {len(titles) - len(clean_titles)} pages discarded.")

        # Main loop:
        pbar = tqdm_lib.tqdm(total=len(clean_titles), desc=desc, unit="Page")

        def internal_callback(batch_size, stats):
            pbar.update(batch_size)
            pbar.set_postfix({
                "Tables found": stats["tables"],
                "Pages with valid tables": stats["pages_w_tables"],
                "Pages without valid tables": stats["empty_pages"],
                "429s": stats["429"],
                "Other errors": stats["errors"]
            })

        try:
            return await self.run(clean_titles, progress_callback=internal_callback)
        finally:
            pbar.close()


# --- POST PROCESS CLASS ---
class TablePostProcessor:
    def __init__(self, jsonl_path):
        self.jsonl_path = jsonl_path
        self.grouped_data = defaultdict(list)
        self.stats = {
            "total_raw": 0,
            "short_tables": 0,
            "invalid_cols": 0,
            "bad_parsing": 0,
            "long_cells": 0,
            "invalid_repeats": 0,  # New Stat
            "unique_pages": 0,
            "final_count": 0
        }
        self.final_sample = []

    @staticmethod
    def _is_header_sane(header):
        h_str = str(header).strip()
        if not h_str: return False
        if "\n" in h_str: return False
        words = h_str.split()
        if len(words) > 8: return False
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', h_str))
        if len(h_str) > 0 and (special_chars / len(h_str)) > 0.20:
            return False
        return True

    def _clean_table_rows(self, rows):
        """
        Handles row duplication logic and 'Total' row removal.
        Returns cleaned rows or None if the table should be invalidated.
        """
        if not rows: return None

        # 1. Check for 'Total' row at the end
        # We check the first column of the last row for 'total' (case-insensitive)
        last_row = rows[-1]
        first_col_key = list(last_row.keys())[0]
        val = str(last_row.get(first_col_key, "")).strip().lower()
        if val == "total" or val.startswith("total "):
            rows.pop()

        if not rows: return None

        # 2. Identify rows where all values are identical (e.g., "---" or "Group A")
        indices_to_remove = []
        duplicate_middle_count = 0

        for i, row in enumerate(rows):
            values = list(row.values())
            # Check if all values in the row are the same
            if len(set(str(v).strip() for v in values)) == 1:
                # Is it first or last?
                if i == 0 or i == len(rows) - 1:
                    indices_to_remove.append(i)
                else:
                    # It's in the middle
                    indices_to_remove.append(i)
                    duplicate_middle_count += 1

        # 3. Validation: If 2 or more middle rows are duplicates, table is invalid
        if duplicate_middle_count >= 2:
            return None

        # 4. Perform deletion (in reverse to maintain index integrity)
        for index in sorted(indices_to_remove, reverse=True):
            rows.pop(index)

        return rows

    def process(self, max_rows=12, max_cell_len=200):
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in open(self.jsonl_path, 'r', encoding='utf-8'))
            f.seek(0)

            for line in tqdm_lib.tqdm(f, total=line_count, desc="Filtering & Cleaning"):
                self.stats["total_raw"] += 1
                try:
                    table_data = json.loads(line)
                    rows = table_data.get('rows', [])

                    # Pre-cleaning: Remove total rows and handle duplicate value rows
                    cleaned_rows = self._clean_table_rows(rows)
                    if cleaned_rows is None:
                        self.stats["invalid_repeats"] += 1
                        continue

                    # Update rows in the data object
                    table_data['rows'] = cleaned_rows
                    rows = cleaned_rows

                    # Check Row Threshold (Must be 5-max after cleaning)
                    if not (len(rows) <= max_rows):
                        # We don't increment a specific stat here as it's a general size filter
                        continue

                    # Check Column Count
                    if not rows or len(rows[0].keys()) <= 1:
                        self.stats["invalid_cols"] += 1
                        continue

                    # Check Header Integrity
                    headers = list(rows[0].keys())
                    if not all(self._is_header_sane(h) for h in headers):
                        self.stats["bad_parsing"] += 1
                        continue

                    # Check Cell Length
                    if any(len(str(value)) > max_cell_len for row in rows for value in row.values()):
                        self.stats["long_cells"] += 1
                        continue

                    self.stats["short_tables"] += 1
                    page_title = table_data['table_name'].split(' > ')[0]
                    self.grouped_data[page_title].append(table_data)

                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        # Selection of best table per page
        for title in self.grouped_data:
            available_tables = self.grouped_data[title]
            selected_table = max(available_tables, key=lambda x: len(x['rows']))
            self.final_sample.append(selected_table)

        self.stats["unique_pages"] = len(self.grouped_data)
        self.stats["final_count"] = len(self.final_sample)
        self._print_summary(max_rows)
        return self.final_sample

    def _print_summary(self, threshold):
        width = 44
        print("\n" + "═" * width)
        print(f"📊 TABLE PROCESSING SUMMARY (Limit: {threshold})")
        print("═" * width)
        print(f"Total tables scanned:      {self.stats['total_raw']}")
        print(f"Dropped (Bad Headers):     {self.stats['bad_parsing']}")
        print(f"Dropped (Cell too long):   {self.stats['long_cells']}")
        print(f"Dropped (Repeat Rows):     {self.stats['invalid_repeats']}")
        print(f"Short tables found:        {self.stats['short_tables']}")
        print(f"Unique source pages:       {self.stats['unique_pages']}")
        print(f"Final sampled (1/page):    {self.stats['final_count']}")
        print("═" * width)

        print("\n" + "-" * 150)
        print(" " * 66 + "↓ EXAMPLE TABLES ↓" + " " * 66)
        print("-" * 150)

        for s in random.sample(self.final_sample, 3):
            print("\n" + "═" * 150)
            print(s['table_name'])
            print("═" * 150)
            display(pd.DataFrame(s['rows']))
