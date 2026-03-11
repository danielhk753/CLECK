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
import tqdm.auto as tqdm_lib
from bs4 import BeautifulSoup
from collections import defaultdict
from typing import Dict, Optional, Any
from IPython.display import display, HTML
from concurrent.futures import ProcessPoolExecutor

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
        context = {
            "header_chain": f"{title} > {hierarchy_path}",
            "caption": None,
            "additional_text": None
        }

        # A. Extract official Caption tag
        caption_tag = table_el.find("caption")
        if caption_tag:
            raw_caption = caption_tag.get_text(" ", strip=True)
            context["caption"] = re.sub(r"\[.*?\]", "", raw_caption).strip()

        # B. Extract Sibling Context (Additional Text)
        # We look back at up to 2 siblings to find descriptive text or a sub-header
        for sibling in table_el.find_previous_siblings(limit=2):
            if sibling.name in ["p", "div", "h4", "h5"]:
                # Get the full text but strip Wikipedia reference brackets [1][2]
                raw_text = sibling.get_text(" ", strip=True)
                clean_text = re.sub(r"\[.*?\]", "", raw_text).strip()

                if not clean_text:
                    continue

                # Logic: If it's a very short paragraph or a specific "Season/Year" mention,
                # we take the whole thing. Otherwise, we just take the first sentence.
                if len(clean_text) < 100:
                    context["additional_text"] = clean_text
                else:
                    # Isolate just the first sentence (split by period followed by space)
                    first_sentence = re.split(r'(?<=[.!?])\s+', clean_text)[1] if '. ' in clean_text else clean_text
                    context["additional_text"] = first_sentence.strip()

                break  # Stop once we find the most immediate text sibling

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
            "table_name": context, #full_table_path,
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
        print(f"Starting Extraction: {len(titles)} titles | Async Limit: {Config.CONCURRENT_REQUESTS} | CPU Workers: {self.cpu_workers}")

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

        return self.output_file

    async def run_with_progress(self, titles, desc="Extracting Tables"):
        """Convenience method to run with an internal progress bar."""

        # Main loop:
        pbar = tqdm_lib.tqdm(total=len(titles), desc=desc, unit="Page")

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
            return await self.run(titles, progress_callback=internal_callback)
        finally:
            pbar.close()


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
            "invalid_repeats": 0,
            "legend_key": 0,
            "long_tables": 0,
            "alphabetical_lists": 0,
            "intra_page_dupes": 0,
            "unique_pages": 0,
            "final_count": 0
        }
        self.final_sample = []

    @staticmethod
    def _is_header_sane(header):
        h_str = str(header).strip()
        if not h_str or "\n" in h_str: return False
        if len(h_str.split()) > 8: return False

        # Calculate special character ratio
        char_count = len(h_str)
        if char_count > 0:
            special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', h_str))
            if (special_chars / char_count) > 0.20:
                return False
        return True

    def _clean_table_rows(self, rows):
        if not rows: return None

        # 1. Remove 'Total' row at the end
        last_row = rows[-1]
        first_col_key = list(last_row.keys())[0]
        val = str(last_row.get(first_col_key, "")).strip().lower()
        if val == "total" or val.startswith("total "):
            rows.pop()

        if not rows: return None

        # 2. Identify duplicate value rows
        indices_to_remove = []
        duplicate_middle_count = 0

        for i, row in enumerate(rows):
            values = list(row.values())
            # Check if all values in the row are the same (e.g., "---")
            if len(set(str(v).strip() for v in values)) == 1:
                if i == 0 or i == len(rows) - 1:
                    indices_to_remove.append(i)
                else:
                    indices_to_remove.append(i)
                    duplicate_middle_count += 1

        # 3. Validation: Too many duplicate middle rows = bad table layout
        if duplicate_middle_count >= 2:
            return None

        # 4. Remove identified rows in reverse order
        for index in sorted(indices_to_remove, reverse=True):
            rows.pop(index)

        return rows

    def _generate_wiki_url(self, page_title, header_chain):
        """
        Creates a deep link to the specific section on Wikipedia.
        Wikipedia anchors use underscores for spaces.
        """
        base_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page_title.replace(' ', '_'))}"

        parts = [p.strip() for p in header_chain.split('>')]
        if len(parts) > 1:
            # The last part of the chain is usually the closest section header
            section = parts[-1]
            anchor = urllib.parse.quote(section.replace(' ', '_'))
            return f"{base_url}#{anchor}"
        return base_url

    def _intra_page_deduplication(self, tables, max_per_page=4):

        if not tables:
            return []

        # prefer wider + longer tables
        tables = sorted(
            tables,
            key=lambda x: (len(x['rows'][0]), len(x['rows'])),
            reverse=True
        )

        # ---------------------------
        # helper 1: similarity
        # ---------------------------

        def calculate_similarity(t1, t2):

            # ---- TITLE IOU ----
            chain1 = t1['table_name'].get('header_chain', '')
            chain2 = t2['table_name'].get('header_chain', '')

            parts1 = {p.strip().lower() for p in chain1.split(">") if p.strip()}
            parts2 = {p.strip().lower() for p in chain2.split(">") if p.strip()}

            inter = len(parts1 & parts2)
            union = len(parts1 | parts2)

            title_iou = inter / union if union else 0

            # ---- COLUMN IOU ----
            cols1 = {c.lower().split("[")[0].strip() for c in t1['rows'][0]}
            cols2 = {c.lower().split("[")[0].strip() for c in t2['rows'][0]}

            inter = len(cols1 & cols2)
            union = len(cols1 | cols2)

            col_iou = inter / union if union else 0

            # ---- ROW SIMILARITY ----

            def valid(vals, min_len=3, min_unique_ratio=1.0, max_numeric_ratio=0.5):
                # Remove empty or None values
                vals = [v for v in vals if v is not None]

                if len(vals) < min_len:
                    return False

                # Check uniqueness
                if len(set(vals)) < len(vals) * min_unique_ratio:
                    return False

                # Check numeric ratio
                num_numeric = sum(1 for v in vals if isinstance(v, (int, float)))
                if len(vals) == 0:
                    return False
                if num_numeric / len(vals) > max_numeric_ratio:
                    return False

                return True

            def col_vals(table, col):
                return [
                    str(r.get(col, "")).lower().strip()
                    for r in table['rows']
                ]

            row_score = 0

            for c1 in t1['rows'][0]:
                v1 = col_vals(t1, c1)
                if not valid(v1):
                    continue

                s1 = set(v1)

                for c2 in t2['rows'][0]:
                    v2 = col_vals(t2, c2)
                    if not valid(v2):
                        continue

                    s2 = set(v2)

                    inter = len(s1 & s2)
                    union = len(s1 | s2)

                    if union == 0:
                        continue

                    score = inter / union

                    if score > row_score:
                        row_score = score

            return title_iou, col_iou, row_score

        # ---------------------------
        # helper 2: clustering
        # ---------------------------

        def cluster(tables):

            kept = []

            for t in tables:

                duplicate = False

                for k in kept:

                    title, col, row = calculate_similarity(t, k)

                    # duplicate rules
                    if (row > 0.9 and col > 0.4) or \
                       (col > 0.9 and title > 0.5) or \
                       (title > 0.85 and col > 0.4 and row > 0.3):
                        duplicate = True
                        break

                if not duplicate:
                    kept.append(t)

            return kept

        kept_tables = cluster(tables)

        self.stats["intra_page_dupes"] += len(tables) - len(kept_tables[:max_per_page])

        return kept_tables[:max_per_page]

    def process(self, max_rows=12, max_cell_len=200):
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
            f.seek(0)

            for line in tqdm_lib.tqdm(f, total=line_count, desc="Filtering & Cleaning"):
                self.stats["total_raw"] += 1
                try:
                    table_data = json.loads(line)
                    rows = table_data.get('rows', [])
                    meta = table_data.get('table_name', {})
                    header_chain = meta.get('header_chain', '')

                    # 1. Legend/Key Filter
                    chain_end = header_chain.split('>')[-1].strip().lower()
                    if chain_end in ["key", "legend"]:
                        self.stats["legend_key"] += 1
                        continue

                    # 2. Clean Rows
                    cleaned_rows = self._clean_table_rows(rows)
                    if cleaned_rows is None:
                        self.stats["invalid_repeats"] += 1
                        continue

                    table_data['rows'] = cleaned_rows
                    rows = cleaned_rows

                    # 3. Size constraints
                    if not (1 <= len(rows) <= max_rows):
                        self.stats["long_tables"] += 1
                        continue

                    if len(rows[0].keys()) <= 1:
                        self.stats["invalid_cols"] += 1
                        continue

                    # 4. Header Sanity
                    if not all(self._is_header_sane(h) for h in rows[0].keys()):
                        self.stats["bad_parsing"] += 1
                        continue

                    # 5. Cell Length constraints
                    if any(len(str(value)) > max_cell_len for row in rows for value in row.values()):
                        self.stats["long_cells"] += 1
                        continue

                    # 6. Alphabetical Tables
                    if len(header_chain.split(" > ")[-1]) <= 2:
                        self.stats["alphabetical_lists"] += 1
                        continue

                    # Add Deep Link
                    page_title = header_chain.split(' > ')[0]
                    table_data['wiki_url'] = self._generate_wiki_url(page_title, header_chain)

                    self.stats["short_tables"] += 1
                    self.grouped_data[page_title].append(table_data)

                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        # Smart Intra-Page Deduplication
        for title, tables in self.grouped_data.items():
            best_tables = self._intra_page_deduplication(tables)
            self.final_sample.extend(best_tables)

        self.stats["unique_pages"] = len(self.grouped_data)
        self.stats["final_count"] = len(self.final_sample)
        self.display_summary()
        return self.final_sample

    def display_summary(self) -> None:

        html = f"""
        <div style="background: #0f172a; padding: 25px; border-radius: 15px; border: 1px solid #ffffff; color: #f8fafc; font-family: sans-serif; max-width: 900px; margin-left: 0; text-align: left;">
            <h2 style="color: #38bdf8; margin-bottom: 25px; letter-spacing: 1px; text-align: center; width: 100%;"> ⚙️ TABLE PROCESSING REPORT </h2>

            <div style="display: flex; gap: 10px; margin-bottom: 10px;">
                <div style="flex: 1; background: #1e293b; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
                    <small style="color: #94a3b8;">RAW TOTAL</small><br/>
                    <span style="font-size: 20px; font-weight: bold;">{self.stats['total_raw']:,}</span>
                </div>
                <div style="flex: 1; background: #1e293b; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
                    <small style="color: #94a3b8;">VALIDATED</small><br/>
                    <span style="font-size: 20px; font-weight: bold; color: #38bdf8;">
                        {self.stats['total_raw'] - (self.stats['legend_key'] + self.stats['bad_parsing'] + self.stats['long_cells'] + self.stats['invalid_repeats']):,}
                    </span>
                </div>
                <div style="flex: 1; background: #1e293b; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
                    <small style="color: #94a3b8;">SHORT TABLES</small><br/>
                    <span style="font-size: 20px; font-weight: bold; color: #c084fc;">{self.stats['short_tables']:,}</span>
                </div>
            </div>

            <div style="display: flex; gap: 10px; margin-bottom: 20px;">
                <div style="flex: 1; background: #1e293b; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #ffffff;">
                    <small style="color: #94a3b8;">SOURCE PAGES</small><br/>
                    <span style="font-size: 20px; font-weight: bold; color: #fbbf24;">{self.stats['unique_pages']:,}</span>
                </div>
                <div style="flex: 1; background: #1e293b; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #ffffff;">
                    <small style="color: #94a3b8;">FINAL SAMPLED</small><br/>
                    <span style="font-size: 20px; font-weight: bold; color: #34d399;">{self.stats['final_count']:,}</span>
                </div>
            </div>

            <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column;">
                <h4 style="color: #f87171; background-color: #0f172a; margin: 0; padding: 10px; text-align: center; border-bottom: 1px solid #334155; font-size: 14px;">
                    Discard Analytics
                </h4>
                <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                    <tr style="border-bottom: 1px solid #334155;">
                        <td style="padding: 10px; text-align: center; background: #1e293b; color: #94a3b8;">Legend / Key Filter</td>
                        <td style="padding: 10px; text-align: center; background: #1e293b; font-weight: bold; color: #f87171;">{self.stats['legend_key']:,}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #334155;">
                        <td style="padding: 10px; text-align: center; background: #1e293b; color: #94a3b8;">Header Parsing Failures</td>
                        <td style="padding: 10px; text-align: center; background: #1e293b; font-weight: bold; color: #f87171;">{self.stats['bad_parsing']:,}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #334155;">
                        <td style="padding: 10px; text-align: center; background: #1e293b; color: #94a3b8;">Cell Content Too Long</td>
                        <td style="padding: 10px; text-align: center; background: #1e293b; font-weight: bold; color: #f87171;">{self.stats['long_cells']:,}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #334155;">
                        <td style="padding: 10px; text-align: center; background: #1e293b; color: #94a3b8;">Malformed Row Structures</td>
                        <td style="padding: 10px; text-align: center; background: #1e293b; font-weight: bold; color: #f87171;">{self.stats['invalid_repeats']:,}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #334155;">
                        <td style="padding: 10px; text-align: center; background: #1e293b; color: #94a3b8;">Alphabetical Lists Removed</td>
                        <td style="padding: 10px; text-align: center; background: #1e293b; font-weight: bold; color: #f87171;">{self.stats['alphabetical_lists']:,}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #334155;">
                        <td style="padding: 10px; text-align: center; background: #1e293b; color: #94a3b8;">Long Tables Removed</td>
                        <td style="padding: 10px; text-align: center; background: #1e293b; font-weight: bold; color: #f87171;">{self.stats['long_tables']:,}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #334155;">
                        <td style="padding: 10px; text-align: center; background: #1e293b; color: #94a3b8;">Intra-Page Duplicates Removed</td>
                        <td style="padding: 10px; text-align: center; background: #1e293b; font-weight: bold; color: #f87171;">{self.stats['intra_page_dupes']:,}</td>
                    </tr>
                </table>
            </div>
        </div>
        """
        display(HTML(html))

    def display_sample(self, index=None):
        """Displays a specific or random table beautifully with its metadata."""
        if not self.final_sample:
            print("No valid tables available to display.")
            return

        idx = index if index is not None else random.randint(0, len(self.final_sample) - 1)
        sample = self.final_sample[idx]

        meta = sample.get('table_name', {})
        header_chain = meta.get('header_chain', 'Unknown')
        caption = meta.get('caption', 'No Caption')
        additional_text = meta.get('additional_text', '')
        wiki_url = sample.get('wiki_url', '#')

        df = pd.DataFrame(sample['rows'])

        # Build the HTML block for the text metadata
        meta_html = f"""
        <div style="background: #1e293b; padding: 20px; border-radius: 8px; border: 1px solid #38bdf8; color: #f8fafc; font-family: sans-serif; margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h3 style="margin: 0; color: #38bdf8; font-size: 16px;">{header_chain}</h3>
                <a href="{wiki_url}" target="_blank" style="background-color: #38bdf8; color: #0f172a; padding: 5px 10px; border-radius: 5px; text-decoration: none; font-size: 12px; font-weight: bold;">🔗 View on Wikipedia</a>
            </div>
            <div style="font-size: 13px;">
                <p style="margin: 5px 0;"><b>Caption:</b> <span style="color: #94a3b8;">{caption if caption else None}</span></p>
                <p style="margin: 5px 0;"><b>Context:</b> <span style="color: #94a3b8;">{additional_text[:200] if additional_text else None}{'...' if additional_text and len(additional_text) > 200 else ''}</span></p>
            </div>
        </div>
        """

        display(HTML(meta_html))
        display(df)
