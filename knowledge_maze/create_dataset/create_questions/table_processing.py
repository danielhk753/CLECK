import re
import time
import requests
import threading
from tqdm import tqdm
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed


class TableExtractor:
    EXCLUDE_SECTIONS = {"references", "see also", "external links", "notes", "further reading"}

    def __init__(self, user_agent="TableExtractor/1.0", min_rows=3, min_density=0.9):
        self.headers = {"User-Agent": f"{user_agent} (Research Project)"}
        self.min_rows = min_rows
        self.min_density = min_density

        # Thread-safe counters
        self.rejected_429_count = 0
        self.filtered_out_count = 0
        self.other_error_count = 0

        # Global halt lock for 429 management
        self._halt_until = 0
        self._lock = threading.Lock()

    def normalize_cell(self, cell_el, is_header=False):
        if not cell_el: return ""
        if is_header:
            for br in cell_el.find_all(["br", "p"]):
                br.replace_with(" ")
        text = cell_el.get_text().strip()
        text = re.sub(r"\[.*?\]", "", text).strip()
        if not is_header:
            clean_num = text.replace(",", "")
            try:
                if "." in clean_num: return float(clean_num)
                return int(clean_num)
            except ValueError:
                pass
        return text

    def extract_wikipedia_tables(self, session, title, lang="en"):
        retries = 3
        success = False
        r = None

        for attempt in range(retries):
            # Check if global cooldown is active
            wait_time = self._halt_until - time.time()
            if wait_time > 0:
                time.sleep(wait_time)

            try:
                url = f"https://{lang}.wikipedia.org/api/rest_v1/page/html/{title}"
                r = session.get(url, headers=self.headers, timeout=15)

                if r.status_code == 200:
                    success = True
                    break
                elif r.status_code == 429:
                    with self._lock:
                        # Set global halt if not already set by another thread
                        if time.time() > self._halt_until:
                            self._halt_until = time.time() + 40
                    continue  # Retry after sleep
                else:
                    self.other_error_count += 1
                    return []
            except Exception:
                if attempt == retries - 1:
                    self.other_error_count += 1
                    return []
                time.sleep(1)

        # Final check if we failed after retries
        if not success:
            if r and r.status_code == 429:
                self.rejected_429_count += 1
            return []

        soup = BeautifulSoup(r.text, "lxml")
        results = []
        current_hierarchy = {}
        inside_content = False

        for el in soup.find_all(["h2", "h3", "h4", "table"]):
            tag = el.name.lower()
            if tag.startswith("h"):
                level = int(tag[1])
                heading_text = self.normalize_cell(el, is_header=True)
                if not inside_content and level == 2: inside_content = True
                if inside_content and level == 2 and heading_text.lower() in self.EXCLUDE_SECTIONS:
                    inside_content = False
                levels_to_clear = [l for l in current_hierarchy.keys() if l >= level]
                for l in levels_to_clear: del current_hierarchy[l]
                current_hierarchy[level] = heading_text
                continue

            if tag == "table" and inside_content:
                table_data = self._process_table(el)
                if table_data:
                    sorted_levels = sorted(current_hierarchy.keys())
                    path = " > ".join([current_hierarchy[l] for l in sorted_levels])
                    results.append({"table_name": f"{title} > {path}", "rows": table_data})

        if not results:
            self.filtered_out_count += 1
        return results

    def _process_table(self, table_el):
        all_tr = table_el.find_all("tr")
        if not all_tr: return None
        header_row = all_tr[0]
        ths = header_row.find_all(["th", "td"])
        headers = [self.normalize_cell(th, is_header=True) for th in ths]
        if len(headers) < 2 or all(not h for h in headers): return None

        rows_raw = []
        for tr in all_tr[1:]:
            cells = tr.find_all(["td", "th"])
            if len(cells) != len(headers): continue
            row_vals = [self.normalize_cell(c, is_header=False) for c in cells]
            if row_vals == headers: continue
            rows_raw.append(row_vals)

        if not rows_raw: return None
        active_indices = [i for i in range(len(headers)) if
                          headers[i] and any(str(row[i]).strip() != "" for row in rows_raw)]

        final_rows = []
        filled_cells = 0
        for row in rows_raw:
            structured_row = {}
            for i in active_indices:
                val = row[i]
                structured_row[headers[i]] = val
                if str(val).strip() != "": filled_cells += 1
            final_rows.append(structured_row)

        if len(final_rows) < self.min_rows: return None
        density = filled_cells / (len(final_rows) * len(active_indices))
        return final_rows if density >= self.min_density else None


class WikiExtractor:
    def __init__(self, user_agent="TableResearch/1.0", max_workers=15):
        self.extractor = TableExtractor(user_agent)
        self.max_workers = max_workers

    def bulk_extract(self, titles):
        all_extracted_tables = []
        pages_with_data = 0
        total_pages = len(titles)

        print(f"Starting Extraction: {total_pages} titles | {self.max_workers} concurrent threads.")

        with requests.Session() as session:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_title = {
                    executor.submit(self.extractor.extract_wikipedia_tables, session, t): t
                    for t in titles
                }

                # Setup tqdm with a postfix to show real-time status
                pbar = tqdm(
                    as_completed(future_to_title),
                    total=total_pages,
                    desc="Extracting Tables",
                    unit="page"
                )

                for future in pbar:
                    try:
                        tables = future.result()

                        # Determine current status for the UI
                        current_time = time.time()
                        is_halted = current_time < self.extractor._halt_until
                        status_msg = "HALTED (429)" if is_halted else "RUNNING"

                        if tables:
                            all_extracted_tables.extend(tables)
                            pages_with_data += 1

                        # Update the progress bar with live stats
                        pbar.set_postfix({
                            "Status": status_msg,
                            "Success": pages_with_data,
                            "429s": self.extractor.rejected_429_count
                        })

                    except Exception as e:
                        # Optional: tqdm.write(f"Unexpected error: {e}")
                        pass

        self._print_extraction_summary(total_pages, pages_with_data, len(all_extracted_tables))
        return all_extracted_tables

    def _print_extraction_summary(self, total_in, pages_hit, table_total):
        rejections = self.extractor.rejected_429_count
        filtered = self.extractor.filtered_out_count
        others = self.extractor.other_error_count

        print("\n" + "=" * 60)
        print("FINAL EXTRACTION SUMMARY")
        print("-" * 60)
        print(f"{'Total Titles Input:':<50} | {total_in}")
        print(f"{'Pages with Successfully Saved Tables:':<50} | {pages_hit}")
        print(f"{'Pages with No Tables Found / Filtered Out:':<50} | {filtered}")
        print(f"{'Rejected due to Rate Limit (429):':<50} | {rejections}")
        if others > 0:
            print(f"{'Other Network Errors:':<50} | {others}")
        print("-" * 60)
        print(f"{'TOTAL TABLES COLLECTED:':<50} | {table_total}")

        success_rate = 100 * (pages_hit / total_in) if total_in > 0 else 0
        print(f"{'SUCCESS RATE:':<50} | {success_rate:.1f}%")
        print("=" * 60 + "\n")