import re
import time
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed


class WikiExtractor:
    def __init__(self, user_agent="TableResearch/1.0", max_workers=15, delay=0.2):
        self.extractor = TableExtractor(user_agent)  # Using your existing class
        self.max_workers = max_workers
        self.delay = delay

    def bulk_extract(self, titles):
        """
        Takes a list of titles, extracts tables in parallel,
        and returns a flattened list of all tables.
        """
        all_extracted_tables = []
        pages_with_tables_count = 0
        total_pages = len(titles)

        print(f"Starting Extraction Phase: Processing {total_pages} pages using {self.max_workers} threads...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Map titles to the extraction function
            future_to_title = {executor.submit(self.extractor.extract_wikipedia_tables, title): title for title in
                               titles}

            # tqdm context for tracking progress
            for future in tqdm(as_completed(future_to_title), total=total_pages, desc="Extracting Tables", unit="page"):
                title = future_to_title[future]
                try:
                    tables = future.result()
                    if tables:
                        all_extracted_tables.extend(tables)
                        pages_with_tables_count += 1
                except:
                    # Individual page failure shouldn't stop the whole run
                    pass

                    # Small optional delay to be polite to the API
                time.sleep(self.delay)

        self._print_extraction_summary(total_pages, pages_with_tables_count, len(all_extracted_tables))
        return all_extracted_tables

    def _print_extraction_summary(self, total_in, pages_hit, table_total):
        print("\n" + "=" * 40)
        print("EXTRACTION SUMMARY")
        print("-" * 40)
        print(f"{'Total Pages Input:':<25} {total_in}")
        print(f"{'Pages Containing Tables:':<25} {pages_hit}")
        print(f"{'Total Tables Found:':<25} {table_total}")

        hit_rate = (pages_hit / total_in * 100) if total_in > 0 else 0
        print(f"{'Page Success Rate:':<25} {hit_rate:.1f}%")
        print("=" * 40 + "\n")


class TableExtractor:
    EXCLUDE_SECTIONS = {"references", "see also", "external links", "notes", "further reading"}

    def __init__(self, user_agent="TableExtractor/1.0", min_rows=3, min_density=0.8):
        self.headers = {"User-Agent": user_agent}
        self.min_rows = min_rows
        self.min_density = min_density

    def normalize_cell(self, cell_el, is_header=False):
        """Replaces <br> with spaces and cleans citations/numbers."""
        if not cell_el or cell_el is None:
            return ""

        # FIX 1: Handle multi-line headers by replacing <br> tags with spaces
        if is_header:
            # We work on a copy to avoid destroying the soup for other logic
            for br in cell_el.find_all(["br", "p"]):
                br.replace_with(" ")

        text = cell_el.get_text().strip()
        text = re.sub(r"\[.*?\]", "", text).strip()  # Remove [1], [a], etc.

        # Try numeric conversion for data cells
        if not is_header:
            clean_num = text.replace(",", "")
            try:
                if "." in clean_num: return float(clean_num)
                return int(clean_num)
            except ValueError:
                pass
        return text

    def extract_wikipedia_tables(self, title, lang="en"):
        retries = 3
        backoff = 2  # Seconds to wait after a rate limit hit

        for attempt in range(retries):
            try:
                url = f"https://{lang}.wikipedia.org/api/rest_v1/page/html/{title}"
                r = requests.get(url, headers=self.headers, timeout=15)

                if r.status_code == 429:
                    # We hit the rate limit! Wait and try again.
                    time.sleep(backoff * (attempt + 1))
                    continue

                r.raise_for_status()
                break  # Success! Break the retry loop.

            except Exception:
                if attempt == retries - 1: return []  # Final failure
                time.sleep(1)

        soup = BeautifulSoup(r.text, "lxml")
        results = []
        current_hierarchy = {}
        inside_content = False

        for el in soup.find_all(["h2", "h3", "h4", "table"]):
            tag = el.name.lower()

            if tag.startswith("h"):
                level = int(tag[1])
                # PASS THE ELEMENT, NOT TEXT
                heading_text = self.normalize_cell(el, is_header=True)

                if not inside_content and level == 2:
                    inside_content = True
                if inside_content and level == 2 and heading_text.lower() in self.EXCLUDE_SECTIONS:
                    inside_content = False

                # FIX 2: Clear hierarchy correctly to avoid 'Alberta > Manitoba'
                levels_to_clear = [l for l in current_hierarchy.keys() if l >= level]
                for l in levels_to_clear:
                    del current_hierarchy[l]
                current_hierarchy[level] = heading_text
                continue

            if tag == "table" and inside_content:
                table_data = self._process_table(el)
                if table_data:
                    sorted_levels = sorted(current_hierarchy.keys())
                    path = " > ".join([current_hierarchy[l] for l in sorted_levels])
                    results.append({
                        "table_name": f"{title} > {path}",
                        "rows": table_data
                    })
        return results

    def _process_table(self, table_el):
        all_tr = table_el.find_all("tr")
        if not all_tr: return None

        # 1. Header Detection (passing element to normalize_cell)
        header_row = all_tr[0]
        ths = header_row.find_all(["th", "td"])
        headers = [self.normalize_cell(th, is_header=True) for th in ths]

        if len(headers) < 2 or all(not h for h in headers): return None

        # 2. Data Extraction
        rows_raw = []
        for tr in all_tr[1:]:
            cells = tr.find_all(["td", "th"])
            if len(cells) != len(headers): continue
            # PASS THE ELEMENT
            row_vals = [self.normalize_cell(c, is_header=False) for c in cells]
            if row_vals == headers: continue
            rows_raw.append(row_vals)

        if not rows_raw: return None

        # 3. Filter Columns and Build Rows
        active_indices = [i for i in range(len(headers)) if
                          headers[i] and any(str(row[i]).strip() != "" for row in rows_raw)]

        final_rows = []
        filled_cells = 0
        for row in rows_raw:
            structured_row = {}
            for i in active_indices:
                val = row[i]
                structured_row[headers[i]] = val
                if str(val).strip() != "":
                    filled_cells += 1
            final_rows.append(structured_row)

        # 4. Final Quality Checks
        if len(final_rows) < self.min_rows:
            return None

        density = filled_cells / (len(final_rows) * len(active_indices))
        if density < self.min_density:
            return None

        return final_rows