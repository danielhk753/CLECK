# class TableValidator:
#
#     @staticmethod
#     def validate_and_extract(table_el, hierarchy_path, title) -> Optional[Dict]:
#
#         # --- NEW: Enhanced External Title Capture ---
#         external_caption = ""
#
#         # 1. First, check for an internal <caption> tag (standard HTML)
#         caption_tag = table_el.find("caption")
#         if caption_tag:
#             external_caption = caption_tag.get_text(" ", strip=True)
#
#         # 2. If no caption, look at the paragraph or div right above the table
#         if not external_caption:
#             # Look at the 2 elements immediately preceding the table
#             for sibling in table_el.find_previous_siblings(limit=2):
#                 if sibling.name in ["p", "div", "h4"]:
#                     # Look for bold tags which usually signify a title
#                     bold_text = sibling.find(["b", "strong"])
#                     if bold_text:
#                         external_caption = bold_text.get_text(" ", strip=True)
#                         break
#                     # If the whole paragraph is short and not bolded, it might still be a title
#                     elif len(sibling.get_text(strip=True)) < 100:
#                         external_caption = sibling.get_text(" ", strip=True)
#                         break
#
#         # Clean up the caption (remove Wikipedia [edit] links or citations)
#         external_caption = re.sub(r"\[.*?\]", "", external_caption).strip()
#
#         # Combine: Page Title > Section > Sub-title
#         full_table_path = f"{title} > {hierarchy_path}"
#         if external_caption:
#             full_table_path += f" ({external_caption})"
#         # --------------------------------------------
#
#         all_tr = table_el.find_all("tr")
#         if not all_tr: return None
#
#         header_row = all_tr[0]
#         ths = header_row.find_all(["th", "td"])
#         headers = [TableValidator.normalize_cell(th, is_header=True) for th in ths]
#
#         if len(headers) < 2 or all(not h for h in headers): return None
#
#         # Create a small helper inside the function or class
#         def get_clean_int(el, attr, default=1):
#             val = el.get(attr, str(default))
#             try:
#                 # 1. Strip whitespace and common trailing characters like ; or style tags
#                 # 2. Extract just the first sequence of digits
#                 match = re.search(r'\d+', str(val))
#                 return int(match.group()) if match else default
#             except (ValueError, TypeError):
#                 return default
#
#         rows_raw = []
#         # This tracks: { (row_index, col_index): value_to_inject }
#         rowspan_tracker = {}
#
#         for r_idx, tr in enumerate(all_tr[1:]):
#             cells = tr.find_all(["td", "th"])
#             current_row_vals = []
#             html_cell_idx = 0
#
#             # We iterate up to the expected number of headers
#             for c_idx in range(len(headers)):
#                 # 1. Check if a previous row "owns" this spot via rowspan
#                 if (r_idx, c_idx) in rowspan_tracker:
#                     current_row_vals.append(rowspan_tracker[(r_idx, c_idx)])
#                     continue
#
#                 # 2. Otherwise, pull from the actual HTML cells
#                 if html_cell_idx < len(cells):
#                     cell = cells[html_cell_idx]
#                     val = TableValidator.normalize_cell(cell, is_header=False)
#
#                     # Handle Rowspan: Mark future rows as occupied
#                     rowspan = get_clean_int(cell, "rowspan", 1)
#                     colspan = get_clean_int(cell, "colspan", 1)
#                     if rowspan > 1:
#                         for offset in range(1, rowspan):
#                             rowspan_tracker[(r_idx + offset, c_idx)] = val
#
#                     # Handle Colspan: Also expand current row horizontally
#                     for _ in range(colspan):
#                         current_row_vals.append(val)
#
#                     html_cell_idx += 1
#                 else:
#                     # Pad with empty if the row is unexpectedly short
#                     current_row_vals.append("")
#
#             # Cleanup: Only keep the values that align with headers
#             # (Truncate if colspan made it too long)
#             current_row_vals = current_row_vals[:len(headers)]
#
#             if current_row_vals == headers: continue
#             rows_raw.append(current_row_vals)
#
#         if not rows_raw: return None
#
#         active_indices = [i for i in range(len(headers)) if
#                           headers[i] and any(str(row[i]).strip() != "" for row in rows_raw)]
#
#         final_rows = []
#         filled_cells = 0
#         for row in rows_raw:
#             structured_row = {}
#             for i in active_indices:
#                 val = row[i]
#                 structured_row[headers[i]] = val
#                 if str(val).strip() != "": filled_cells += 1
#             final_rows.append(structured_row)
#
#         if len(final_rows) < Config.MIN_ROWS: return None
#
#         # --- SAFETY & CONSTRAINT: 100% First Column Fullness ---
#         # 1. First, make sure we even have rows and columns left
#         if not final_rows or not active_indices:
#             return None
#
#         # 2. Safely get the key name for the first visible column
#         first_col_key = headers[active_indices[0]]
#
#         # 3. Check if every single row has data in that first column
#         is_first_col_complete = all(
#             str(row.get(first_col_key, "")).strip() != ""
#             for row in final_rows
#         )
#
#         if not is_first_col_complete:
#             return None
#         # ------------------------------------------------------
#
#         # Density checks
#         total_slots = len(final_rows) * len(active_indices)
#         density = filled_cells / total_slots if total_slots > 0 else 0
#
#         if density < Config.MIN_DENSITY: return None
#
#         return {
#             "table_name": full_table_path,
#             "rows": final_rows,
#             "density": density
#         }




# class TablePostProcessor:
#     def __init__(self, jsonl_path):
#         self.jsonl_path = jsonl_path
#         self.grouped_data = defaultdict(list)
#         self.stats = {
#             "total_raw": 0,
#             "short_tables": 0,
#             "invalid_cols": 0,
#             "bad_parsing": 0,
#             "long_cells": 0,  # New Stat
#             "unique_pages": 0,
#             "final_count": 0
#         }
#         self.final_sample = []
#
#     @staticmethod
#     def _is_header_sane(header):
#         h_str = str(header).strip()
#         if not h_str: return False
#
#         # 1. Newline Check (The most common sign of merged rows)
#         if "\n" in h_str: return False
#
#         # 2. Word Count Check
#         # Your example has 12 words. Most headers have 1-4.
#         words = h_str.split()
#         if len(words) > 8:  # If a header is more than 8 words, it's likely a sentence or merge
#             return False
#
#         # 3. Special Character Check
#         special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', h_str))
#         if len(h_str) > 0 and (special_chars / len(h_str)) > 0.20:
#             return False
#
#         return True
#
#     def process(self, max_rows=12, max_cell_len=200):
#         """
#         Processes the jsonl file: filters by rows/columns, checks header sanity,
#         and samples the best table per page.
#         """
#         # --- 1. Filter and Group ---
#         with open(self.jsonl_path, 'r', encoding='utf-8') as f:
#             # Count lines for progress bar
#             line_count = sum(1 for _ in open(self.jsonl_path, 'r', encoding='utf-8'))
#             f.seek(0)
#
#             for line in tqdm_lib.tqdm(f, total=line_count, desc="Filtering & Cleaning"):
#                 self.stats["total_raw"] += 1
#                 try:
#                     table_data = json.loads(line)
#                     rows = table_data.get('rows', [])
#
#                     # Check Row Threshold
#                     if not (4 < len(rows) <= max_rows):
#                         continue
#
#                     # Check Column Count (Must be > 1)
#                     if not rows or len(rows[0].keys()) <= 1:
#                         self.stats["invalid_cols"] += 1
#                         continue
#
#                     # Check Header Integrity
#                     headers = list(rows[0].keys())
#                     if not all(self._is_header_sane(h) for h in headers):
#                         self.stats["bad_parsing"] += 1
#                         continue
#
#                     # Check Cell Content Length
#                     # Check every cell in every row for massive text blocks
#                     is_too_long = False
#                     for row in rows:
#                         if any(len(str(value)) > max_cell_len for value in row.values()):
#                             is_too_long = True
#                             break
#
#                     if is_too_long:
#                         self.stats["long_cells"] += 1
#                         continue
#
#                     # If passes all checks
#                     self.stats["short_tables"] += 1
#                     page_title = table_data['table_name'].split(' > ')[0]
#                     self.grouped_data[page_title].append(table_data)
#
#                 except (json.JSONDecodeError, KeyError, IndexError):
#                     continue
#
#         # --- 2. Smart Selection (Best Table Per Page) ---
#         page_titles = list(self.grouped_data.keys())
#
#         for title in tqdm_lib.tqdm(page_titles, desc="Selecting Best Tables"):
#             available_tables = self.grouped_data[title]
#             # Select table with maximum rows (highest info density)
#             selected_table = max(available_tables, key=lambda x: len(x['rows']))
#             self.final_sample.append(selected_table)
#
#         self.stats["unique_pages"] = len(self.grouped_data)
#         self.stats["final_count"] = len(self.final_sample)
#
#         self._print_summary(max_rows)
#         return self.final_sample
#
#     def _print_summary(self, threshold):
#         """
#         Prints the summary in the requested ASCII block style.
#         """
#         width = 44
#         print("\n" + "═" * width)
#         print(f"📊 TABLE PROCESSING SUMMARY (Limit: {threshold})")
#         print("═" * width)
#         print(f"Total tables scanned:      {self.stats['total_raw']}")
#         print(f"Dropped (Single Col):      {self.stats['invalid_cols']}")
#         print(f"Dropped (Bad Headers):     {self.stats['bad_parsing']}")
#         print(f"Dropped (Cell too long):   {self.stats['long_cells']}")
#         print(f"Short tables found:        {self.stats['short_tables']}")
#         print(f"Unique source pages:       {self.stats['unique_pages']}")
#         print(f"Final sampled (1/page):    {self.stats['final_count']}")
#         print("═" * width)
#
#         print("\n" + "-" * 150)
#         print(" " * 66 + "↓ EXAMPLE TABLES ↓" + " " * 66)
#         print("-" * 150 + "\n")
#
#         for s in random.sample(self.final_sample, 4):
#             print("\n" + "═" * 150)
#             print(s['table_name'])
#             print("═" * 150)
#             display(pd.DataFrame(s['rows']))





# class GroupValidator:
#     def __init__(self, input_tables, grouping_prompt=VALIDATE_PROMPT):
#         self.input_tables = input_tables
#         self.model = gptAPI(grouping_prompt,
#                             output_format='bool',
#                             temperature=0.3,
#                             max_concurrent_requests=100,
#                             max_rpm=4900)
#         self.stats = {
#             "total_tables": len(self.input_tables),
#             "unique_after_clustering": 0,
#             "valid_tables": 0,
#         }
#         self.valid_tables = []
#         self.bad_examples = []
#
#     async def process(self):
#         """Main pipeline: Cluster -> Validate -> Summarize"""
#         print(f"Starting pipeline for {self.stats['total_tables']} tables...")
#
#         # 1. Deduplicate/Cluster similar names
#         self.filter_similar_groups()
#
#         # 2. Run LLM Validation
#         await self.validate()
#
#         # 3. Print results
#         self._print_summary()
#         return self.valid_tables
#
#     def filter_similar_groups(self, threshold=0.8):
#         """Clusters group names and keeps only one representative per cluster."""
#         if not self.input_tables:
#             return
#
#         names = [table['Group name'] for table in self.input_tables]
#
#         # Vectorize the names using TF-IDF
#         vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
#         tfidf_matrix = vectorizer.fit_transform(names)
#
#         # Calculate Cosine Similarity
#         cosine_sim = cosine_similarity(tfidf_matrix)
#
#         # Clustering: Cluster items that are more than 'threshold' similar
#         # (Distance is 1 - similarity)
#         clustering = AgglomerativeClustering(
#             n_clusters=None,
#             distance_threshold=1 - threshold,
#             metric='precomputed',
#             linkage='complete'
#         ).fit(1 - cosine_sim)
#
#         # Keep only the first index of each cluster
#         unique_indices = []
#         visited_clusters = set()
#         for i, label in enumerate(clustering.labels_):
#             if label not in visited_clusters:
#                 unique_indices.append(i)
#                 visited_clusters.add(label)
#
#         self.input_tables = [self.input_tables[i] for i in unique_indices]
#         self.stats["unique_after_clustering"] = len(self.input_tables)
#
#     async def validate(self):
#         """Filter self.input_tables based on LLM True/False results"""
#         # We use the 'table_name' (which contains the group info) for validation
#         model_inputs = [table['Group name'] for table in self.input_tables]
#
#         # Get boolean mask from LLM
#         results = await self.model.async_run(model_inputs)
#
#         # Apply mask
#         self.valid_tables = []
#         self.bad_examples = []
#
#         for table, is_valid in zip(self.input_tables, results):
#             if is_valid:
#                 self.valid_tables.append(table)
#             else:
#                 self.bad_examples.append(table['Group name'])
#
#         self.stats["valid_tables"] = len(self.valid_tables)
#
#     def _print_summary(self):
#         width = 50
#         print("\n" + "═" * width)
#         print("📊 GROUP VALIDATION SUMMARY")
#         print("═" * width)
#         print(f"Total Tables Processed:    {self.stats['total_tables']}")
#         print(f"Unique Clusters:           {self.stats['unique_after_clustering']}")
#         print(f"LLM Validated 'Natural':   {self.stats['valid_tables']}")
#         print(f"Discarded as 'Junk':       {len(self.bad_examples)}")
#         print("═" * width)
#
#         if self.bad_examples:
#             print("\n❌ Discarded Examples (First 5):")
#             for ex in self.bad_examples[:5]:
#                 print(f" - {ex}")
#         print("-" * width + "\n")