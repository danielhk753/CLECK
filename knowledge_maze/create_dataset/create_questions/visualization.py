# QUESTION_PROMPT = """
# ### ROLE
# You are an expert Data Journalist. Your task is to generate a natural, professional question based on a dataset.
#
# ### NATURAL LANGUAGE CONSTRAINTS
# - **THE HIERARCHY RULE**: Always prioritize the "Group Name" over the "Wikipedia Page Title." The Page Title is the background, but the Group Name is the specific subject you are reporting on.
#     * *DATA*: Page: "COVID-19 testing", Group: "Campus screening strategies"
#     * *WRONG*: "Which country tested first?" (Uses Page Title context)
#     * *RIGHT*: "Which **campus screening strategy** was implemented first?" (Uses Group Name context)
#
# - **SCOPE DEFINITION**: You MUST include the timeframe, category, or specific condition mentioned in the Group Name, even if it applies to every row. Without this, the question has an "undefined scope."
#     * *WRONG*: "Who was the winner?"
#     * *RIGHT*: "Who was the winner of the **2024 Rugby Championship**?"
#
# - **ENTITY MATCHING**: Look at the rows. If the entities are Universities, the question must ask about "universities" or "campuses," not "countries" or "cities."
#
# - **IMPLICIT SUBJECT**: If the table is about a specific entity (e.g., The US National Team), you must name that entity so the question is self-contained for a third party.
#     * *RIGHT*: "How many points did **the United States** win by against Chile?"
#
# - **NO SPOILERS & NO POINTERS**: Do not mention "the table" and do not include the answer in the question.
#
# - **SCOPE INTEGRITY**: The question must cover the specific boundaries of the "Group Name" without over-promising the result.
#     * *EXAMPLE*: If the Group Name is "Proposed Infrastructure Projects," do not ask "Which project was built?"; ask "Which **proposed** project...".
#
# ### TASK GUIDELINES
# 1. **STRATEGY**: Apply the {strategy_name} logic.
# 2. **COLUMN FOCUS**: You are encouraged to use the '{focus_column}' column, but you may use any columns necessary to satisfy the strategy.
# 3. **SELF-CONTAINED**: The question must make sense on its own without needing to see the table first, but it must not be wordy.
# 4. **DATA INTEGRITY**: The 'data_column' field in your output must contain the exact header name(s) used to answer the question.
#
# ### OUTPUT FORMAT (Strict JSON)
# {
#   "thought_process": "Explain why this question is natural and how it uses the relevant columns.",
#   "question": "The natural language question",
#   "data_column": "The exact header name used (or a list of strings if multiple columns are required)"
# }
# """

# class StrategySelector:
#     def __init__(self, sanitized_tables):
#         self.data = sanitized_tables
#         self.updated_data = []
#
#     class StrategyContext:
#         """
#         A helper class to calculate and store table statistics once,
#         so we don't re-compute them for every strategy check.
#         """
#
#         def __init__(self, sample):
#             self.data = sample['Table data - cleaned']
#             self.classes = sample['Column class']
#             self.group_name = sample['Group name'].lower()
#             self.row_count = len(self.data)
#
#             # Column Counts
#             self.total_cols = len(self.classes)
#             self.num_cols = sum(1 for c in self.classes.values() if c == "numerical")
#             self.list_cols = sum(1 for c in self.classes.values() if c == "list")
#             self.time_cols = sum(1 for c in self.classes.values() if c == "temporal")
#             self.ent_cols = sum(1 for c in self.classes.values() if c == "entity")
#
#             # Densities
#             if self.total_cols > 0:
#                 self.num_density = self.num_cols / self.total_cols
#                 self.list_density = self.list_cols / self.total_cols
#                 self.temp_density = self.time_cols / self.total_cols
#             else:
#                 self.num_density = self.list_density = self.temp_density = 0
#
#             # Boolean Flags
#             self.has_num = self.num_cols > 0
#             self.has_list = self.list_cols > 0
#             self.has_time = self.time_cols > 0
#             self.has_entity = self.ent_cols > 0
#
#     def get_top_strategies(self, data_sample):
#         ctx = self.StrategyContext(data_sample)
#         scores = {}
#         threshold = 0.45
#
#         # 1. EARLY EXITS
#         if ctx.row_count < 2:
#             return []
#
#         # 2. SPECIAL ROW-COUNT LOGIC (Small tables need specific strategies)
#         if ctx.row_count <= 3:
#             return self._handle_small_tables(ctx) # COMPARATIVE_DIFFERENCE, EXTREME_NUMERICAL, EXTREME_TEMPORAL
#
#         # 3. STRATEGY EVALUATION MODULES
#         self._evaluate_entity_strategies(ctx, scores) # COUNT_TOTAL
#         self._evaluate_categorical_strategies(ctx, scores)  # FREQUENCY_MODE, CATEGORICAL_FILTER_LOOKUP, UNIQUE_DIVERSITY
#         self._evaluate_numerical_strategies(ctx, scores)  # EXTREME_NUMERICAL, AGGREGATION, TOP_K_RANKING, COMPARATIVE_DIFFERENCE
#         self._evaluate_temporal_strategies(ctx, scores)  # EXTREME_TEMPORAL, FILTER_COUNT_TEMPORAL
#
#         # 4. FINAL SELECTION - Filter by threshold
#         valid_strategies = {k: v for k, v in scores.items() if v >= threshold}
#
#         # Sort by score descending and take top 3
#         return sorted(valid_strategies, key=valid_strategies.get, reverse=True)[:3]
#
#     # =========================================================================
#     # Strategy-based functions
#     # =========================================================================
#
#     @staticmethod
#     def _handle_small_tables(ctx):
#         """Logic for tables with 2-3 rows where statistical depth is impossible."""
#         scores = {}
#         if ctx.row_count == 2 and ctx.has_num:
#             scores["COMPARATIVE_DIFFERENCE"] = 1.0
#         elif ctx.row_count == 3:
#             if ctx.has_num:
#                 scores["EXTREME_NUMERICAL"] = 0.9
#                 scores["COMPARATIVE_DIFFERENCE"] = 0.8
#             if ctx.has_time:
#                 scores["EXTREME_TEMPORAL"] = 0.9
#
#         return sorted(scores, key=scores.get, reverse=True)
#
#     @staticmethod
#     def _evaluate_entity_strategies(ctx, scores):
#         """
#         Evaluates: COUNT_TOTAL
#
#         HIGH SCORE REQUIREMENTS - COUNT_TOTAL:
#         - Row count must be >= 4.
#         - Entities must be unique (row count matches unique entity count).
#         - Must NOT be an explicit subset (avoids 'Top', 'Ranked', 'Best').
#         - Must NOT be trivial (row count cannot match a number mentioned in the title).
#         - BOOST: Based on 'Definitive' keywords (Recipients, Members, Timeline, Statistics).
#         """
#         if not ctx.has_entity:
#             return
#
#         # 1. ENTITY UNIQUENESS CHECK
#         # We find the entity column and ensure every row is a different person/thing.
#         ent_col = next((col for col, cls in ctx.classes.items() if cls == "entity"), None)
#         if ent_col:
#             unique_entities = set(str(row.get(ent_col)).lower() for row in ctx.data if row.get(ent_col))
#             # If entities repeat, a "Total Count" question is semantically incorrect.
#             if len(unique_entities) != ctx.row_count:
#                 return
#
#         # 2. MINIMUM VOLUME
#         if ctx.row_count < 4:
#             return
#
#         # 3. TRIVIALITY CHECK
#         # Extract digits to see if the title already tells you the count
#         title_numbers = [int(s) for s in ctx.group_name.split() if s.isdigit()]
#         if any(n == ctx.row_count for n in title_numbers):
#             return
#
#         # 4. QUALITATIVE SUBSET CHECK
#         count_blocking_keywords = {"top", "ranked", "leading", "best", "roster", "squad", "highlights"}
#         if any(word in ctx.group_name for word in count_blocking_keywords):
#             return
#
#         # 5. ASSIGN BASE SCORE
#         scores["COUNT_TOTAL"] = 0.46
#
#         # 6. DEFINITIVE LIST BOOST (Golden Words)
#         # These words imply the table is a full record of a category.
#         golden_words = ["recipients", "members", "participants", "timeline", "statistics", "standings", "results"]
#         if any(word in ctx.group_name for word in golden_words):
#             scores["COUNT_TOTAL"] = 0.6
#
#     @staticmethod
#     def _evaluate_categorical_strategies(ctx, scores):
#         """
#         Evaluates: FREQUENCY_MODE, CATEGORICAL_FILTER_LOOKUP, UNIQUE_DIVERSITY
#
#         HIGH SCORE REQUIREMENTS:
#         - UNIQUE_DIVERSITY: High ratio of unique values (many different categories).
#         - CATEGORICAL_FILTER_LOOKUP: Significant 'clumping' (at least 2 duplicates)
#           and presence of Num/Entity columns for the answer.
#         - FREQUENCY_MODE: At least 3 unique categories, no ties for 1st place,
#           and at least one repetition.
#         """
#         if not ctx.has_list:
#             return
#
#         # Initialize stats we'll gather across all list columns
#         has_clumping = False
#
#         # max_diversity: Tracks the highest ratio of unique categories found across columns.
#         # High diversity (close to 1.0) means almost every row has a unique category label.
#         max_diversity = 0
#
#         # max_concentration: Tracks the highest dominance of a single category.
#         # High concentration means one value (the 'Mode') appears in a large percentage of rows.
#         max_concentration = 0
#
#         # best_freq_score: The highest calculated score for FREQUENCY_MODE based on the
#         # 'Bouncer' rules (3+ categories, no ties, dominance ratio).
#         best_freq_score = 0
#
#         list_col_names = [col for col, cls in ctx.classes.items() if cls == "list"]
#
#         for col in list_col_names:
#             vals = [str(row.get(col)) for row in ctx.data if row.get(col) is not None]
#             if not vals: continue
#
#             counts = Counter(vals)
#             num_total = len(vals)
#             num_unique = len(counts)
#             top_entries = counts.most_common(2)
#
#             # 1. CALCULATE RATIOS
#             unique_ratio = num_unique / num_total
#             max_diversity = max(max_diversity, unique_ratio)
#
#             if top_entries:
#                 concentration_ratio = top_entries[0][1] / num_total
#                 max_concentration = max(max_concentration, concentration_ratio)
#
#             # 2. CLUMPING CHECK (Required for Filter & Frequency)
#             # We need at least 2 duplicates in the column to make it 'clumped'
#             if (num_total - num_unique) >= 2:
#                 has_clumping = True
#
#             # 3. FREQUENCY_MODE BOUNCER
#             # Avoid binary choices (need 3+ types), avoid ties, and ensure repetition exists
#             is_not_tie = len(top_entries) > 1 and top_entries[0][1] > top_entries[1][1]
#
#             if num_unique >= 3 and is_not_tie and (num_total > num_unique):
#                 # We favor columns where the mode is dominant but not 100% of the table
#                 # Score range: 0.85 to 0.95
#                 col_freq_score = 0.85 + (0.1 * (top_entries[0][1] / num_total))
#                 best_freq_score = max(best_freq_score, col_freq_score)
#
#         # --- STRATEGY ASSIGNMENT ---
#
#         # A. UNIQUE_DIVERSITY: The "Variety" Strategy
#         # Score is high if almost every row belongs to a different category.
#         if max_diversity > 0:
#             scores["UNIQUE_DIVERSITY"] = 0.8 + (0.2 * max_diversity)
#
#         # B. CATEGORICAL_FILTER_LOOKUP: The "Grouping" Strategy
#         if has_clumping:
#             # Higher concentration means bigger groups to filter, which is better.
#             scores["CATEGORICAL_FILTER_LOOKUP"] = 0.7 + (0.15 * max_concentration)
#
#             # Boost: Filtering is only useful if there is an Entity or Number to look up
#             if ctx.has_entity or ctx.has_num:
#                 scores["CATEGORICAL_FILTER_LOOKUP"] += 0.1
#
#         # C. FREQUENCY_MODE: The "Dominance" Strategy
#         if best_freq_score > 0:
#             # We give Frequency a slight edge over Diversity if the bouncer passes,
#             # as it usually leads to more interesting "insight" questions.
#             scores["FREQUENCY_MODE"] = 0.8 + 0.2 * best_freq_score
#
#     @staticmethod
#     def _evaluate_numerical_strategies(ctx, scores):
#         """
#         Evaluates: EXTREME_NUMERICAL, AGGREGATION, TOP_K_RANKING, COMPARATIVE_DIFFERENCE, FILTER_COUNT_NUMERICAL
#
#         HIGH SCORE REQUIREMENTS:
#         - Variance: At least one numerical column must have different values.
#         - AGGREGATION: Boosted by 'Total/Sum' keywords in name/headers; penalized in 'Top/Best' subsets.
#         - TOP_K_RANKING: Requires row_count > 6 to be meaningful.
#         - COMPARATIVE_DIFFERENCE: Optimized for small tables (<= 5 rows).
#         """
#         if not ctx.has_num:
#             return
#
#         # has_variance: Logic gate ensuring numerical values aren't uniform.
#         has_variance = False
#         num_col_names = [col for col, cls in ctx.classes.items() if cls == "numerical"]
#
#         for col in num_col_names:
#             try:
#                 vals = [float(row[col]) for row in ctx.data if row.get(col) is not None]
#                 if vals and max(vals) > min(vals):
#                     has_variance = True
#                     break
#             except (ValueError, TypeError):
#                 continue
#
#         if not has_variance:
#             return
#
#         # --- KEYWORD & CONTEXT ANALYSIS ---
#         agg_keywords = {
#             "total", "gross", "income", "revenue", "average", "sum", "gva", "gdp", "profit",
#             "capacity", "amount", "cost", "value", "score", "points", "budget", "salary", "funding"
#         }
#         percent_keywords = {"percent", "%", "share", "rate", "ratio"}
#         subset_keywords = {"top", "highest", "best", "rankings", "leaders", "most", "biggest"}
#
#         name_words = set(ctx.group_name.split())
#
#         # is_agg_context / header_is_agg: Detection of summation-friendly metadata.
#         is_agg_context = any(k in name_words for k in agg_keywords)
#         header_is_agg = any(any(k in col.lower() for k in agg_keywords) for col in num_col_names)
#
#         # is_pct_context / header_is_pct: Detection of rate-based data.
#         is_pct_context = any(k in name_words for k in percent_keywords)
#         header_is_pct = any(any(k in col.lower() for k in percent_keywords) for col in num_col_names)
#
#         # is_subset_context: Identifies if the table is a 'Top X' list.
#         is_subset_context = any(k in name_words for k in subset_keywords)
#
#         # 1. EXTREME & TOP_K
#         scores["EXTREME_NUMERICAL"] = 0.8 + (0.15 * ctx.num_density)
#         scores["FILTER_COUNT_NUMERICAL"] = 0.7 + (0.1 * ctx.num_density)
#
#         # TOP_K_RANKING needs enough rows to actually 'rank'.
#         if ctx.row_count > 4:
#             scores["TOP_K_RANKING"] = 0.8 + (0.15 * ctx.num_density)
#         else:
#             scores["TOP_K_RANKING"] = 0.5
#
#         # 2. AGGREGATION BOUNCER
#         # We only want a high score if it's an agg context and NOT a percentage.
#         if (is_agg_context or header_is_agg) and not (is_pct_context or header_is_pct):
#             scores["AGGREGATION"] = 0.95
#         elif is_subset_context or is_pct_context or header_is_pct:
#             # Aggregating a subset or percentages is usually a bad strategy.
#             scores["AGGREGATION"] = 0.4
#         else:
#             # Default for numerical tables that aren't obviously subsets or percentages.
#             scores["AGGREGATION"] = 0.75
#
#         # 3. COMPARATIVE_DIFFERENCE
#         if ctx.row_count <= 5:
#             scores["COMPARATIVE_DIFFERENCE"] = 0.85
#         else:
#             scores["COMPARATIVE_DIFFERENCE"] = 0.5
#
#     @staticmethod
#     def _evaluate_temporal_strategies(ctx, scores):
#         """
#     Evaluates: EXTREME_TEMPORAL, FILTER_COUNT_TEMPORAL, CATEGORICAL_FILTER_TEMPORAL
#
#     HIGH SCORE REQUIREMENTS:
#     ------------------------
#     1. EXTREME_TEMPORAL (Earliest/Latest):
#        - Requires 'Temporal Variance': At least two different years must exist in the data.
#        - Must pass the 'Triviality Bouncer':
#          - Titles with explicit ranges (e.g., "1990-2005") are penalized.
#          - Titles with a single year (e.g., "2024 Awards") are penalized if all
#            table rows contain only that specific year.
#        - Scores: 0.8+ (Clean Data) vs. 0.2 (Trivial Data).
#
#     2. FILTER_COUNT_TEMPORAL (Frequency of events in a specific time):
#        - Requires high 'Temporal Density' (ratio of time-points to rows).
#        - Remains viable even if the title mentions a range, as individual year
#          counts are rarely revealed in titles.
#        - Scores: 0.70 - 0.90 based on density.
#     """
#         if not ctx.has_time:
#             return
#
#         # 1. TEMPORAL VARIANCE CHECK
#         time_cols = [col for col, cls in ctx.classes.items() if cls == "temporal"]
#         all_years = set()
#         for col in time_cols:
#             all_years.update([str(row.get(col))[:4] for row in ctx.data if row.get(col)])
#
#         if len(all_years) < 2:
#             return  # No point in temporal strategies if everything happened in the same year
#
#         # 2. TEMPORAL BOUNCER (Title Leakage + Context)
#         range_pattern = re.compile(r'(\d{4})[\-–\s]+(?:to[\-–\s]+)?(\d{4})')
#         single_year_pattern = re.compile(r'\b(19|20)\d{2}\b')
#
#         match_range = range_pattern.search(ctx.group_name)
#         match_single = single_year_pattern.search(ctx.group_name)
#
#         is_trivial = False
#
#         # CASE A: Explicit Range (e.g., "1990-2000")
#         if match_range:
#             is_trivial = True
#
#         # CASE B: Single Year Triviality (e.g., Title is "2024 films" and table only has 2024)
#         elif match_single:
#             title_year = match_single.group(0)
#             # If the title year is the ONLY year in the data, it's trivial.
#             if all(year == title_year for year in all_years):
#                 is_trivial = True
#
#         # 3. STRATEGY SCORING
#
#         # EXTREME_TEMPORAL: High penalty if trivial, otherwise high reward.
#         scores["EXTREME_TEMPORAL"] = 0.2 if is_trivial else (0.8 + (0.1 * ctx.temp_density))
#
#         # FILTER_COUNT_TEMPORAL: "How many entries in [Year]?"
#         # We boost this if there's high density (lots of dates to pick from).
#         scores["FILTER_COUNT_TEMPORAL"] = 0.7 + (0.15 * ctx.temp_density)
#
#     # =========================================================================
#     # WRAPPER
#     # =========================================================================
#     def get_strategies(self):
#         initial_count = len(self.data)
#         for table in tqdm_lib.tqdm(self.data, desc="Detecting top strategies"):
#             strategies = self.get_top_strategies(table)
#             if strategies:
#                 self.updated_data.append({
#                     "Group name": table["Group name"],
#                     "Table data - cleaned": table["Table data - cleaned"],
#                     "Table data - original": table["Table data - original"],
#                     "Wikipedia link": table["Wikipedia link"],
#                     "Top strategies": strategies,
#                     "Entity column": next((k for k, v in table['Column class'].items() if v == 'entity'), None)
#                 })
#
#         # Summary Printing:
#         passed = len(self.updated_data)
#         print("\n" + "═" * 55)
#         print(f"📊 STRATEGY SELECTION SUMMARY")
#         print("-" * 55)
#         print(f"✅ Groups Passed:     {passed}")
#         print(f"❌ Groups Discarded:  {initial_count - passed}")
#         print(f"📈 Success Rate:      {(passed / initial_count * 100) if initial_count else 0:.1f}%")
#         print("═" * 55 + "\n")


# class StrategySelector:
#     def __init__(self, sanitized_tables):
#         self.data = sanitized_tables
#         self.updated_data = []
#
#     @staticmethod
#     def get_top_strategies(data_sample):
#         data = data_sample['Table data - cleaned']
#         classes = data_sample['Column class']
#         group_name = data_sample['Group name'].lower()
#         row_count = len(data)
#         threshold = 0.45
#
#         if row_count < 2: return []
#
#         # 1. COLUMN DENSITY
#         total_cols = len(classes)
#         num_cols = sum(1 for c in classes.values() if c == "numerical")
#         list_cols = sum(1 for c in classes.values() if c == "list")
#         time_cols = sum(1 for c in classes.values() if c == "temporal")
#         ent_cols = sum(1 for c in classes.values() if c == "entity")
#
#         if total_cols != 0:
#             num_density = num_cols / total_cols
#             list_density = list_cols / total_cols
#             temp_density = time_cols / total_cols
#         else:
#             num_density, list_density, temp_density = 0, 0, 0
#
#         has_list, has_num, has_time, has_entity = list_cols > 0, num_cols > 0, time_cols > 0, ent_cols > 0
#         scores = {}
#
#         # --- 2. ROW COUNT CONSTRAINTS ---
#         if row_count == 2:
#             if has_num: scores["COMPARATIVE_DIFFERENCE"] = 1.0
#             return [k for k, v in scores.items() if v >= threshold]
#
#         if row_count == 3:
#             if has_num:
#                 scores["EXTREME_NUMERICAL"] = 0.9
#                 scores["COMPARATIVE_DIFFERENCE"] = 0.8
#             if has_time: scores["EXTREME_TEMPORAL"] = 0.9
#             return sorted([k for k, v in scores.items() if v >= threshold], key=lambda x: scores[x], reverse=True)
#
#         # --- 3. LIST DIVERSITY & CONCENTRATION CHECK ---
#         diversity_score = 0
#         concentration_score = 0
#         has_clumping = False  # New: Are there enough duplicates to be interesting?
#         max_unique_found = 0
#
#         if has_list:
#             list_col_names = [col for col, cls in classes.items() if cls == "list"]
#             for col in list_col_names:
#                 vals = [str(row.get(col)) for row in data if row.get(col) is not None]
#                 if not vals: continue
#
#                 counts = Counter(vals)
#                 num_total = len(vals)
#                 current_unique = len(counts)
#                 max_unique_found = max(max_unique_found, current_unique)
#
#                 # Check for the "at least 2 duplicates" rule
#                 if (num_total - current_unique) >= 2:
#                     has_clumping = True
#
#                 unique_ratio = current_unique / num_total
#                 top_entries = counts.most_common(2)
#                 most_common_ratio = top_entries[0][1] / num_total
#
#                 # --- FREQUENCY_MODE BOUNCER LOGIC ---
#                 # 1. Must have at least 3 distinct categories (unique_vals >= 3)
#                 # 2. Must have repetition (num_total > current_unique)
#                 # 3. Must NOT have a tie for the top spot
#                 is_not_tie = len(top_entries) > 1 and top_entries[0][1] > top_entries[1][1]
#
#                 if current_unique >= 3 and is_not_tie and (num_total > current_unique):
#                     # Boost score for a healthy, non-binary, non-tied frequency
#                     scores["FREQUENCY_MODE"] = 0.85 + (0.1 * most_common_ratio)
#                 else:
#                     # Table is likely binary, tied, or unique; kill the strategy
#                     scores["FREQUENCY_MODE"] = 0.1
#
#                 # High unique_ratio -> many different categories (Diversity)
#                 diversity_score = max(diversity_score, unique_ratio)
#                 # High most_common_ratio -> one category dominates (Concentration)
#                 concentration_score = max(concentration_score, most_common_ratio)
#
#         # --- 4. CORE SCORING ---
#
#         # ENTITY LOGIC
#         if has_entity:
#             ent_col_name = next(col for col, cls in classes.items() if cls == "entity")
#             unique_vals = set(str(row.get(ent_col_name)) for row in data)
#
#             # --- NEW: BLOCKING LOGIC FOR COUNT_TOTAL ---
#             blocking_keywords = {"top", "highest", "best", "biggest", "largest", "leading", "most", "ranked"}
#             # Check if any blocking word exists in the group name
#             is_limited_list = any(word in group_name for word in blocking_keywords)
#
#             if len(unique_vals) == row_count and row_count >= 4 and (not is_limited_list):
#                 scores["COUNT_TOTAL"] = 0.5
#
#         # CATEGORICAL LOGIC
#         if has_list:
#             # UNIQUE_DIVERSITY: Just needs more than 1 category to exist
#             if max_unique_found > 1:
#                 scores["UNIQUE_DIVERSITY"] = 0.6 + (0.3 * diversity_score)
#
#             # FREQUENCY and FILTER: Need "clumping" (the duplicates you mentioned)
#             if has_clumping:
#                 scores["CATEGORICAL_FILTER_LOOKUP"] = 0.75 + (0.15 * concentration_score)
#
#             if (has_num or has_entity) and has_clumping:
#                 scores["CATEGORICAL_FILTER_LOOKUP"] = 0.7 + (0.1 * list_density)
#
#         # NUMERICAL LOGIC
#         if has_num:
#
#             num_col_names = [col for col, cls in classes.items() if cls == "numerical"]
#
#             # Check for variance in the first numerical column found
#             # (You can loop if you have multiple, but one is usually enough for a flag)
#             has_variance = False
#             for col in num_col_names:
#                 try:
#                     num_vals = [float(row[col]) for row in data if row.get(col) is not None]
#                     if num_vals and max(num_vals) > min(num_vals):
#                         has_variance = True
#                         break
#                 except (ValueError, TypeError):
#                     continue
#
#             agg_keywords = {
#                 "total", "gross", "income", "revenue", "average", "sum", "gva", "gdp", "profit",
#                 "capacity", "amount", "cost", "value", "score", "points", "budget", "salary", "funding"
#             }
#             percent_keywords = ["percent", "%", "share", "rate", "ratio"]
#
#             # Clean the group name and split into a set of individual words
#             name_words = set(group_name.lower().split())
#
#             is_agg_context = any(k in name_words for k in agg_keywords)
#             header_is_agg = any(
#                 any(k in col.lower() for k in agg_keywords)
#                 for col, cls in classes.items() if cls == "numerical"
#             )
#             is_pct_context = any(k in name_words for k in percent_keywords)
#
#             # Only allow "Extreme" or "Ranking" if the numbers aren't all the same
#             if has_variance:
#                 scores["TOP_K_RANKING"] = (0.8 if row_count > 8 else 0.6) + (0.15 * num_density)
#                 scores["EXTREME_NUMERICAL"] = 0.8 + (0.15 * num_density)
#                 scores["FILTER_COUNT_NUMERICAL"] = 0.7 + (0.1 * num_density)
#
#             if is_agg_context or header_is_agg:
#                 scores["AGGREGATION"] = 0.95
#                 if has_variance: scores["EXTREME_NUMERICAL"] += 0.05
#
#             if is_pct_context and has_variance:
#                 scores["TOP_K_RANKING"] += 0.1
#                 scores["EXTREME_NUMERICAL"] += 0.05
#
#             if has_variance:
#                 # High score for small tables (where it's a perfect fit)
#                 if row_count <= 6:
#                     scores["COMPARATIVE_DIFFERENCE"] = 0.85
#                 else:
#                     # Lower but valid score for large tables
#                     scores["COMPARATIVE_DIFFERENCE"] = 0.65
#
#         # TEMPORAL LOGIC
#         if has_time:
#             # Fix 2: Boost based on temp_density
#             scores["EXTREME_TEMPORAL"] = 0.8 + (0.15 * temp_density)
#             scores["FILTER_COUNT_TEMPORAL"] = 0.75 + (0.1 * temp_density)
#
#         # 5. FINAL FILTER
#         valid_strategies = {k: v for k, v in scores.items() if v >= threshold}
#         return sorted(valid_strategies, key=valid_strategies.get, reverse=True)[:3]
#
#     def get_strategies(self):
#         initial_count = len(self.data)
#
#         for table in tqdm_lib.tqdm(self.data, desc="Detecting top strategies"):
#             strategies = self.get_top_strategies(table)
#
#             if len(strategies) > 0:
#                 self.updated_data.append({
#                     "Group name": table["Group name"],
#                     "Table data - cleaned": table["Table data - cleaned"],
#                     "Table data - original": table["Table data - original"],
#                     "Wikipedia link": table["Wikipedia link"],
#                     "Top strategies": strategies,
#                     "Entity column": next((k for k, v in table['Column class'].items() if v == 'entity'), None)
#                 })
#
#         # Fix 3: Summary Report
#         passed = len(self.updated_data)
#         discarded = initial_count - passed
#
#         print("\n" + "═" * 55)
#         print(f"📊 STRATEGY SELECTION SUMMARY")
#         print("-" * 55)
#         print(f"✅ Groups Passed:     {passed}")
#         print(f"❌ Groups Discarded:  {discarded} (No valid strategies)")
#         print(f"📈 Success Rate:      {(passed / initial_count) * 100:.1f}%")
#         print("═" * 55 + "\n")

import re


# class WikiTableCleaner:
#     def __init__(self):
#         # 1. Open/Subjective Set Triggers (Reject these tables)
#         self.open_set_triggers = [
#             r'\bnotable\b', r'\bfamous\b', r'\bselected\b',
#             r'\bmajor\b', r'\bexamples?\b', r'\bpartial\b',
#             r'\bsome\b', r'\bvarious\b'
#         ]
#
#         # 2. Ranking Keywords (Trigger "Top X" naming)
#         self.ranking_triggers = [
#             'highest', 'lowest', 'most', 'least', 'best', 'worst', 'highest-grossing',
#             'top', 'largest', 'smallest', 'earliest', 'latest', 'biggest'
#         ]
#
#         # 3. High-Confidence Entity Headers
#         self.universal_priority = [
#             'name', 'title', 'airline', 'team', 'character', 'honor',
#             'mayor', 'organization', 'mission', 'planet', 'film', 'company',
#             'artist', 'album', 'song', 'work', 'recipient', 'location', 'city',
#             'species', 'book', 'game', 'series'
#         ]
#
#         # 4. Kill List (Statistical/Metadata columns to ignore)
#         self.kill_list = [
#             'rank', 'gross', 'collection', 'revenue', 'year', 'date',
#             'pos', 'notes', 'ref', 'no.', 'total', 'index', 'score',
#             'points', 'won', 'lost', 'drawn', 'c.', 'id'
#         ]
#
#     def _is_textual(self, value):
#         """
#         Enforces 100% textual entities.
#         - Must contain at least one letter [a-zA-Z].
#         - Must NOT be a pure number, date, or special char string.
#         """
#         val_str = str(value).strip()
#
#         # Reject empty/null
#         if not val_str or val_str.lower() in ["none", "null", "n/a", "-", "—", "unknown", "tba", "tbd"]:
#             return False
#
#         # Must contain at least one letter (Rejects "1999", "1,000", "$500", "---")
#         if not re.search(r'[a-zA-Z]', val_str):
#             return False
#
#         # Reject Date Formats (YYYY-MM-DD, DD/MM/YYYY, etc.)
#         if re.match(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$', val_str):
#             return False
#
#         return True
#
#     def _get_entity_column(self, rows, context_str):
#         """Scans headers to find the best candidate for the Entity Name."""
#         if not rows: return None
#         headers = list(rows[0].keys())
#         context = context_str.lower()
#
#         scores = {}
#         for h in headers:
#             h_low = h.lower().strip()
#
#             # Immediate rejection
#             if any(k == h_low or f" {k}" in h_low or f"{k} " in h_low for k in self.kill_list):
#                 continue
#
#             score = 0
#             # 1. Exact Match in Priority List
#             if h_low in self.universal_priority:
#                 score += 50
#
#             # 2. Singular Match in Page Title (e.g., Title: "Mayors", Header: "Mayor")
#             singular = h_low.rstrip('s')
#             if len(singular) > 2 and singular in context:
#                 score += 40
#
#             # 3. Position Bias (Leftmost columns are usually entities)
#             score += (len(headers) - headers.index(h)) * 2
#
#             scores[h] = score
#
#         if not scores: return None
#         return max(scores, key=scores.get)
#
#     # def _generate_clean_name(self, page_title, hierarchy, row_count):
#     #     """Generates a natural language group name with Top-X logic."""
#     #     # Clean the title
#     #     clean_title = re.sub(r'^(List of (the )?|Awards and nominations received by )', '', page_title,
#     #                          flags=re.IGNORECASE).strip()
#     #
#     #     # Clean the hierarchy (remove redundant words found in title)
#     #     title_words = set(clean_title.lower().split())
#     #     clean_hierarchy = []
#     #     for section in hierarchy:
#     #         # Remove parentheses like (1990 film)
#     #         s_clean = re.sub(r'\(.*?\)', '', section).strip()
#     #         # Remove "List of" from section
#     #         s_clean = re.sub(r'^(List of|List)\s+', '', s_clean, flags=re.IGNORECASE).strip()
#     #
#     #         # Check for redundancy
#     #         s_words = s_clean.split()
#     #         unique = [w for w in s_words if w.lower() not in title_words]
#     #
#     #         if unique and s_clean.lower() not in ["members", "current", "former", "list"]:
#     #             clean_hierarchy.append(" ".join(unique))
#     #         elif s_clean.lower() in ["current", "former", "defunct"]:
#     #             clean_hierarchy.append(s_clean)
#     #
#     #     # Assemble Name
#     #     base = f"{' '.join(clean_hierarchy)} {clean_title}".strip()
#     #
#     #     # "Top X" Logic: If name implies ranking, add the count
#     #     if any(t in base.lower() for t in self.ranking_triggers):
#     #         # Only add if a number isn't already there (avoid "Top 10 Top 10")
#     #         if not re.search(r'\b\d+\b', base):
#     #             base = f"Top {row_count} {base}"
#     #
#     #     return base[0].upper() + base[1:]
#
#     def _generate_clean_name(self, page_title, hierarchy, rows, has_rank_col=False):
#         # 1. Standard Title Cleanup
#         clean_title = re.sub(r'^(List of (the )?|Awards and nominations received by )', '',
#                              page_title, flags=re.IGNORECASE).strip()
#
#         # 2. Hierarchy Cleanup
#         title_words = set(clean_title.lower().split())
#         clean_hierarchy = []
#         for section in hierarchy:
#             s_clean = re.sub(r'\(.*?\)', '', section).strip()
#             s_clean = re.sub(r'^(List of|List)\s+', '', s_clean, flags=re.IGNORECASE).strip()
#
#             s_words = s_clean.split()
#             unique = [w for w in s_words if w.lower() not in title_words]
#             if unique and s_clean.lower() not in ["members", "current", "list"]:
#                 clean_hierarchy.append(" ".join(unique))
#             elif s_clean.lower() in ["current", "former", "defunct"]:
#                 clean_hierarchy.append(s_clean)
#
#         # 3. Assemble Name
#         base = f"{' '.join(clean_hierarchy)} {clean_title}".strip()
#
#         # 4. RANK LOGIC: The "Top X" Trigger
#         # Trigger if:
#         # a) Keywords are present (Highest, Largest, etc.) OR
#         # b) A Rank column was detected in the original table
#         is_ranked_list = any(t in base.lower() for t in self.ranking_triggers) or has_rank_col
#
#         if is_ranked_list:
#             # Ensure we don't double-add "Top" if it's already in the string
#             if not re.search(r'\b(Top|Ranked|#)\s*\d+\b', base, re.IGNORECASE):
#                 base = f"Top {rows} {base}"
#
#         return base[0].upper() + base[1:]
#
#     def process_batch(self, raw_samples):
#         """
#         Input: List of raw dicts (from your scraper).
#         Output: List of cleaned, verified dataset dicts.
#         """
#         cleaned_dataset = []
#
#         for sample in raw_samples:
#             # 1. Parse Metadata
#             full_path = sample.get('table_name', '')
#             parts = full_path.split(' > ')
#             page_title = parts[0]
#             hierarchy = parts[1:] if len(parts) > 1 else []
#             rows = sample.get('rows', [])
#
#             # A. Closed Set Check (Reject subjective lists)
#             context = (page_title + " " + " ".join(hierarchy)).lower()
#             if any(re.search(p, context) for p in self.open_set_triggers):
#                 continue  # Skip this sample
#
#             # B. Identify Entity Column
#             entity_col = self._get_entity_column(rows, context)
#             if not entity_col:
#                 continue  # Skip (No clear entity)
#
#             # C. STRICT Textual Validation (The 100% Filter)
#             # If ANY row in the entity column is non-textual, discard the whole table.
#             is_valid_table = True
#             for row in rows:
#                 val = row.get(entity_col)
#                 if not self._is_textual(val):
#                     is_valid_table = False
#                     break
#
#             if not is_valid_table:
#                 continue  # Skip (Dirty data found)
#
#             # D. Generate Final Name
#             group_name = self._generate_clean_name(page_title, hierarchy, len(rows))
#
#             # E. Build Output Object
#             cleaned_dataset.append({
#                 "Group name": group_name,
#                 # Construct a dummy URL or use real one if available in input
#                 "Wikipedia link": f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
#                 "Entity column": entity_col,
#                 "Table data": rows
#             })
#
#         self.print_summary(cleaned_dataset, len(raw_samples))
#
#         return cleaned_dataset
#
#     def print_summary(self, cleaned_dataset, total_raw_count):
#         """
#         Prints a detailed breakdown of the processing run.
#         """
#         passed = len(cleaned_dataset)
#         rejected = total_raw_count - passed
#
#         print("\n" + "=" * 80)
#         print(f"{'WIKI TABLE PROCESSING SUMMARY':^80}")
#         print("=" * 80)
#         print(f"Total Tables Processed:   {total_raw_count}")
#         print(f"Passed (Closed Sets):     {passed}")
#         print(f"Rejected:                 {rejected}")
#
#         if total_raw_count > 0:
#             success_rate = (passed / total_raw_count) * 100
#             print(f"Success Rate:             {success_rate:.1f}%")
#
#         print("-" * 80)
#         print(f"{'EXTRACTED GROUPS':^80}")
#         print("-" * 80)
#
#         for i, item in enumerate(cleaned_dataset[:20], 1):
#             name = item['Group name']
#             col = item['Entity column']
#             rows = len(item['Table data'])
#             print(f"{i:2}. {name:<80} | Column: {col:<20} | Rows: {rows}")
#
#         print("=" * 80 + "\n")

# ------------------------------------------------------------------------------------------------------------------

import re

import re


class WikiTableCleaner:
    def __init__(self):
        # 1. Open/Subjective Set Triggers (Reject these tables)
        self.open_set_triggers = [
            r'\bnotable\b', r'\bfamous\b', r'\bselected\b',
            r'\bmajor\b', r'\bexamples?\b', r'\bpartial\b',
            r'\bsome\b', r'\bvarious\b'
        ]

        # 2. High-Confidence Entity Headers (Used to sort candidates)
        self.universal_priority = [
            'name', 'title', 'airline', 'team', 'character', 'honor',
            'mayor', 'organization', 'mission', 'planet', 'film', 'company',
            'artist', 'album', 'song', 'work', 'recipient', 'location', 'city',
            'species', 'book', 'game', 'series'
        ]

        # 3. Kill List (Statistical/Metadata columns to ignore)
        self.kill_list = [
            'rank', 'gross', 'collection', 'revenue', 'year', 'date',
            'pos', 'notes', 'ref', 'no.', 'total', 'index', 'score',
            'points', 'won', 'lost', 'drawn', 'c.', 'id'
        ]

    def _is_textual(self, value):
        """
        Enforces 100% textual entities.
        - Must contain at least one letter [a-zA-Z].
        - Must NOT be a pure number, date, or special char string.
        """
        val_str = str(value).strip()

        if not val_str or val_str.lower() in ["none", "null", "n/a", "-", "—", "unknown", "tba", "tbd"]:
            return False

        if not re.search(r'[a-zA-Z]', val_str):
            return False

        if re.match(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$', val_str):
            return False

        return True

    def _get_possible_entity_cols(self, rows, context_str):
        """Returns a prioritized list of plausible entity columns."""
        if not rows: return []
        headers = list(rows[0].keys())
        context = context_str.lower()

        candidates = []
        for h in headers:
            h_low = h.lower().strip()

            # Immediate rejection for purely statistical/metadata columns
            if any(k == h_low or f" {k}" in h_low or f"{k} " in h_low for k in self.kill_list):
                continue

            score = 0
            # 1. Exact Match in Priority List
            if h_low in self.universal_priority:
                score += 50

            # 2. Singular Match in Context
            singular = h_low.rstrip('s')
            if len(singular) > 2 and singular in context:
                score += 40

            # 3. Position Bias (Leftmost columns usually hold the entity)
            score += (len(headers) - headers.index(h)) * 2

            candidates.append((h, score))

        # Sort by score descending and return just the column names
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates]

    def process_batch(self, raw_samples):
        """
        Input: List of raw dicts (from your scraper).
        Output: List of cleaned, verified dataset dicts with 3 required keys.
        """
        cleaned_dataset = []

        for sample in raw_samples:
            # 1. Parse Metadata Dictionary
            table_name_dict = sample.get('table_name', {})

            if isinstance(table_name_dict, str):
                header_chain = table_name_dict
                caption = None
                additional_text = None
            else:
                header_chain = table_name_dict.get('header_chain', '')
                caption = table_name_dict.get('caption')
                additional_text = table_name_dict.get('additional_text')

            parts = header_chain.split(' > ')
            page_title = parts[0] if parts else ''
            rows = sample.get('rows', [])

            # Combine text for rigorous context filtering
            context_parts = filter(None, [header_chain, caption, additional_text])
            full_context = " | ".join(context_parts).lower()

            # A. Closed Set Check (Reject subjective lists)
            if any(re.search(p, full_context) for p in self.open_set_triggers):
                continue

            # B. Get Initial Entity Column Candidates
            possible_cols = self._get_possible_entity_cols(rows, full_context)
            if not possible_cols:
                continue

            # C. STRICT Textual Validation Filter
            # Test each candidate column. If ANY row in that column is non-textual, drop the column.
            valid_entity_cols = []
            for col in possible_cols:
                is_valid = True
                for row in rows:
                    if not self._is_textual(row.get(col)):
                        is_valid = False
                        break
                if is_valid:
                    valid_entity_cols.append(col)

            # If no columns survived the 100% textual check, discard the table
            if not valid_entity_cols:
                continue

            # E. Build Final Simplified Output Object
            cleaned_dataset.append({
                "Table context": {
                    "header_chain": header_chain,
                    "caption": caption,
                    "additional_text": additional_text,
                    "possible_entity_cols": valid_entity_cols
                },
                "Wiki link": f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
                "Table data": rows
            })

        self.print_summary(cleaned_dataset, len(raw_samples))

        return cleaned_dataset

    def print_summary(self, cleaned_dataset, total_raw_count):
        """
        Prints a detailed breakdown of the processing run.
        """
        passed = len(cleaned_dataset)
        rejected = total_raw_count - passed

        print("\n" + "=" * 90)
        print(f"{'WIKI TABLE PROCESSING SUMMARY':^90}")
        print("=" * 90)
        print(f"Total Tables Processed:   {total_raw_count}")
        print(f"Passed (Closed Sets):     {passed}")
        print(f"Rejected:                 {rejected}")

        if total_raw_count > 0:
            success_rate = (passed / total_raw_count) * 100
            print(f"Success Rate:             {success_rate:.1f}%")

        print("-" * 90)
        print(f"{'EXTRACTED GROUPS':^90}")
        print("-" * 90)

        for i, item in enumerate(cleaned_dataset[:20], 1):
            title = item['Table context']['header_chain'].split(' > ')[0]
            cols = item['Table context']['possible_entity_cols']
            rows = len(item['Table data'])
            print(f"{i:2}. {title:<40} | Cols: {str(cols):<25} | Rows: {rows}")

        print("=" * 90 + "\n")


import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class RobustWikiExtractor:
    def __init__(self):
        # --- Model Setup (Hugging Face) ---
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"

        # Load tokenizer and model in 4-bit to save VRAM
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            load_in_4bit=True,  # bnb 4-bit quantization
            torch_dtype=torch.float16
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

        # --- Deterministic Filters ---
        self.open_set_triggers = [r'\bnotable\b', r'\bfamous\b', r'\bselected\b', r'\bmajor\b', r'\bexamples?\b']
        self.ranking_triggers = ['highest', 'lowest', 'most', 'least', 'best', 'worst', 'top', 'largest']
        self.kill_list = ['rank', 'gross', 'year', 'date', 'pos', 'notes', 'ref', 'no.', 'index']

    def _is_textual(self, value):
        val_str = str(value).strip()
        return bool(val_str and re.search(r'[a-zA-Z]', val_str) and
                    not re.match(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$', val_str))

    def _get_entity_col(self, rows, context):
        if not rows: return None
        headers = list(rows[0].keys())
        scores = {}
        for h in headers:
            h_low = h.lower().strip()
            if any(k in h_low for k in self.kill_list): continue
            score = 50 if h_low.rstrip('s') in context.lower() else 10
            score += (len(headers) - headers.index(h))
            scores[h] = score
        return max(scores, key=scores.get) if scores else None

    def _llm_rephrase(self, raw_name):
        """Rephrases the raw string using the local 1.5B model."""
        messages = [
            {"role": "system",
             "content": "You are a helpful assistant that rephrases Wikipedia table titles into clean, plural, human-readable group names. Remove redundant words like 'List of', 'And', and junk characters. Output only the name."},
            {"role": "user", "content": f"Rephrase this: {raw_name}"}
        ]

        # Generate
        outputs = self.pipe(messages, max_new_tokens=20, do_sample=False)
        result = outputs[0]['generated_text'][-1]['content'].strip().replace('"', '')

        # Immediate Regex cleanup if the LLM hallucinates "List of"
        result = re.sub(r'^(List of|And|The)\s+', '', result, flags=re.IGNORECASE)
        return result

    def process_batch(self, raw_samples):
        cleaned_dataset = []
        for sample in raw_samples:
            path_parts = sample.get('table_name', '').split(' > ')
            page_title, hierarchy = path_parts[0], path_parts[1:]
            rows = sample.get('rows', [])

            # 1. Verification
            context = (page_title + " " + " ".join(hierarchy)).lower()
            if any(re.search(p, context) for p in self.open_set_triggers): continue

            # 2. Entity Logic
            entity_col = self._get_entity_col(rows, context)
            if not entity_col or not all(self._is_textual(r.get(entity_col)) for r in rows):
                continue

            # 3. Base Naming (Regex Prep)
            raw_group_name = f"{' '.join(hierarchy)} {page_title}"

            # 4. Local LLM Polish
            robust_name = self._llm_rephrase(raw_group_name)

            # 5. Top X Logic
            if any(t in robust_name.lower() for t in self.ranking_triggers):
                if not re.search(r'\d+', robust_name):
                    robust_name = f"Top {len(rows)} {robust_name}"

            cleaned_dataset.append({
                "Group name": robust_name,
                "Wikipedia page": f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
                "Entity column": entity_col,
                "Table data": rows
            })
        return cleaned_dataset

#
# def _plot_two_layered_pie(self):
#     # 1. Define the Mapping
#     mapping = {
#         "Numerical": ["AGGREGATION", "EXTREME_NUMERICAL", "TOP_K_RANKING",
#                       "COMPARATIVE_DIFFERENCE", "FILTER_COUNT_NUMERICAL"],
#         "Categorical": ["UNIQUE_DIVERSITY", "FREQUENCY_MODE", "CATEGORICAL_FILTER_LOOKUP", "COUNT_TOTAL"],
#         "Temporal": ["EXTREME_TEMPORAL", "FILTER_COUNT_TEMPORAL"]
#     }
#
#     # 2. Tally the results
#     strat_counts = Counter([sample['Selected strategy'] for sample in self.valid_results])
#
#     # 3. Organize data for the two layers
#     inner_labels = []
#     inner_sizes = []
#     outer_labels = []
#     outer_sizes = []
#
#     # Define color maps for each group to keep them visually distinct
#     colors_map = {
#         "Numerical": plt.cm.Blues,
#         "Categorical": plt.cm.Reds,
#         "Temporal": plt.cm.Greens
#     }
#     inner_colors = []
#     outer_colors = []
#
#     for group, strats in mapping.items():
#         group_total = sum(strat_counts[s] for s in strats if s in strat_counts)
#         if group_total > 0:
#             inner_labels.append(group)
#             inner_sizes.append(group_total)
#             inner_colors.append(colors_map[group](0.6))
#
#             # Sub-strategies for the outer ring
#             for i, s in enumerate(strats):
#                 if strat_counts[s] > 0:
#                     outer_labels.append(s)
#                     outer_sizes.append(strat_counts[s])
#                     # Gradient effect for sub-strategies
#                     outer_colors.append(colors_map[group](0.4 - (i * 0.05)))
#
#     # 4. Plotting
#     fig, ax = plt.subplots(figsize=(10, 10))
#
#     # Outer Ring (Specific Strategies)
#     ax.pie(outer_sizes, labels=outer_labels, radius=1, colors=outer_colors,
#            wedgeprops=dict(width=0.3, edgecolor='w'), pctdistance=0.85, autopct='%1.1f%%')
#
#     # Inner Ring (Parent Categories)
#     ax.pie(inner_sizes, labels=inner_labels, radius=0.7, colors=inner_colors,
#            labeldistance=0.4, wedgeprops=dict(width=0.3, edgecolor='w'),
#            textprops={'fontsize': 10, 'fontweight': 'bold'})
#
#     ax.set(aspect="equal", title='Strategy Distribution by Data Type')
#     plt.show()


import re
import random

try:
    import spacy
except ImportError:
    raise ImportError("Please run: pip install spacy && python -m spacy download en_core_web_sm")


class GroupMaker:
    def __init__(self, debug=False):
        self.debug = debug

        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "textcat"])
        except OSError:
            raise ImportError("Please run: python -m spacy download en_core_web_sm")

        self.kill_list = {
            'rank', 'ranking', 'gross', 'collection', 'revenue', 'year', 'date', 'born', 'died',
            'pos', 'notes', 'ref', 'no.', 'no', 'total', 'index', 'score', '#',
            'points', 'pts', 'won', 'lost', 'drawn', 'draws', 'wins', 'losses',
            'c.', 'id', 'key', 'summary', 'age', 'height', 'weight', 'cap',
            'goals', 'assists', 'apps', 'appearances', 'gp', 'ga', 'gf',
            'population', 'area', 'density', 'elevation', 'founded', 'established',
            'season', 'episode', 'duration', 'runtime', 'budget', 'box office',
            'attended', 'attendance', 'capacity', 'salary', 'earnings', 'prize',
        }

        self.strong_entity_keywords = {
            'name', 'title', 'film', 'movie', 'show', 'series', 'album', 'song', 'track',
            'stadium', 'airport', 'station', 'organization', 'organisation', 'company',
            'institution', 'member', 'ambassador', 'mission', 'building', 'bridge',
            'ship', 'vessel', 'aircraft', 'character', 'book', 'novel', 'article',
            'school', 'university', 'college', 'hospital', 'museum', 'monument',
            'nominee', 'contestant', 'candidate', 'participant', 'entry', 'player',
            'tunnel', 'road', 'highway', 'galaxy', 'planet', 'species', 'breed',
        }

        self.soft_entity_keywords = {
            'person', 'club', 'city', 'country', 'team', 'player', 'award',
            'island', 'village', 'species', 'mayor', 'leader', 'president',
            'minister', 'director', 'actor', 'actress', 'artist', 'athlete',
            'region', 'district', 'province', 'state', 'place', 'location',
            'site', 'landmark', 'park', 'mountain', 'river', 'lake',
        }

        self.award_context_words = {
            'award', 'nomination', 'nominated', 'filmography', 'discography', 'bibliography'
        }

        self.meta_context_words = {
            'key', 'legend', 'abbreviations', 'definitions', 'glossary', 'codes'
        }

        # Column keywords indicating "the work being evaluated" in award tables
        self.award_work_keywords = {
            'nominated work', 'nominated film', 'nominated series', 'nominated album',
            'film', 'work', 'project', 'production', 'title', 'movie', 'show',
            'programme', 'program', 'series', 'episode', 'song', 'album', 'book',
        }

        # Column keywords indicating the primary entity in diplomatic/mission tables
        self.mission_host_keywords = {
            'host country', 'host state', 'country', 'nation', 'organization',
            'organisation', 'territory',
        }

    # ==========================================
    # PUBLIC API
    # ==========================================

    def identify_entity_column(self, table_context, table_data):
        if not table_data:
            return self._reject(None, "empty table")

        header_chain = table_context.get('header_chain', '') or ''
        caption = table_context.get('caption', '') or ''
        preceding_text = table_context.get('preceding_text', '') or ''
        full_context = f"{header_chain} {caption} {preceding_text}".lower()

        # Hard reject: explicit metadata/legend context
        if any(t in full_context for t in self.meta_context_words):
            return self._reject(None, "metadata/legend context")

        headers = list(table_data[0].keys())
        n_rows = len(table_data)
        is_award_table = any(w in full_context for w in self.award_context_words)
        is_mission_table = any(w in full_context for w in {'diplomatic', 'mission', 'embassy', 'ambassador'})

        # Hard reject: structural legend (short codes -> long definitions)
        if self._is_structural_legend(table_data, headers):
            return self._reject(None, "structural legend (key/value expansion table)")

        # Strip malformed headers (headers that are actually data values)
        bad_headers = [h for h in headers if self._is_malformed_header(h)]
        clean_headers = [h for h in headers if h not in bad_headers]
        if bad_headers:
            self._log(f"  MALFORMED HEADERS removed: {bad_headers}")
        if not clean_headers:
            return self._reject(None, "all headers are malformed")

        # Step 1: Hard elimination
        candidates = []
        for col in clean_headers:
            vals = [row.get(col, '') for row in table_data]
            reason = self._hard_eliminate(col, vals, n_rows, is_award_table)
            if reason is None:
                candidates.append(col)
            else:
                self._log(f"  ELIMINATED '{col}': {reason}")

        if not candidates:
            return self._reject(None, "no candidates survived elimination")

        # Step 2: Score
        scored = []
        for col in candidates:
            vals = [row.get(col, '') for row in table_data]
            score, breakdown = self._score_column(
                col, vals, full_context, clean_headers.index(col),
                is_award_table, is_mission_table, candidates
            )
            scored.append((col, score, breakdown))
            self._log(f"  SCORED '{col}': {score} | {breakdown}")

        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored:
            return self._reject(None, "no scored candidates")

        best_col, best_score, _ = scored[0]

        # Step 3: Reject lone survivor that looks like a description/sentence column
        if len(scored) == 1:
            vals = [row.get(best_col, '') for row in table_data]
            non_empty = [str(v) for v in vals if str(v).strip() not in {'', '—', '–', 'n/a', 'none', '-'}]
            # Strip parenthetical annotations before token count — e.g. 'Tunnel X (on motorway near Y)'
            stripped = [re.sub(r'\s*\([^)]*\)', '', str(v)).strip() for v in non_empty]
            avg_tokens = sum(len(v.split()) for v in stripped if v) / max(len([v for v in stripped if v]), 1)
            has_keyword = any(k in best_col.lower() for k in self.strong_entity_keywords | self.soft_entity_keywords)
            if avg_tokens > 7 and not has_keyword:
                return self._reject(
                    None,
                    f"sole survivor '{best_col}' looks like a description column (avg_tokens={avg_tokens:.1f})"
                )

        # Step 4: Minimum absolute score — even if unopposed, a weak winner is rejected
        MIN_ABSOLUTE_SCORE = 100
        if best_score < MIN_ABSOLUTE_SCORE:
            return self._reject(
                None,
                f"winner '{best_col}' score {best_score} below minimum threshold {MIN_ABSOLUTE_SCORE}"
            )

        # Step 5: Confidence gap check
        if len(scored) >= 2:
            runner_col, runner_score, _ = scored[1]
            gap = best_score - runner_score

            best_col_low = best_col.lower()
            best_is_strong = any(k in best_col_low for k in self.strong_entity_keywords)

            if best_is_strong:
                # Strong keyword in header — very reliable signal, small gap is fine
                gap_threshold = 10
            else:
                # No strong keyword — start with conservative threshold
                gap_threshold = 28

                # Context corroboration: if winner's header appears in the page context
                # AND subject_match fired, we have two independent signals pointing the
                # same way — reduce required gap
                best_breakdown = scored[0][2]
                has_context_boost = (
                    'header_in_context' in best_breakdown or
                    'subject_match' in best_breakdown or
                    'mission_host_boost' in best_breakdown
                )
                if has_context_boost:
                    gap_threshold = 20

            if gap < gap_threshold:
                return self._reject(
                    None,
                    f"ambiguous: '{best_col}' ({best_score}) vs '{runner_col}' ({runner_score}), gap={gap} < threshold={gap_threshold}"
                )

        self._log(f"  WINNER: '{best_col}' with score {best_score}")
        return best_col

    # ==========================================
    # TABLE-LEVEL CHECKS
    # ==========================================

    def _is_structural_legend(self, table_data, headers):
        """
        Detects key/value expansion tables even without 'key'/'legend' in context.

        Two patterns we look for (either is sufficient):

        Pattern A — abbreviation expander:
          Col[0] has very short values (codes/abbreviations, avg < 5 chars),
          Col[1] values are much longer (definitions, avg > 15 chars AND 3x col0).
          E.g.:  GK | Goalkeeper,  W | Win,  * | Tie between players

        Pattern B — paired-column expander (multiple code->definition pairs side by side):
          The table has an even number of columns where odd cols are all short
          and even cols are all long — i.e. the table is a multi-column legend.
          E.g.: GK | Goalkeeper | CM | Central midfielder | RB | Right back

        We do NOT apply this check when ANY column has a strong entity keyword
        in its header (e.g. "Name", "Title") — that means it's a real data table
        that just happens to be small.
        """
        if len(headers) < 2:
            return False

        # Don't fire if any header is a clear entity column — real data table
        all_headers_low = [h.lower() for h in headers]
        has_entity_header = any(
            any(k in h for k in self.strong_entity_keywords | self.soft_entity_keywords)
            for h in all_headers_low
        )
        if has_entity_header:
            return False

        def col_avg_len(col_name):
            vals = [str(row.get(col_name, '')).strip() for row in table_data if str(row.get(col_name, '')).strip()]
            if not vals:
                return 0
            return sum(len(v) for v in vals) / len(vals)

        def col_max_len(col_name):
            vals = [str(row.get(col_name, '')).strip() for row in table_data if str(row.get(col_name, '')).strip()]
            return max((len(v) for v in vals), default=0)

        # Pattern A: first two columns are code -> definition
        avg0 = col_avg_len(headers[0])
        avg1 = col_avg_len(headers[1])
        if avg0 > 0 and avg0 < 6 and avg1 > 12 and avg1 > avg0 * 2.5:
            # Guard: if col[0] values are mostly years (4-digit numbers), this is NOT a legend.
            # Year + Category is the most common false-positive pattern.
            col0_vals = [str(row.get(headers[0], '')).strip() for row in table_data if str(row.get(headers[0], '')).strip()]
            year_count = sum(1 for v in col0_vals if re.fullmatch(r'\d{4}', v))
            if col0_vals and year_count / len(col0_vals) > 0.5:
                self._log(f"  STRUCTURAL LEGEND Pattern A skipped: col0 is a year column")
            else:
                self._log(f"  STRUCTURAL LEGEND (Pattern A): col0 avg={avg0:.1f}, col1 avg={avg1:.1f}")
                return True

        # Pattern B: even number of cols, alternating short/long pairs
        if len(headers) >= 4 and len(headers) % 2 == 0:
            odd_avgs  = [col_avg_len(headers[i]) for i in range(0, len(headers), 2)]
            even_avgs = [col_avg_len(headers[i]) for i in range(1, len(headers), 2)]
            all_odd_short  = all(a < 6  for a in odd_avgs  if a > 0)
            all_even_long  = all(a > 10 for a in even_avgs if a > 0)
            if all_odd_short and all_even_long:
                self._log(f"  STRUCTURAL LEGEND (Pattern B): alternating short/long cols")
                return True

        return False

    def _is_malformed_header(self, header):
        """
        Detects column headers that are actually data values, not column names.
        Two patterns:
          1. Broadcast station codes: "CKSA-DT 2 Citytv CITL-DT 4 CTV"
          2. Award/event names with embedded dates: "Filmfare Awards South August 1999"
             These appear in transposed award tables where rows are categories
             and columns are award ceremonies — the header is data, not a schema label.
        """
        h = str(header).strip()

        # Pattern 1: broadcast station codes with channel numbers
        if re.search(r"[A-Z]{2,5}-[A-Z]{2,3}", h) and re.search(r"\s\d\s", h):
            return True

        # Pattern 2: contains a 4-digit year — it's a date-stamped event name, not a column label
        # Guard: short headers like "1999" alone are handled elsewhere; we require length > 10
        # to avoid catching legitimate short column names that happen to have years
        if len(h) > 10 and re.search(r"\b(19|20)\d{2}\b", h):
            return True

        return False

    # ==========================================
    # ELIMINATION
    # ==========================================

    def _hard_eliminate(self, col, vals, n_rows, is_award_table):
        col_low = col.lower().strip()

        # 1. Exact kill list match
        if col_low in self.kill_list:
            return "kill_list"

        # 2. Partial kill list — but don't kill if column also has an entity signal
        kill_partials = ['date', 'year', 'rank', 'score', 'point', 'goal',
                         'ratio', 'percent', 'pct', '%']
        if any(k in col_low for k in kill_partials):
            has_entity_signal = any(
                k in col_low for k in self.strong_entity_keywords | self.soft_entity_keywords
            )
            if not has_entity_signal:
                return "kill_list_partial"

        # 3. Empty column
        non_empty = [v for v in vals if str(v).strip() not in {'', '—', '–', 'n/a', 'none', '-', '?', 'tbd', 'tba'}]
        if not non_empty:
            return "empty_column"

        # 4. Too many nulls
        null_ratio = 1 - (len(non_empty) / n_rows)
        if null_ratio > 0.3:
            return f"too_many_nulls ({null_ratio:.0%})"

        # 5. Numeric column
        numeric_count = sum(1 for v in non_empty if re.fullmatch(r'[\d,.\-\(\)%\$£€\+\s]+', str(v).strip()))
        if numeric_count / len(non_empty) > 0.7:
            return f"numeric ({numeric_count}/{len(non_empty)})"

        # 6. Date column — strip parenthetical year suffixes before checking.
        # "Battle of Khatoli (1517)" should NOT fire as a date column.
        def _has_standalone_date(v):
            cleaned = re.sub(r'\s*\([^)]*\)\s*$', '', str(v)).strip()
            return bool(re.search(
                r'\b\d{4}\b|\b\d{1,2}[\/\-]\d{1,2}|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
                cleaned.lower()
            ))

        date_count = sum(1 for v in non_empty if _has_standalone_date(v))
        if date_count / len(non_empty) > 0.75:
            return f"date_column ({date_count}/{len(non_empty)})"

        # 7. List-like cells
        list_count = sum(1 for v in non_empty if len(re.split(r'(?<!\w)[,;/](?!\w)', str(v))) > 2)
        if list_count / len(non_empty) > 0.35:
            return f"list_cells ({list_count}/{len(non_empty)})"

        # 8. Low cardinality
        unique_ratio = len(set(str(v).lower().strip() for v in non_empty)) / len(non_empty)
        # In award tables, the 'Award' column (e.g. "Saturn Awards" repeated) is often low-cardinality
        # Use a tighter threshold for award context since the work/nominee col is what matters
        col_low_inner = col.lower().strip()
        if is_award_table and col_low_inner in {'award', 'awards', 'award name'}:
            cardinality_threshold = 0.45  # treat like normal table — repeated award names are not entities
        else:
            cardinality_threshold = 0.15 if is_award_table else 0.45
        if unique_ratio < cardinality_threshold:
            return f"low_cardinality (unique_ratio={unique_ratio:.2f}, threshold={cardinality_threshold})"

        # 9. Must have real alphabetic text
        alpha_count = sum(1 for v in non_empty if re.search(r'[a-zA-Z]{2,}', str(v)))
        if alpha_count / len(non_empty) < 0.5:
            return f"non_text ({alpha_count}/{len(non_empty)} have alpha)"

        # 10. Very short values (abbreviation/code columns)
        avg_len = sum(len(str(v).strip()) for v in non_empty) / len(non_empty)
        if avg_len < 2.5:
            return f"too_short (avg_len={avg_len:.1f})"

        return None

    # ==========================================
    # SCORING
    # ==========================================

    def _score_column(self, col, vals, full_context, col_index, is_award_table, is_mission_table, candidates=None):
        col_low = col.lower().strip()
        breakdown = {}
        score = 0

        # --- Header keyword signals ---
        if any(k in col_low for k in self.strong_entity_keywords):
            breakdown['strong_keyword'] = 60
            score += 60
        elif any(k in col_low for k in self.soft_entity_keywords):
            breakdown['soft_keyword'] = 30
            score += 30

        # --- Context-specific boosts and penalties ---
        # Award tables: boost work columns; penalize category only if a work col exists
        if is_award_table:
            if any(k in col_low for k in self.award_work_keywords):
                breakdown['award_work_boost'] = 50
                score += 50
            if col_low in {'category', 'categories'}:
                other_candidates = [c for c in (candidates or []) if c != col]
                work_col_exists = any(
                    any(k in c.lower() for k in self.award_work_keywords)
                    for c in other_candidates
                )
                if work_col_exists:
                    breakdown['award_category_penalty'] = -40
                    score -= 40

        # Mission/diplomatic tables: boost host country / organization column
        if is_mission_table and any(k in col_low for k in self.mission_host_keywords):
            breakdown['mission_host_boost'] = 45
            score += 45

        # --- Header appears verbatim in page context ---
        if col_low in full_context:
            breakdown['header_in_context'] = 35
            score += 35

        # --- Page subject matches column header ---
        subject = self._get_singular_subject(full_context)
        if subject != "___NONE___" and (subject in col_low or col_low.startswith(subject)):
            breakdown['subject_match'] = 50
            score += 50

        # --- Position bonus: entity col is almost always in first 4 columns ---
        position_bonus = max(0, (4 - col_index)) * 8
        breakdown['position'] = position_bonus
        score += position_bonus

        # --- NER signal ---
        non_empty_vals = [str(v) for v in vals if str(v).strip() not in {'', '—', '–', 'n/a', 'none', '-'}]
        ner_ratio = self._compute_ner_ratio(non_empty_vals)
        ner_score = int(ner_ratio * 55)
        breakdown['ner'] = ner_score
        score += ner_score

        # --- Average token length: entity names are 1-6 words ---
        # Strip parenthetical annotations before token count
        stripped_vals = [re.sub(r'\s*\([^)]*\)', '', v).strip() for v in non_empty_vals]
        avg_tokens = sum(len(v.split()) for v in stripped_vals if v) / max(len([v for v in stripped_vals if v]), 1)
        if 1 <= avg_tokens <= 6:
            breakdown['token_length'] = 15
            score += 15
        elif avg_tokens > 10:
            breakdown['token_length'] = -20
            score -= 20

        # --- Cardinality bonus ---
        unique_ratio = len(set(str(v).lower().strip() for v in non_empty_vals)) / max(len(non_empty_vals), 1)
        if unique_ratio > 0.9:
            breakdown['high_cardinality'] = 15
            score += 15
        elif unique_ratio > 0.7:
            breakdown['med_cardinality'] = 8
            score += 8

        return score, breakdown

    # ==========================================
    # HELPERS
    # ==========================================

    def _get_singular_subject(self, text):
        match = re.search(r'list of (\w+)', text.lower())
        if not match:
            return "___NONE___"
        subject = match.group(1).rstrip('s')
        return subject if len(subject) > 2 else "___NONE___"

    def _compute_ner_ratio(self, values, sample_size=20):
        if not values:
            return 0.0
        sample = random.sample(values, min(len(values), sample_size))
        target_labels = {"PERSON", "ORG", "LOC", "GPE", "WORK_OF_ART", "PRODUCT", "FAC", "EVENT", "NORP"}
        hits = sum(
            1 for val in sample
            if any(ent.label_ in target_labels for ent in self.nlp(str(val)).ents)
        )
        return hits / len(sample)

    def _reject(self, value, reason):
        self._log(f"  REJECT: {reason}")
        return value

    def _log(self, msg):
        if self.debug:
            print(msg)


# ==========================================
# DEBUG RUNNER
# ==========================================

def run_debug_examples():
    gm = GroupMaker(debug=True)

    examples = [
        {
            "label": "[FIX-1] Award table: 'Nominated work' should beat 'Category'",
            "context": {"header_chain": "List of awards and nominations received by Ryan Gosling > Golden Globe Awards", "caption": "", "preceding_text": ""},
            "data": [
                {"Year": 2008, "Category": "Best Actor – Musical or Comedy", "Nominated work": "Lars and the Real Girl", "Result": "Nominated"},
                {"Year": 2011, "Category": "Best Actor – Drama", "Nominated work": "Blue Valentine", "Result": "Nominated"},
                {"Year": 2012, "Category": "Best Actor – Drama", "Nominated work": "The Ides of March", "Result": "Nominated"},
                {"Year": 2017, "Category": "Best Actor – Musical or Comedy", "Nominated work": "La La Land", "Result": "Won"},
            ]
        },
        {
            "label": "[FIX-1b] Award table with both 'Award' and 'Nominated work' — 'Nominated work' should win",
            "context": {"header_chain": "List of awards and nominations received by Charlie Kaufman > Miscellaneous", "caption": "", "preceding_text": ""},
            "data": [
                {"Year": 1999, "Award": "Saturn Awards", "Category": "Best Writing", "Nominated work": "Being John Malkovich", "Result": "Won"},
                {"Year": 2004, "Award": "Saturn Awards", "Category": "Best Writing", "Nominated work": "Eternal Sunshine", "Result": "Nominated"},
                {"Year": 2015, "Award": "Saturn Awards", "Category": "Best Animated Film", "Nominated work": "Anomalisa", "Result": "Nominated"},
            ]
        },
        {
            "label": "[FIX-2] Mission table: 'Host country' should beat 'Host city'",
            "context": {"header_chain": "List of diplomatic missions of Croatia > Americas", "caption": "", "preceding_text": ""},
            "data": [
                {"Host country": "Argentina", "Host city": "Buenos Aires", "Mission": "Embassy", "Concurrent accreditation": "Paraguay Uruguay"},
                {"Host country": "Brazil", "Host city": "Brasília", "Mission": "Embassy", "Concurrent accreditation": "Colombia Venezuela"},
                {"Host country": "Canada", "Host city": "Ottawa", "Mission": "Embassy", "Concurrent accreditation": ""},
                {"Host country": "Chile", "Host city": "Santiago", "Mission": "Embassy", "Concurrent accreditation": ""},
            ]
        },
        {
            "label": "[FIX-3] Structural legend — no 'key' in context, detected by shape",
            "context": {"header_chain": "List of R.S.C. Anderlecht players > Playing positions", "caption": "", "preceding_text": ""},
            "data": [
                {"GK": "RB", "Goalkeeper": "Right back", "CM": "LW", "Central midfielder": "Left winger"},
                {"GK": "CB", "Goalkeeper": "Centre back", "CM": "AM", "Central midfielder": "Attacking midfielder"},
                {"GK": "LB", "Goalkeeper": "Left back", "CM": "FW", "Central midfielder": "Forward"},
            ]
        },
        {
            "label": "[FIX-4] Malformed header (TV station codes) stripped, Lloydminster should win or reject",
            "context": {"header_chain": "List of United States television stations available in Canada", "caption": "", "preceding_text": ""},
            "data": [
                {"Lloydminster": "Moose Jaw", "CKSA-DT 2 Citytv CITL-DT 4 CTV": "CHAB-TV 4 CBC"},
                {"Lloydminster": "Prince Albert", "CKSA-DT 2 Citytv CITL-DT 4 CTV": "CKBI-TV 5 CBC"},
                {"Lloydminster": "Saskatoon", "CKSA-DT 2 Citytv CITL-DT 4 CTV": "CBKST 11 CBC"},
            ]
        },
        {
            "label": "[FIX-5] 'Battle of X (year)' — year in parens should not kill entity col",
            "context": {"header_chain": "List of wars involving the Delhi Sultanate > Lodi Dynasty", "caption": "", "preceding_text": ""},
            "data": [
                {"Name of conflict": "Battle of Khatoli (1517)", "Delhi Sultanate": "Lodi dynasty", "Opposition": "Sisodia dynasty", "Outcome": "Mewar Victory"},
                {"Name of conflict": "Siege of Gwalior (1518)", "Delhi Sultanate": "Lodi dynasty", "Opposition": "Tomaras of Gwalior", "Outcome": "Lodi Victory"},
                {"Name of conflict": "Battle of Dholpur (1519)", "Delhi Sultanate": "Lodi dynasty", "Opposition": "Sisodia dynasty", "Outcome": "Mewar Victory"},
                {"Name of conflict": "Battle of Panipat (1526)", "Delhi Sultanate": "Lodi dynasty", "Opposition": "Timurid dynasty", "Outcome": "Timurid Victory"},
            ]
        },
        {
            "label": "[FIX-6] Bundestag changelog — lone survivor is a sentence, should reject",
            "context": {"header_chain": "List of members of the 14th Bundestag > Summary", "caption": "", "preceding_text": ""},
            "data": [
                {"Time": 1998, "Reason of change": "First meeting", "SPD": 298, "CDU/CSU": 245, "Total": 669},
                {"Time": "5 Oct 1999", "Reason of change": "Uwe Hiksch leaves the SPD caucus and becomes member of PDS", "SPD": 297, "CDU/CSU": 245, "Total": 669},
                {"Time": "9 Aug 2000", "Reason of change": "Ilse Schumann dies; her seat is vacated permanently", "SPD": 296, "CDU/CSU": 245, "Total": 668},
            ]
        },
        {
            "label": "[REGRESSION] Player stats — Name should win despite all-numeric other cols",
            "context": {"header_chain": "List of top scorers in the Premier League", "caption": "", "preceding_text": ""},
            "data": [
                {"Name": "Harry Kane", "Goals": 23, "Assists": 5, "Apps": 30},
                {"Name": "Erling Haaland", "Goals": 27, "Assists": 3, "Apps": 28},
                {"Name": "Mohamed Salah", "Goals": 19, "Assists": 11, "Apps": 32},
            ]
        },
        {
            "label": "[REGRESSION] Box office films — Title should win",
            "context": {"header_chain": "List of American films of 2019 > Box office", "caption": "", "preceding_text": ""},
            "data": [
                {"Rank": 1, "Title": "Avengers: Endgame", "Distributor": "Disney", "Domestic gross": "$858,373,000"},
                {"Rank": 2, "Title": "The Lion King", "Distributor": "Disney", "Domestic gross": "$543,638,043"},
                {"Rank": 3, "Title": "Star Wars: The Rise of Skywalker", "Distributor": "Disney", "Domestic gross": "$515,202,542"},
            ]
        },
        {
            "label": "[REGRESSION] Ambassadors list — Ambassadors col should win",
            "context": {"header_chain": "List of ambassadors of France to the United States", "caption": "", "preceding_text": ""},
            "data": [
                {"From": 1815, "Until": 1821, "Ambassadors": "Jean-Guillaume, baron Hyde de Neuville"},
                {"From": 1821, "Until": 1824, "Ambassadors": "Jacques de Menou"},
                {"From": 1824, "Until": 1830, "Ambassadors": "Joseph Alexandre Jacques Durant de Mareuil"},
            ]
        },
        {
            "label": "[REGRESSION] Explicit key table should reject",
            "context": {"header_chain": "List of Atlanta Braves team records > Table key", "caption": "", "preceding_text": ""},
            "data": [
                {"RBI": "ERA", "Run(s) batted in": "Earned run average"},
                {"RBI": "OPS", "Run(s) batted in": "On-base percentage plus slugging"},
                {"RBI": "*", "Run(s) batted in": "Tie between two or more players"},
            ]
        },
        {
            "label": "[REGRESSION] Newspaper list — Newspaper col should win",
            "context": {"header_chain": "List of newspapers in Chile > Daily circulation", "caption": "", "preceding_text": ""},
            "data": [
                {"Newspaper": "El Mercurio", "City": "Santiago", "Founded": "1 June 1900", "Publisher": "El Mercurio S.A.P."},
                {"Newspaper": "La Tercera", "City": "Santiago", "Founded": "7 July 1950", "Publisher": "Copesa"},
                {"Newspaper": "Diario Financiero", "City": "Santiago", "Founded": "25 October 1988", "Publisher": "Grupo Claro"},
            ]
        },
    ]

    print("=" * 70)
    results = []
    for ex in examples:
        print(f"\nEXAMPLE: {ex['label']}")
        print(f"Context: {ex['context']['header_chain']}")
        result = gm.identify_entity_column(ex['context'], ex['data'])
        print(f">>> RESULT: {result}")
        results.append((ex['label'], result))
        print("-" * 70)

    print("\n\nSUMMARY")
    print("=" * 70)
    for label, result in results:
        status = "✓" if result is not None else "✗ (rejected)"
        print(f"  {status}  {result or 'None':45s}  {label}")


if __name__ == "__main__":
    run_debug_examples()

# import re
# import random
# import spacy
#
#
# class GroupMaker:
#     def __init__(self):
#         # Load NLP model
#         try:
#             self.nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "textcat"])
#         except OSError:
#             raise ImportError("Please run: python -m spacy download en_core_web_sm")
#
#         # Expanded kill list
#         self.kill_list = {
#             'rank', 'gross', 'collection', 'revenue', 'year', 'date',
#             'pos', 'notes', 'ref', 'no.', 'total', 'index', 'score',
#             'points', 'won', 'lost', 'drawn', 'c.', 'id', 'key', 'summary'
#         }
#
#         self.target_patterns = {
#             'name', 'title', 'person', 'club', 'city', 'country', 'team',
#             'company', 'airport', 'island', 'village', 'monument', 'species',
#             'player', 'film', 'station', 'award', 'organization', 'mayor'
#         }
#         self.strong_entities = {'title', 'name', 'film', 'stadium', 'airport', 'station', 'organization', 'company',
#                                 'institution', 'member', 'ambassador', 'mission'}
#
#     # ==========================================
#     # HELPERS
#     # ==========================================
#     def _get_singular_subject(self, header_chain):
#         """Extracts 'mayor' from 'List of mayors of Naples'."""
#         clean = header_chain.lower().replace("list of ", "").strip()
#         # Take the first word and strip common pluralization
#         subject = clean.split(' ')[0].rstrip('s')
#         return subject if len(subject) > 2 else "___NONE___"
#
#     def _is_categorical(self, values, is_award_table=False):
#         """Cardinality check: Relaxed for awards/filmographies."""
#         if not values: return True
#         unique_vals = set(str(v).strip().lower() for v in values)
#         ratio = len(unique_vals) / len(values)
#
#         threshold = 0.10 if is_award_table else 0.40
#         return ratio < threshold
#
#     def _compute_ner_ratio(self, values, sample_size=15):
#         if not values: return 0.0
#         sample = random.sample(values, min(len(values), sample_size))
#         target_labels = {"PERSON", "ORG", "LOC", "GPE", "WORK_OF_ART", "PRODUCT", "FAC"}
#         valid_count = sum(1 for val in sample if any(ent.label_ in target_labels for ent in self.nlp(str(val)).ents))
#         return valid_count / len(sample)
#
#     # ==========================================
#     # MAIN COMPONENT 1
#     # ==========================================
#     def identify_entity_column(self, table_context, table_data):
#         if not table_data: return None
#
#         header_chain = table_context.get('header_chain', '') or ''
#         full_context = f"{header_chain} {table_context.get('caption', '')}".lower()
#
#         # Fix 1: Metadata Nuke
#         if any(trigger in full_context for trigger in {'key', 'legend', 'abbreviations', 'definitions'}):
#             return None
#
#         headers = list(table_data[0].keys())
#         is_award_table = any(word in full_context for word in ['award', 'nomination', 'filmography'])
#
#         surviving_cols = []
#         for col in headers:
#             vals = [row.get(col, '') for row in table_data]
#
#             # Fix 2: Better Text Validation (Ignore placeholders like N/A)
#             clean_vals = [str(v).strip().lower() for v in vals if v]
#             real_text = [v for v in clean_vals if v not in {'—', 'n/a', 'none', '-'} and re.search(r'[a-zA-Z]', v)]
#
#             if len(real_text) < (len(vals) * 0.3): continue  # Relaxed to 30% for messy tables
#             if self._is_categorical(vals, is_award_table): continue
#
#             surviving_cols.append(col)
#
#         # Scoring Phase
#         scored_cols = []
#         for index, col in enumerate(surviving_cols):
#             col_low = col.lower()
#             score = 0
#
#             # Fix 3: Strong Keyword Boost
#             if any(strong in col_low for strong in self.strong_entities): score += 65
#             if col_low in full_context: score += 40
#             score += (5 - index) * 5  # Position bias
#
#             scored_cols.append({"name": col, "score": score})
#
#         scored_cols.sort(key=lambda x: x['score'], reverse=True)
#         if not scored_cols: return None
#
#         # Tie-Breaker Logic
#         top = scored_cols[0]
#         if len(scored_cols) > 1:
#             runner = scored_cols[1]
#             if (top['score'] - runner['score']) < 20:
#                 # If tied, check if one matches the 'Spirit' of the page title
#                 if "country" in full_context and "country" in top['name'].lower(): return top['name']
#                 if "country" in full_context and "country" in runner['name'].lower(): return runner['name']
#                 # Default to None if we can't be sure
#                 return None
#
#         return top['name']
#
#     def _is_categorical(self, values, is_award):
#         if not values: return True
#         unique = len(set(str(v).lower().strip() for v in values))
#         ratio = unique / len(values)
#         return ratio < (0.10 if is_award else 0.40)