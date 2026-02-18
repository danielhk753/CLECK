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


class WikiTableCleaner:
    def __init__(self):
        # 1. Open/Subjective Set Triggers (Reject these tables)
        self.open_set_triggers = [
            r'\bnotable\b', r'\bfamous\b', r'\bselected\b',
            r'\bmajor\b', r'\bexamples?\b', r'\bpartial\b',
            r'\bsome\b', r'\bvarious\b'
        ]

        # 2. Ranking Keywords (Trigger "Top X" naming)
        self.ranking_triggers = [
            'highest', 'lowest', 'most', 'least', 'best', 'worst', 'highest-grossing',
            'top', 'largest', 'smallest', 'earliest', 'latest', 'biggest'
        ]

        # 3. High-Confidence Entity Headers
        self.universal_priority = [
            'name', 'title', 'airline', 'team', 'character', 'honor',
            'mayor', 'organization', 'mission', 'planet', 'film', 'company',
            'artist', 'album', 'song', 'work', 'recipient', 'location', 'city',
            'species', 'book', 'game', 'series'
        ]

        # 4. Kill List (Statistical/Metadata columns to ignore)
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

        # Reject empty/null
        if not val_str or val_str.lower() in ["none", "null", "n/a", "-", "—", "unknown", "tba", "tbd"]:
            return False

        # Must contain at least one letter (Rejects "1999", "1,000", "$500", "---")
        if not re.search(r'[a-zA-Z]', val_str):
            return False

        # Reject Date Formats (YYYY-MM-DD, DD/MM/YYYY, etc.)
        if re.match(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$', val_str):
            return False

        return True

    def _get_entity_column(self, rows, context_str):
        """Scans headers to find the best candidate for the Entity Name."""
        if not rows: return None
        headers = list(rows[0].keys())
        context = context_str.lower()

        scores = {}
        for h in headers:
            h_low = h.lower().strip()

            # Immediate rejection
            if any(k == h_low or f" {k}" in h_low or f"{k} " in h_low for k in self.kill_list):
                continue

            score = 0
            # 1. Exact Match in Priority List
            if h_low in self.universal_priority:
                score += 50

            # 2. Singular Match in Page Title (e.g., Title: "Mayors", Header: "Mayor")
            singular = h_low.rstrip('s')
            if len(singular) > 2 and singular in context:
                score += 40

            # 3. Position Bias (Leftmost columns are usually entities)
            score += (len(headers) - headers.index(h)) * 2

            scores[h] = score

        if not scores: return None
        return max(scores, key=scores.get)

    def _generate_clean_name(self, page_title, hierarchy, row_count):
        """Generates a natural language group name with Top-X logic."""
        # Clean the title
        clean_title = re.sub(r'^(List of (the )?|Awards and nominations received by )', '', page_title,
                             flags=re.IGNORECASE).strip()

        # Clean the hierarchy (remove redundant words found in title)
        title_words = set(clean_title.lower().split())
        clean_hierarchy = []
        for section in hierarchy:
            # Remove parentheses like (1990 film)
            s_clean = re.sub(r'\(.*?\)', '', section).strip()
            # Remove "List of" from section
            s_clean = re.sub(r'^(List of|List)\s+', '', s_clean, flags=re.IGNORECASE).strip()

            # Check for redundancy
            s_words = s_clean.split()
            unique = [w for w in s_words if w.lower() not in title_words]

            if unique and s_clean.lower() not in ["members", "current", "former", "list"]:
                clean_hierarchy.append(" ".join(unique))
            elif s_clean.lower() in ["current", "former", "defunct"]:
                clean_hierarchy.append(s_clean)

        # Assemble Name
        base = f"{' '.join(clean_hierarchy)} {clean_title}".strip()

        # "Top X" Logic: If name implies ranking, add the count
        if any(t in base.lower() for t in self.ranking_triggers):
            # Only add if a number isn't already there (avoid "Top 10 Top 10")
            if not re.search(r'\b\d+\b', base):
                base = f"Top {row_count} {base}"

        return base[0].upper() + base[1:]

    def process_batch(self, raw_samples):
        """
        Input: List of raw dicts (from your scraper).
        Output: List of cleaned, verified dataset dicts.
        """
        cleaned_dataset = []

        for sample in raw_samples:
            # 1. Parse Metadata
            full_path = sample.get('table_name', '')
            parts = full_path.split(' > ')
            page_title = parts[0]
            hierarchy = parts[1:] if len(parts) > 1 else []
            rows = sample.get('rows', [])

            # A. Closed Set Check (Reject subjective lists)
            context = (page_title + " " + " ".join(hierarchy)).lower()
            if any(re.search(p, context) for p in self.open_set_triggers):
                continue  # Skip this sample

            # B. Identify Entity Column
            entity_col = self._get_entity_column(rows, context)
            if not entity_col:
                continue  # Skip (No clear entity)

            # C. STRICT Textual Validation (The 100% Filter)
            # If ANY row in the entity column is non-textual, discard the whole table.
            is_valid_table = True
            for row in rows:
                val = row.get(entity_col)
                if not self._is_textual(val):
                    is_valid_table = False
                    break

            if not is_valid_table:
                continue  # Skip (Dirty data found)

            # D. Generate Final Name
            group_name = self._generate_clean_name(page_title, hierarchy, len(rows))

            # E. Build Output Object
            cleaned_dataset.append({
                "Group name": group_name,
                # Construct a dummy URL or use real one if available in input
                "Wikipedia link": f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
                "Entity column": entity_col,
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

        print("\n" + "=" * 80)
        print(f"{'WIKI TABLE PROCESSING SUMMARY':^80}")
        print("=" * 80)
        print(f"Total Tables Processed:   {total_raw_count}")
        print(f"Passed (Closed Sets):     {passed}")
        print(f"Rejected:                 {rejected}")

        if total_raw_count > 0:
            success_rate = (passed / total_raw_count) * 100
            print(f"Success Rate:             {success_rate:.1f}%")

        print("-" * 80)
        print(f"{'EXTRACTED GROUPS':^80}")
        print("-" * 80)

        for i, item in enumerate(cleaned_dataset[:20], 1):
            name = item['Group name']
            col = item['Entity column']
            rows = len(item['Table data'])
            print(f"{i:2}. {name:<80} | Column: {col:<20} | Rows: {rows}")

        print("=" * 80 + "\n")

# ------------------------------------------------------------------------------------------------------------------
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