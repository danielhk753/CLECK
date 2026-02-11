# TODO: Tell it in the prompt to not mention specific known facts so the question remains natural.
# TODO: Maybe use a different model more suited for this tasks???

import re
import json
import random
import tqdm.auto as tqdm_lib
from collections import Counter

from knowledge_maze.tools.api_tools import gptAPI

# QUESTION_PROMPT = """
# ### ROLE
# You are an expert Data Journalist and Logic Puzzle Creator. Your task is to generate a natural language question based
# on a specific "Group" and its "Table Data" using the provided "Strategy."
#
# ### THE FEW-SHOT RULE
# You will be provided with "GOOD EXAMPLES" and "BAD EXAMPLES."
# - Emulate the style, complexity, and clarity of the GOOD EXAMPLES.
# - Avoid the pitfalls described in the BAD EXAMPLES.
# - If the current data violates any "UNACCEPTABLE IF" conditions or "REQUIREMENTS," you MUST return the NULL OUTPUT.
#
# ### TASK GUIDELINES
# 1. **FOLLOW THE STRATEGY**: Use the provided Definition and Requirements to derive the question.
# 2. **CONTEXTUAL INDEPENDENCE**: Use the "Group Name" to establish the setting. The question must be self-contained and sound professional.
# 3. **DATA INTEGRITY**: Ensure the "data_column" exists exactly as written in the table.
# 4. **NO META-TALK**: Do not mention "the table" or "the data."
#
# ### OUTPUT FORMAT (Strict JSON)
# {
#   "thought_process": "Explain how you applied the logic and why the data meets the requirements.",
#   "selected_strategy": "The name of the strategy used",
#   "question": "The natural language question",
#   "data_column": "The exact header name used for the reasoning"
# }
#
# ### NULL OUTPUT
# If the data does not support the strategy or fails requirements:
# {"thought_process": "REASON WHY DATA FAILED", "selected_strategy": null, "question": null, "data_column": null}
# """
QUESTION_PROMPT = """
### ROLE
You are an expert Data Journalist. Your task is to generate a natural, professional question based on a dataset.

### NATURAL LANGUAGE CONSTRAINTS
- **NO REDUNDANCY**: If the Group Name contains a date range (e.g., "1997–2001") or a fixed context that applies to ALL rows, do NOT mention it. It is implied context.
- **SUBJECT FOCUS**: Focus on the entities. Instead of "In the table of coaches...", ask "Which coach...".
- **CONCISE & FLUID**: Avoid "According to the data" or "Based on the provided list." 
- **NO SPOILERS**: Do not include the answer or the specific value you are looking for in the question premise.

### TASK GUIDELINES
1. **STRATEGY**: Apply the {strategy_name} logic.
2. **COLUMN FOCUS**: You are encouraged to use the '{focus_column}' column, but you may use any columns necessary to satisfy the strategy.
3. **SELF-CONTAINED**: The question must make sense on its own without needing to see the table first, but it must not be wordy.
4. **DATA INTEGRITY**: The 'data_column' field in your output must contain the exact header name(s) used to answer the question.

### OUTPUT FORMAT (Strict JSON)
{
  "thought_process": "Explain why this question is natural and how it uses the relevant columns.",
  "question": "The natural language question",
  "data_column": "The exact header name used (or a list of strings if multiple columns are required)"
}
"""

class TableSanitizer:
    def __init__(self, tables):
        self.CURRENCY_SYMBOLS = {'$', '€', '£', '¥', '₹', '₽', '₩', '฿', '₫', '₪', 'US$', 'A$', 'C$'}
        # Metrics that are allowed to follow a pure number
        self.VALID_METRICS = {
            'm', 'km', 'mi', 'kg', 'lb', 'ft', 'in', 'oz', 'g', 'cm', 'mm',
            '%', 'mph', 'kph', 'sec', 'min', 'hr', 'days', 'years', 'points', 'pts'
        }
        self.keywords_blacklist = {
            'rank', 'no.', 'no', '#', 'index', 'sl no', 'notes',
            'note', 'comments', 'comment', 'remarks', 'remark', 'description'
        }
        self.tables = tables
        self.sanitized_tables = []

    @staticmethod
    def _classify_value(val, col_header=""):
        val = str(val).strip()
        if not val or val in ["—", "-", "N/A", "None"]:
            return "empty", None

        # 1. NEW: STRIP WIKIPEDIA REFERENCES (e.g., [a], [12])
        # This prevents "[a]" from breaking your numerical check
        val_clean = re.sub(r'\[.*?\]', '', val).strip()

        # Keywords that suggest this column MUST be numerical
        num_keywords = ["gross", "income", "revenue", "gva", "gdp", "profit", "capacity",
                        "attendance", "population", "area", "total", "budget", "salary"]
        header_is_num = any(k in col_header for k in num_keywords)

        # 1. TEMPORAL CHECK
        is_date_header = "date" in col_header.lower() or "year" in col_header.lower()

        # Updated to catch:
        # - 1983-1984 (Full years)
        # - 1983–84   (Shortened second year)
        # - 1998–present
        period_pattern = r'\b\d{4}\s*[–-]\s*(\d{2,4}|present|current|active)\b'

        date_pattern = r'(\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b)|(\b\d{4}[./-]\d{2}[./-]\d{2}\b)'
        nat_date_pattern = r'\b(?:\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})\b'
        year_pattern = r'^\b(1\d|20)\d{2}\b$'

        if (is_date_header or
                re.search(period_pattern, val_clean, re.I) or
                re.search(date_pattern, val_clean) or
                re.search(nat_date_pattern, val_clean, re.I) or
                re.match(year_pattern, val_clean)):
            return "temporal", val_clean

        # 2. STRICT NUMERICAL CHECK
        # 1. Matches digits at the start
        # 2. Allows optional space
        # 3. Allows units like MW, kW, km/h, mph
        # 4. Stops before parentheses to avoid conversion confusion
        num_match = re.match(r'^[£\$]?([\d,.]+)\s*([a-zA-Z/%]+)?$', val_clean, re.I)

        if num_match:
            number_part = num_match.group(1).replace(',', '')
            unit_part = num_match.group(2).lower() if num_match.group(2) else ""

            # Guard against "3FB" (as discussed before)
            if unit_part and not val_clean.split()[0].replace('£', '').replace('$', '').replace(',', '').replace('.',
                                                                                                                 '').isdigit():
                return "string", val

            try:
                num_float = float(number_part)

                # Apply scaling
                multiplier = 1
                if 'billion' in val_clean.lower():
                    multiplier = 1e9
                elif 'million' in val_clean.lower():
                    multiplier = 1e6

                return "numerical", num_float * multiplier
            except ValueError:
                pass

        # 3. THE "FORCED" NUMERICAL FALLBACK
        # If the header says "Gross" and the value is "1,200", but regex missed it
        if header_is_num:
            only_digits = re.sub(r'[^\d.]', '', val_clean)
            if only_digits:
                try:
                    return "numerical", float(only_digits)
                except:
                    pass

        return "string", val

    def _process_table(self, sample):
        group_name = sample['Group name']
        table_data = sample['Table data']
        wiki_link = sample['Wikipedia link']
        ent_col = sample['Entity column']

        if not table_data:
            return None

        num_rows = len(table_data)
        col_headers = table_data[0].keys()
        column_class = {}
        keep_as_other = []

        for col in col_headers:
            clean_col = col.lower().strip()

            # Check blacklist
            if clean_col in self.keywords_blacklist:
                column_class[col] = "trash"
                continue

            if col == ent_col:
                column_class[col] = "entity"
                continue

            # Pass column header to classifier to catch "Date" overrides
            classifications = [self._classify_value(row.get(col), col_header=col) for row in table_data]
            valid_entries = [c for c in classifications if c[0] != "empty"]

            # 1. DENSITY CHECK
            if len(valid_entries) < (num_rows * 0.5):
                column_class[col] = "trash"
                continue

            # 2. INVARIANCE CHECK (Delete if all values are identical)
            raw_values = [str(c[1]) for c in valid_entries]
            unique_vals = set(raw_values)
            if len(unique_vals) <= 1:
                column_class[col] = "trash"
                continue

            # 3. ORDINAL/SEQUENCE CHECK
            if all(str(v).isdigit() for v in raw_values):
                numeric_vals = sorted([int(v) for v in raw_values])
                if numeric_vals == list(range(min(numeric_vals), max(numeric_vals) + 1)):
                    column_class[col] = "trash"
                    continue

            types = [c[0] for c in valid_entries]
            dominant_type = Counter(types).most_common(1)[0][0]

            # 4. FINAL CLASSIFICATION LOGIC
            if dominant_type == "string":
                if 1 < len(unique_vals) < len(valid_entries):
                    column_class[col] = "list"
                else:
                    # SHORT STRING CHECK (1-2 words, alpha only)
                    # Updated check to allow hyphens and parentheses
                    is_clean_short = all(
                        len(v.split()) <= 3 and all(c.isalnum() or c in " -()" for c in v)
                        for v in raw_values
                    )
                    if is_clean_short:
                        column_class[col] = "trash"
                        keep_as_other.append(col)
                    else:
                        column_class[col] = "trash"

            elif dominant_type == "numerical":
                vals = [c[1] for c in valid_entries if c[0] == "numerical"]
                if all(1000 <= v <= 2100 for v in vals):
                    column_class[col] = "temporal"
                else:
                    column_class[col] = "numerical"
            else:
                column_class[col] = dominant_type

        # 5. CONSTRUCT CLEANED TABLE
        cleaned_table = []
        for row in table_data:
            new_row = {col: row.get(col) for col, cls in column_class.items() if cls != "trash"}
            cleaned_table.append(new_row)

        final_classes = {k: v for k, v in column_class.items() if v != "trash"}

        return {
            "Group name": group_name,
            "Table data - cleaned": cleaned_table,
            "Table data - original": table_data,
            "Wikipedia link": wiki_link,
            "Column class": final_classes
        }

    def process_tables(self):
        for table in tqdm_lib.tqdm(self.tables, desc="Sanitizing tables"):
            self.sanitized_tables.append(self._process_table(table))


class StrategySelector:
    def __init__(self, sanitized_tables):
        self.data = sanitized_tables
        self.updated_data = []

    @staticmethod
    def get_top_strategies(data_sample):
        data = data_sample['Table data - cleaned']
        classes = data_sample['Column class']
        group_name = data_sample['Group name'].lower()
        row_count = len(data)
        threshold = 0.45

        if row_count < 2: return []

        # 1. COLUMN DENSITY
        total_cols = len(classes)
        num_cols = sum(1 for c in classes.values() if c == "numerical")
        list_cols = sum(1 for c in classes.values() if c == "list")
        time_cols = sum(1 for c in classes.values() if c == "temporal")
        ent_cols = sum(1 for c in classes.values() if c == "entity")

        if total_cols != 0:
            num_density = num_cols / total_cols
            list_density = list_cols / total_cols
            temp_density = time_cols / total_cols
        else:
            num_density, list_density, temp_density = 0, 0, 0

        has_list, has_num, has_time, has_entity = list_cols > 0, num_cols > 0, time_cols > 0, ent_cols > 0
        scores = {}

        # --- 2. ROW COUNT CONSTRAINTS ---
        if row_count == 2:
            if has_num: scores["COMPARATIVE_DIFFERENCE"] = 1.0
            return [k for k, v in scores.items() if v >= threshold]

        if row_count == 3:
            if has_num:
                scores["EXTREME_NUMERICAL"] = 0.9
                scores["COMPARATIVE_DIFFERENCE"] = 0.8
            if has_time: scores["EXTREME_TEMPORAL"] = 0.9
            return sorted([k for k, v in scores.items() if v >= threshold], key=lambda x: scores[x], reverse=True)

        # --- 3. LIST DIVERSITY & CONCENTRATION CHECK ---
        diversity_score = 0
        concentration_score = 0

        if has_list:
            list_col_names = [col for col, cls in classes.items() if cls == "list"]
            for col in list_col_names:
                vals = [str(row.get(col)) for row in data if row.get(col) is not None]
                if not vals: continue

                unique_ratio = len(set(vals)) / len(vals)
                counts = Counter(vals)
                most_common_ratio = counts.most_common(1)[0][1] / len(vals)

                # High unique_ratio -> many different categories (Diversity)
                diversity_score = max(diversity_score, unique_ratio)
                # High most_common_ratio -> one category dominates (Concentration)
                concentration_score = max(concentration_score, most_common_ratio)

        # --- 4. CORE SCORING ---

        # ENTITY LOGIC
        if has_entity:
            ent_col_name = next(col for col, cls in classes.items() if cls == "entity")
            unique_vals = set(str(row.get(ent_col_name)) for row in data)
            if len(unique_vals) == row_count and row_count >= 4:
                scores["COUNT_TOTAL"] = 0.5

        # CATEGORICAL LOGIC
        if has_list:
            # Boost Diversity if there are many unique categories
            scores["UNIQUE_DIVERSITY"] = 0.6 + (0.3 * diversity_score)
            # Boost Frequency/Filter if one category is very dominant
            scores["FREQUENCY_MODE"] = 0.7 + (0.2 * concentration_score)
            scores["FILTER_COUNT_CATEGORICAL"] = 0.75 + (0.15 * concentration_score)

            if has_num or has_entity:
                scores["GROUPED_LOOKUP"] = 0.7 + (0.1 * list_density)

        # NUMERICAL LOGIC
        if has_num:
            agg_keywords = ["total", "gross", "income", "revenue", "average", "sum", "gva", "gdp", "profit", "capacity"]
            percent_keywords = ["percent", "%", "share", "rate", "ratio"]

            is_agg_context = any(k in group_name for k in agg_keywords)
            is_pct_context = any(k in group_name for k in percent_keywords)

            scores["TOP_K_RANKING"] = (0.8 if row_count > 8 else 0.6) + (0.15 * num_density)
            scores["EXTREME_NUMERICAL"] = 0.8 + (0.15 * num_density)
            scores["FILTER_COUNT_RANGE"] = 0.7 + (0.1 * num_density)

            if is_agg_context:
                scores["AGGREGATION"] = 0.95
                scores["EXTREME_NUMERICAL"] += 0.05

            if is_pct_context:
                scores["TOP_K_RANKING"] += 0.1
                scores["EXTREME_NUMERICAL"] += 0.05

            if 4 <= row_count <= 6:
                scores["COMPARATIVE_DIFFERENCE"] = 0.7

        # TEMPORAL LOGIC
        if has_time:
            # Fix 2: Boost based on temp_density
            scores["EXTREME_TEMPORAL"] = 0.8 + (0.15 * temp_density)
            scores["FILTER_COUNT_RANGE"] = max(scores.get("FILTER_COUNT_RANGE", 0), 0.75 + (0.1 * temp_density))

        # 5. FINAL FILTER
        valid_strategies = {k: v for k, v in scores.items() if v >= threshold}
        return sorted(valid_strategies, key=valid_strategies.get, reverse=True)[:3]

    def get_strategies(self):
        for table in tqdm_lib.tqdm(self.data, desc="Detecting top strategies"):
            strategies = self.get_top_strategies(table)

            if len(strategies) > 0:
                self.updated_data.append({
                    "Group name": table["Group name"],
                    "Table data - cleaned": table["Table data - cleaned"],
                    "Table data - original": table["Table data - original"],
                    "Wikipedia link": table["Wikipedia link"],
                    "Top strategies": strategies,
                    "Entity column": next((k for k, v in table['Column class'].items() if v == 'entity'), None)
                })

# class QuestionGenerator:
#     def __init__(self, input_tables, strategy_definitions):
#         self.input_tables = input_tables
#
#         # Strategy definitions contain the prompt logic (definition, examples, etc.)
#         if isinstance(strategy_definitions, dict) and "strategies" in strategy_definitions:
#             self.definitions = strategy_definitions["strategies"]
#         else:
#             self.definitions = strategy_definitions
#
#         self.model = gptAPI(QUESTION_PROMPT,
#                             output_format='string',
#                             temperature=0.2)
#
#         self.valid_results = []
#         self.stats = {
#             "initial_tables": len(input_tables),
#             "final_success": 0,
#             "failed_all_strategies": 0
#         }
#
#     def _prepare_prompt(self, table, strategy_name):
#         # Fetch definition from the global strategy library
#         strat_def = self.definitions.get(strategy_name, {})
#
#         definition = strat_def.get('definition', 'No definition provided.')
#         requirements = strat_def.get('requirements', 'No specific requirements.')
#         unacceptable = strat_def.get('unacceptable_if', 'No specific exclusion criteria.')
#
#         good_shots = ""
#         for i, ex in enumerate(strat_def.get('good_examples', []), 1):
#             good_shots += f"GOOD EXAMPLE {i}:\nInput Group: {ex.get('input_group', 'N/A')}\nOutput: {ex.get('output', 'N/A')}\n\n"
#
#         # Construct task using the CLEANED data
#         user_input = f"""
#         ### STRATEGY TO APPLY: {strategy_name}
#         DEFINITION: {definition}
#         HARD REQUIREMENTS: {requirements}
#         UNACCEPTABLE IF: {unacceptable}
#
#         ### FEW-SHOT DEMONSTRATIONS
#         {good_shots if good_shots else "No examples provided."}
#
#         ### CURRENT TASK
#         DATA TO ANALYZE:
#         Group Name: {table.get('Group name')}
#         Table Data: {table.get('Table data - cleaned')}
#
#         Apply the strategy to the "DATA TO ANALYZE" above.
#         If the data supports a high-quality question for this strategy, generate the JSON.
#         If not, return the NULL OUTPUT.
#         """
#         return user_input
#
#     async def generate_questions(self):
#         # Each table starts with its own list of pre-ranked strategies
#         # We track progress by index: table_idx -> strategy_attempt_index
#         remaining_tables = {i: 0 for i in range(len(self.input_tables))}
#
#         # We will loop until all tables have a question OR exhausted their Top strategies
#         while remaining_tables:
#             current_batch_indices = list(remaining_tables.keys())
#             batch_prompts = []
#             batch_strategies = []
#
#             for idx in current_batch_indices:
#                 table = self.input_tables[idx]
#                 strat_list = table.get('Top strategies', [])
#                 attempt_idx = remaining_tables[idx]
#
#                 # Check if we still have ranked strategies to try
#                 if attempt_idx < len(strat_list):
#                     selected_strat = strat_list[attempt_idx]
#                     batch_prompts.append(self._prepare_prompt(table, selected_strat))
#                     batch_strategies.append(selected_strat)
#                 else:
#                     # Exhausted all top 3 strategies for this table
#                     self.stats["failed_all_strategies"] += 1
#                     del remaining_tables[idx]
#
#             if not batch_prompts:
#                 break
#
#             # Execute batch API call
#             raw_responses = await self.model.async_run(batch_prompts)
#
#             for i, response in enumerate(raw_responses):
#                 table_idx = current_batch_indices[i]
#                 parsed = self._parse_json(response)
#
#                 if parsed and parsed.get('question'):
#                     # SUCCESS: Store and remove from active processing
#                     self._store_valid(self.input_tables[table_idx], parsed, batch_strategies[i])
#                     del remaining_tables[table_idx]
#                 else:
#                     # FAILURE: Increment attempt index to try the NEXT ranked strategy in the next loop
#                     remaining_tables[table_idx] += 1
#
#         self.stats["final_success"] = len(self.valid_results)
#         self._print_summary()
#         return self.valid_results
#
#     @staticmethod
#     def _parse_json(text):
#         try:
#             if "{" in text:
#                 json_str = text[text.find("{"):text.rfind("}") + 1]
#                 return json.loads(json_str)
#             return None
#         except:
#             return None
#
#     def _store_valid(self, table, result, strategy_used):
#         # Extract the specific column the AI chose to answer the question
#         target_col = result.get('data_column')
#
#         # Identify the Entity column from the table metadata
#         # We assume the first column with class 'entity' is the primary key
#         classes = table.get('Column class', {})
#         entity_col = next((col for col, cls in classes.items() if cls == 'entity'), None)
#
#         # Create the mapping for verification
#         answer_map = {}
#         if target_col and entity_col:
#             for row in table.get('Table data - cleaned', []):
#                 ent_val = str(row.get(entity_col, "Unknown"))
#                 answer_map[ent_val] = row.get(target_col)
#
#         self.valid_results.append({
#             "group_name": table['Group name'],
#             "strategy": strategy_used,
#             "question": result['question'],
#             "thought": result.get('thought_process'),
#             "answer_map": answer_map,
#             "metadata": {
#                 "wiki_link": table.get('Wikipedia link'),
#                 "target_column": target_col,
#                 "entity_column": entity_col
#             },
#             "data": table.get('Table data - original')
#         })
#
#     def _print_summary(self):
#         print("\n" + "═" * 50)
#         print("🚀 SEQUENTIAL STRATEGY EXECUTION COMPLETE")
#         print("═" * 50)
#         print(f"Initial Tables:      {self.stats['initial_tables']}")
#         print(f"Successfully Gen:    {self.stats['final_success']}")
#         print(f"Exhausted Strats:    {self.stats['failed_all_strategies']}")
#         print("═" * 50)

class QuestionGenerator:
    def __init__(self, input_tables, strategy_definitions):
        self.input_tables = input_tables
        self.definitions = strategy_definitions.get("strategies", strategy_definitions)
        self.model = gptAPI(QUESTION_PROMPT,
                            output_format='string',
                            temperature=0.2,
                            max_concurrent_requests=100,
                            max_rpm=4900)

        # Track usage to push the model toward under-used strategies
        self.global_strategy_counts = Counter()

        # 1. PRE-CALCULATE INVENTORY SCARCITY
        # We look at how many tables in this specific run actually SUPPORT each strategy
        self.inventory = Counter()
        for table in input_tables:
            # We treat the 'Top strategies' as a pool of possibilities
            for s in table.get('Top strategies', []):
                self.inventory[s] += 1

        self.valid_results = []
        self.stats = {
            "initial_tables": len(input_tables),
            "final_success": 0,
            "failed_all_strategies": 0
        }

    def _prepare_strat_list(self, table):
        strat_list = list(table.get('Top strategies', []))

        # --- THE DIVERSITY ENGINE ---
        # We sort by:
        # 1. Current usage (self.global_strategy_counts)
        # 2. Scarcity (how many tables TOTAL can even do this strategy)
        # 3. Randomness (to break ties in the first batch)

        random.shuffle(strat_list)

        def diversity_score(s):
            # Lower score = higher priority
            # We prioritize strategies that:
            # - Haven't been used much yet
            # - Are rare in the whole dataset
            usage = self.global_strategy_counts[s]
            scarcity = self.inventory.get(s, 1)
            return (usage * 1000) + scarcity

        strat_list.sort(key=diversity_score)
        return strat_list

    def _prepare_prompt(self, table, strategy_name):
        strat_def = self.definitions.get(strategy_name, {})

        # Identify the most logical column for this strategy to reduce LLM "wandering"
        classes = table.get('Column class', {})
        mapping = {
            "numerical": ["AGGREGATION", "EXTREME_NUMERICAL", "TOP_K_RANKING", "COMPARATIVE_DIFFERENCE",
                          "FILTER_COUNT_RANGE"],
            "list": ["UNIQUE_DIVERSITY", "FREQUENCY_MODE", "FILTER_COUNT_CATEGORICAL", "GROUPED_LOOKUP"],
            "temporal": ["EXTREME_TEMPORAL"]
        }
        target_type = next((t for t, strats in mapping.items() if strategy_name in strats), "list")
        focus_col = next((col for col, cls in classes.items() if cls == target_type), "None")

        good_shots = ""
        for i, ex in enumerate(strat_def.get('good_examples', []), 1):
            good_shots += f"GOOD EXAMPLE {i}:\nInput Group: {ex.get('input_group', 'N/A')}\nOutput: {ex.get('output', 'N/A')}\n\n"

        return f"""
        ### STRATEGY TO APPLY: {strategy_name}
        DEFINITION: {strat_def.get('definition', 'N/A')}
        HARD REQUIREMENTS: {strat_def.get('requirements', 'N/A')}
        UNACCEPTABLE IF: {strat_def.get('unacceptable_if', 'N/A')}
        RECOMMENDED FOCUS COLUMN: {focus_col}

        ### FEW-SHOT DEMONSTRATIONS
        {good_shots if good_shots else "No examples provided."}

        ### CURRENT TASK
        DATA TO ANALYZE:
        Group Name: {table.get('Group name')}
        Table Data: {table.get('Table data - cleaned')}

        Apply the strategy. If it fits, generate JSON. If not, return NULL OUTPUT.
        """

    async def generate_questions(self):
        # table_idx -> how many strategies we have tried for it
        remaining_tables = {i: 0 for i in range(len(self.input_tables))}
        round_num = 1

        print(f"🚀 Starting Question Generation for {len(self.input_tables)} tables...")

        while remaining_tables:
            active_batch_indices = []
            batch_prompts = []
            batch_strategies = []

            for idx in list(remaining_tables.keys()):
                table = self.input_tables[idx]
                strat_list = list(table.get('Top strategies', []))

                # 1. Shuffle to break ties for strategies with the same count
                random.shuffle(strat_list)

                # 2. Sort by count (strategies with 0 will now be in random order at the front)
                strat_list.sort(key=lambda s: self.global_strategy_counts[s])

                attempt_idx = remaining_tables[idx]

                if attempt_idx < len(strat_list):
                    selected_strat = strat_list[attempt_idx]

                    # "Soft-reserve" this strategy for this batch so the next
                    # table in the SAME for-loop is pushed to a different one
                    self.global_strategy_counts[selected_strat] += 1

                    batch_prompts.append(self._prepare_prompt(table, selected_strat))
                    batch_strategies.append(selected_strat)
                    active_batch_indices.append(idx)
                else:
                    self.stats["failed_all_strategies"] += 1
                    del remaining_tables[idx]

            if not batch_prompts:
                break

            print(f"--- Round {round_num}: Processing {len(active_batch_indices)} tables ---")
            raw_responses = await self.model.async_run(batch_prompts)

            for i, response in enumerate(raw_responses):
                table_idx = active_batch_indices[i]
                parsed = self._parse_json(response)

                if parsed and parsed.get('question'):
                    strat_used = batch_strategies[i]
                    self.global_strategy_counts[strat_used] += 1
                    self._store_valid(self.input_tables[table_idx], parsed, strat_used)
                    del remaining_tables[table_idx]
                else:
                    if table_idx in remaining_tables:
                        remaining_tables[table_idx] += 1

            round_num += 1

        self.stats["final_success"] = len(self.valid_results)
        self._print_summary()
        return self.valid_results

    def _store_valid(self, table, result, strategy_used):
        # Use the provided 'Entity column' key directly
        entity_col = table.get('Entity column')

        # Handle data_column as a single string OR a list (for multi-col strategies)
        target_cols = result.get('data_column')
        if isinstance(target_cols, str):
            target_cols = [target_cols]

        relevant_data = []
        for row in table.get('Table data - cleaned', []):
            # Build a row containing the entity and all relevant data columns
            entry = {entity_col: row.get(entity_col)}
            for col in target_cols:
                entry[col] = row.get(col)
            relevant_data.append(entry)

        # Structured Output as requested (No noise)
        self.valid_results.append({
            "Group name": table['Group name'],
            "Question": result['question'],
            "Selected strategy": strategy_used,
            "Wiki link": table.get('Wikipedia link'),
            "Full table data": table.get('Table data - original'),
            "Relevant table data": relevant_data
        })

    @staticmethod
    def _parse_json(text):
        try:
            if "{" in text:
                return json.loads(text[text.find("{"):text.rfind("}") + 1])
            return None
        except:
            return None

    def _print_summary(self):
        # 1. Define the master list of all 11 strategies
        all_strategies = [
            "AGGREGATION", "EXTREME_NUMERICAL", "TOP_K_RANKING",
            "COMPARATIVE_DIFFERENCE", "FILTER_COUNT_RANGE", "UNIQUE_DIVERSITY",
            "FREQUENCY_MODE", "FILTER_COUNT_CATEGORICAL", "GROUPED_LOOKUP",
            "EXTREME_TEMPORAL", "COUNT_TOTAL"
        ]

        # 2. Sort all strategies based on their global count (descending)
        # We use .get(s, 0) to handle strategies that haven't been used yet
        sorted_strategies = sorted(all_strategies, key=lambda s: self.global_strategy_counts.get(s, 0), reverse=True)

        print("\n" + "═" * 50)
        print("📊 QUESTION GENERATION SUMMARY".center(50))
        print("═" * 50)
        print(f"Initial Tables:      {self.stats['initial_tables']}")
        print(f"Successfully Gen:    {self.stats['final_success']}")
        print(f"Exhausted Strats:    {self.stats['failed_all_strategies']}")
        print("-" * 50)
        print("Strategy Distribution:")

        for s in sorted_strategies:
            count = self.global_strategy_counts.get(s, 0)
            # Optional: highlight the 0s or high numbers with padding for alignment
            print(f"  - {s:<26}: {count}")

        print("═" * 50)

#
# class QuestionGenerator:
#     def __init__(self, input_tables, strategies):
#         self.input_tables = input_tables
#
#         # Check if the JSON is wrapped in a "strategies" key
#         if isinstance(strategies, dict) and "strategies" in strategies:
#             self.strategies = strategies["strategies"]
#         else:
#             self.strategies = strategies
#
#         self.model = gptAPI(QUESTION_PROMPT,
#                             output_format='string',
#                             temperature=0.2)  # Lowered temp for even stricter logic
#
#         self.valid_results = []
#         self.stats = {
#             "initial_tables": len(input_tables),
#             "final_success": 0,
#             "failed_after_3_tries": 0
#         }
#
#     def _prepare_prompt(self, table, strategy_name):
#         strat = self.strategies[strategy_name]
#
#         # Use .get() with defaults to prevent KeyErrors if the JSON is missing a field
#         definition = strat.get('definition', 'No definition provided.')
#         requirements = strat.get('requirements', 'No specific requirements.')
#         unacceptable = strat.get('unacceptable_if', 'No specific exclusion criteria.')
#
#         # --- Build the Few-Shot Block ---
#         good_shots = ""
#         # Default to an empty list if 'good_examples' is missing
#         for i, ex in enumerate(strat.get('good_examples', []), 1):
#             good_shots += f"GOOD EXAMPLE {i}:\nInput Group: {ex.get('input_group', 'N/A')}\nOutput Question: {ex.get('output', 'N/A')}\n\n"
#
#         bad_shots = ""
#         for i, ex in enumerate(strat.get('bad_examples', []), 1):
#             bad_shots += f"BAD EXAMPLE {i}:\nInput Group: {ex.get('input_group', 'N/A')}\nReason: {ex.get('reason', 'N/A')}\nOutput: {ex.get('output', 'N/A')}\n\n"
#
#         # --- Construct the Final User Input ---
#         user_input = f"""
#         ### STRATEGY TO APPLY: {strategy_name}
#         DEFINITION: {definition}
#         HARD REQUIREMENTS: {requirements}
#         UNACCEPTABLE IF: {unacceptable}
#
#         ### FEW-SHOT DEMONSTRATIONS
#         {good_shots if good_shots else "No examples provided."}
#         {bad_shots if bad_shots else "No examples provided."}
#
#         ### CURRENT TASK
#         DATA TO ANALYZE:
#         Group Name: {table.get('Group name')}
#         Entity Column: {table.get('Entity column')}
#         Table Data (key: table_data): {table.get('Table data')}
#
#         Apply the strategy to the "DATA TO ANALYZE" above. If it fits, generate the JSON. If not, return the NULL OUTPUT.
#         """
#         return user_input
#
#     async def generate_questions(self, max_rounds=3):
#         remaining_tables = {i: [] for i in range(len(self.input_tables))}
#
#         for round_num in range(1, max_rounds + 1):
#             if not remaining_tables:
#                 break
#
#             print(f"--- Round {round_num}: Processing {len(remaining_tables)} tables ---")
#
#             current_batch_indices = list(remaining_tables.keys())
#             batch_prompts = []
#             batch_strategies = []
#
#             for idx in current_batch_indices:
#                 table = self.input_tables[idx]
#                 # Filter strategies that haven't been tried for THIS table yet
#                 untried = [s for s in self.strategies.keys() if s not in remaining_tables[idx]]
#
#                 if not untried:  # Safety if max_rounds > number of strategies
#                     del remaining_tables[idx]
#                     continue
#
#                 selected_strat = random.choice(untried)
#                 remaining_tables[idx].append(selected_strat)
#                 batch_prompts.append(self._prepare_prompt(table, selected_strat))
#                 batch_strategies.append(selected_strat)
#
#             # API Call
#             raw_responses = await self.model.async_run(batch_prompts)
#
#             for i, response in enumerate(raw_responses):
#                 table_idx = current_batch_indices[i]
#                 parsed = self._parse_json(response)
#
#                 if parsed and parsed.get('question'):
#                     self._store_valid(self.input_tables[table_idx], parsed)
#                     del remaining_tables[table_idx]
#                 else:
#                     if round_num == max_rounds:
#                         del remaining_tables[table_idx]
#                         self.stats["failed_after_3_tries"] += 1
#
#         self.stats["final_success"] = len(self.valid_results)
#         self._print_summary()
#         return self.valid_results
#
#     def _parse_json(self, text):
#         try:
#             if "{" in text:
#                 json_str = text[text.find("{"):text.rfind("}") + 1]
#                 return json.loads(json_str)
#             return None
#         except Exception:
#             return None
#
#     def _store_valid(self, table, result):
#         table_data = table.get('Table data', [])
#         data_col = result.get('data_column')
#         entity_col = table.get('Entity column')
#
#         entity_data_map = {}
#         if data_col and entity_col:
#             for row in table_data:
#                 entity_name = str(row.get(entity_col, "Unknown Entity"))
#                 # Note: table_data row keys might be case-sensitive depending on your input
#                 value = row.get(data_col)
#                 if entity_name != "Unknown Entity":
#                     entity_data_map[entity_name] = value
#
#         self.valid_results.append({
#             "group_name": table['Group name'],
#             "strategy": result.get('selected_strategy'),
#             "thought": result.get('thought_process'),
#             "question": result['question'],
#             "answer_map": entity_data_map,
#             "metadata": {
#                 "wiki_page": table.get('Wikipedia page'),
#                 "target_column": data_col,
#                 "entity_column": entity_col
#             }
#         })
#
#     def _print_summary(self):
#         print("\n" + "═" * 50)
#         print("🚀 QUESTION GENERATION COMPLETE")
#         print("═" * 50)
#         print(f"Initial Tables:      {self.stats['initial_tables']}")
#         print(f"Successfully Gen:    {self.stats['final_success']}")
#         print(f"Failed (3 Rounds):   {self.stats['failed_after_3_tries']}")
#         print("═" * 50)