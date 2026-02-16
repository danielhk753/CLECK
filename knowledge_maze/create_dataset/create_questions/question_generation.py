import os
import re
import json
import random
import pandas as pd
import tqdm.auto as tqdm_lib
from collections import Counter
from json import JSONDecodeError
from IPython.display import display, HTML

from knowledge_maze.tools.api_tools import gptAPI

QUESTION_PROMPT = """
### ROLE
You are an expert Data Journalist. Your task is to generate a natural, professional question based on a dataset.

### NATURAL LANGUAGE CONSTRAINTS
- **THE HIERARCHY RULE**: Always prioritize the "Group Name" over the "Wikipedia Page Title." The Page Title is the background, 
but the Group Name is the specific subject you are reporting on.
- **SCOPE DEFINITION**: You MUST include the timeframe, category, or specific condition mentioned in the Group Name.
- **ENTITY MATCHING**: Ensure the subject (e.g., "universities", "players") matches the actual entities in the rows.
- **IMPLICIT SUBJECT**: Ensure the question is self-contained. Name the specific entity if the table focuses on one (e.g., "The US National Team").
- **NO SPOILERS & NO POINTERS**: Do not mention "the table" and do not include the answer in the question.

### TASK GUIDELINES
1. **STRATEGY**: Apply the {strategy_name} logic.
2. **COLUMN FOCUS**: You are encouraged to use the '{focus_column}' column.
3. **SELF-CONTAINED**: The question must make sense on its own.
4. **FALLBACK PROTOCOL**: If the data provided is insufficient or the strategy is impossible to apply accurately, do NOT 
return a question. Instead, set the "question" field to null and provide a detailed explanation in the "rejection_reason" field.

### OUTPUT FORMAT (Strict JSON)
{
  "thought_process": "Briefly explain why this question is natural and how it uses the relevant columns.",
  "question": "The natural language question (or null if the strategy is impossible to apply)",
  "rejection_reason": "If question is null, explain exactly why (e.g., 'Only 1 row found, cannot rank', 'Focus column has no numerical variance')",
  "data_column": "The exact header name used (or a list if multiple). Leave empty if question is null."
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
            'note', 'comments', 'comment', 'remarks', 'remark', 'description', 'ordinal'
        }
        self.tables = tables
        self.sanitized_tables = []

    @staticmethod
    def _classify_value(val, col_header=""):
        val = str(val).strip()
        if not val or val in ["—", "-", "N/A", "None"]:
            return "empty", None

        # 0. STRIP WIKIPEDIA REFERENCES (e.g., [a], [12])
        # This prevents "[a]" from breaking your numerical check
        val_clean = re.sub(r'\[[^]]*]', '', val).strip()

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

        # 1.5 RANKING CHECK (e.g., 1st, 2nd, 3rd, 10th)
        # Catches the pattern: digit(s) followed by st, nd, rd, or th
        if re.search(r'^\d+(st|nd|rd|th)$', val_clean, re.I):
            return "string", val_clean

        # 2. STRICT NUMERICAL CHECK
        # 1. Matches digits at the start
        # 2. Allows optional space
        # 3. Allows units like MW, kW, km/h, mph
        # 4. Stops before parentheses to avoid conversion confusion
        num_match = re.match(r'^[£$]?([\d,.]+)\s*([a-z/%]+)?$', val_clean, re.I)

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
                except (KeyError, IndexError):
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
            is_integer_col = False
            numeric_candidates = []

            try:
                # Check if values look like integers (including 1.0, 2.0)
                # We use a regex that allows digits and optionally .0, .00, etc.
                if all(re.match(r'^\d+(\.0+)?$', str(v).strip()) for v in raw_values):
                    numeric_candidates = [int(float(v)) for v in raw_values]
                    is_integer_col = True
            except (ValueError, TypeError):
                is_integer_col = False

            if is_integer_col and numeric_candidates:
                numeric_vals = sorted(numeric_candidates)
                # Perfect 1-to-N sequence check
                if numeric_vals == list(range(1, len(valid_entries) + 1)):
                    # print(f"🗑️ Trashing index column: {col}")
                    column_class[col] = "trash"
                    continue

            types = [c[0] for c in valid_entries]
            dominant_type = Counter(types).most_common(1)[0][0]

            # 4. FINAL CLASSIFICATION LOGIC
            if dominant_type == "string":

                # --- ADD THIS CHECK HERE ---
                is_ranking = any(re.search(r'\d+(st|nd|rd|th)', str(v), re.I) for v in raw_values)

                # Update this IF condition to include is_ranking
                if (1 < len(unique_vals) < len(valid_entries)) or is_ranking:
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
    def __init__(self, input_tables):
        self.sanitizer = TableSanitizer(input_tables)
        self.sanitizer.process_tables()
        self.data = self.sanitizer.sanitized_tables
        self.updated_data = []

    class StrategyContext:
        """
        A helper class to calculate and store table statistics once,
        so we don't re-compute them for every strategy check.
        """
        def __init__(self, sample):
            self.data = sample['Table data - cleaned']
            self.classes = sample['Column class']
            self.group_name = sample['Group name'].lower()
            self.row_count = len(self.data)

            # 1. Map columns by type for easy access
            self.cols_by_type = {
                "numerical": [c for c, cls in self.classes.items() if cls == "numerical"],
                "list": [c for c, cls in self.classes.items() if cls == "list"],
                "temporal": [c for c, cls in self.classes.items() if cls == "temporal"],
                "entity": [c for c, cls in self.classes.items() if cls == "entity"]
            }

            self.entity_col = self.cols_by_type["entity"][0] if self.cols_by_type["entity"] else None
            self.total_cols = len(self.classes)

            # 2. Add the missing density attributes
            if self.total_cols > 0:
                self.num_density = len(self.cols_by_type["numerical"]) / self.total_cols
                self.list_density = len(self.cols_by_type["list"]) / self.total_cols
                self.temp_density = len(self.cols_by_type["temporal"]) / self.total_cols
            else:
                self.num_density = self.list_density = self.temp_density = 0

            # 3. Boolean Flags
            self.has_num = len(self.cols_by_type["numerical"]) > 0
            self.has_list = len(self.cols_by_type["list"]) > 0
            self.has_time = len(self.cols_by_type["temporal"]) > 0
            self.has_entity = self.entity_col is not None

            # 4. Validity & Completeness checks for entity column
            # GLOBAL FILTER: Validating the Entity Column
            self.is_valid_table = False
            if self.entity_col and self.row_count > 0:
                raw_vals = [row.get(self.entity_col) for row in self.data]

                # 1. Check for None or N/A (Fullness)
                has_nulls = any(v is None or str(v).lower() in ['n/a', 'nan', 'none', 'null', ''] for v in raw_vals)

                # 2. Check for "Not Totally Numeric"
                # We want entities like "Apple" or "ID-105", not just "105"
                try:
                    # If we can convert every single value to a float, it's "too numeric" for an entity
                    [float(str(v)) for v in raw_vals]
                    is_purely_numeric = True
                except (ValueError, TypeError):
                    is_purely_numeric = False

                # Final Validation Flag
                if not has_nulls and not is_purely_numeric:
                    self.is_valid_table = True

    # =========================================================================
    # Strategy Evaluation Modules
    # =========================================================================

    @staticmethod
    def _handle_small_tables(ctx, scores):
        num_cols = ctx.cols_by_type["numerical"]
        time_cols = ctx.cols_by_type["temporal"]

        if ctx.row_count == 2 and ctx.has_num:
            scores["COMPARATIVE_DIFFERENCE"] = {"score": 1.0, "headers": num_cols}
        elif ctx.row_count == 3:
            if ctx.has_num:
                scores["EXTREME_NUMERICAL"] = {"score": 0.9, "headers": num_cols}
                scores["COMPARATIVE_DIFFERENCE"] = {"score": 0.8, "headers": num_cols}
            if ctx.has_time:
                scores["EXTREME_TEMPORAL"] = {"score": 0.9, "headers": time_cols}

    @staticmethod
    def _evaluate_entity_strategies(ctx, scores):
        """
        COUNT_TOTAL
        - Goal: Ask about the total number of unique items in a list.
        - Conditions: Table must have 4+ rows, an 'entity' column with 100% unique values,
          and a title that doesn't already "leak" the count (e.g., "Top 10 players").
        """
        if not ctx.has_entity: return

        ent_col = ctx.entity_col
        unique_entities = set(str(row.get(ent_col)).lower() for row in ctx.data if row.get(ent_col))

        if len(unique_entities) != ctx.row_count or ctx.row_count < 4:
            return

        # Triviality & Keyword checks (Logic same as before)
        title_numbers = [int(s) for s in ctx.group_name.split() if s.isdigit()]
        if any(n == ctx.row_count for n in title_numbers): return

        count_blocking = {"top", "ranked", "leading", "best", "roster", "squad"}
        if any(word in ctx.group_name for word in count_blocking): return

        score = 0.6 if any(w in ctx.group_name for w in ["recipients", "members", "results"]) else 0.46
        scores["COUNT_TOTAL"] = {"score": score, "headers": [ent_col]}

    @staticmethod
    def _evaluate_categorical_strategies(ctx, scores):
        """
        UNIQUE_DIVERSITY
        - Goal: Ask about the variety of categories present.
        - Conditions: Triggered when a column has a very high ratio of unique labels
          (e.g., a list of 20 movies where every movie has a different director).

        FREQUENCY_MODE
        - Goal: Identify the "most common" category (The Mode).
        - Conditions: Requires at least 3 unique categories and a clear "winner"
          (no ties for 1st place). Boosted when one category significantly dominates.

        CATEGORICAL_FILTER_LOOKUP
        - Goal: Group items by category and perform a lookup.
        - Conditions: Triggered by "clumping"—when values in a category column repeat
          frequently, allowing for questions like "Which players belong to Team X?".
        """
        if not ctx.has_list: return

        best_freq = {"score": 0, "col": None}
        best_div = {"score": 0, "col": None}
        clumping_cols = []

        for col in ctx.cols_by_type["list"]:

            # Fix: check for percent data that got here by accident
            if any(pct in col.lower() for pct in ["%", "percent", "rate", "ratio"]):
                continue

            vals = [str(row.get(col)) for row in ctx.data if row.get(col)]
            if not vals: continue

            counts = Counter(vals)
            num_total, num_unique = len(vals), len(counts)
            top_entries = counts.most_common(2)

            # UNIQUE_DIVERSITY logic
            div_score = 0.8 + (0.2 * (num_unique / num_total))
            if div_score > best_div["score"]:
                best_div = {"score": div_score, "col": col}

            # FREQUENCY_MODE & CLUMPING logic
            if (num_total - num_unique) >= 2:
                clumping_cols.append(col)
                is_not_tie = len(top_entries) > 1 and top_entries[0][1] > top_entries[1][1]
                if num_unique >= 3 and is_not_tie:
                    freq_score = 0.85 + (0.1 * (top_entries[0][1] / num_total))
                    if freq_score > best_freq["score"]:
                        best_freq = {"score": freq_score, "col": col}

        if best_div["col"]:
            scores["UNIQUE_DIVERSITY"] = {"score": best_div["score"], "headers": [best_div["col"]]}

        if best_freq["col"]:
            scores["FREQUENCY_MODE"] = {"score": best_freq["score"], "headers": [best_freq["col"]]}

        if clumping_cols:
            # For Lookup, we need the category column + entity or number to look up
            lookup_headers = [clumping_cols[0]]
            if ctx.has_entity:
                lookup_headers.append(ctx.entity_col)
            elif ctx.has_num:
                lookup_headers.append(ctx.cols_by_type["numerical"][0])

            # For CATEGORICAL_FILTER_LOOKUP - make sure the entities are unique
            ent_col = ctx.entity_col
            unique_entities = set(str(row.get(ent_col)).lower() for row in ctx.data if row.get(ent_col))

            if len(unique_entities) == ctx.row_count:
                scores["CATEGORICAL_FILTER_LOOKUP"] = {"score": 0.8, "headers": lookup_headers}

    @staticmethod
    def _evaluate_numerical_strategies(ctx, scores):
        """
        EXTREME_NUMERICAL
        - Goal: Find the highest/lowest values (Maximum/Minimum).
        - Conditions: Favored when data contains "outliers" (values more than 2x the average).

        TOP_K_RANKING
        - Goal: Ask about the 2nd, 3rd, or "Top X" items.
        - Conditions: Requires 5+ rows and high data uniqueness (no messy ties in the rankings).

        AGGREGATION
        - Goal: Calculate Sums or Averages.
        - Conditions: Triggered by keywords (Revenue, Total) or "Safe-to-Sum" columns
          (positive numbers that aren't Years, Ranks, or IDs). Blocked for Percentages.

        COMPARATIVE_DIFFERENCE
        - Goal: Compare the numerical gap between two specific entities.
        - Conditions: Dominant in small tables (2-5 rows) but remains active in large
          tables if a clear 'entity' exists to be compared.

        FILTER_COUNT_NUMERICAL
        - Goal: Count how many rows meet a specific numerical threshold.
        - Conditions: Requires "clumped" numbers (repeated values), such as many items
          having the exact same price or score.
        """
        if not ctx.has_num: return

        # 1. VARIANCE CHECK & COLUMN IDENTIFICATION
        valid_num_cols = []
        max_uniqueness = 0.0
        min_uniqueness = 1.0

        for col in ctx.cols_by_type["numerical"]:
            try:
                vals = [float(row[col]) for row in ctx.data if row.get(col) is not None]
                if vals and max(vals) > min(vals):
                    valid_num_cols.append(col)
                    ratio = len(set(vals)) / len(vals)
                    max_uniqueness = max(max_uniqueness, ratio)
                    min_uniqueness = min(min_uniqueness, ratio)
            except (ValueError, TypeError):
                continue

        if not valid_num_cols: return

        # 2. CONTEXT & KEYWORD ANALYSIS
        # Aggregation Triggers
        agg_keywords = {"total", "gross", "income", "revenue", "average", "sum", "gva", "gdp", "profit", "capacity",
                        "cost", "budget", "salary"}
        # Aggregation Blockers (Things you shouldn't sum)
        non_summable_keywords = {"rank", "year", "date", "id", "code", "zip", "latitude", "longitude"}
        # Percentage/Subset Triggers
        percent_keywords = {"percent", "%", "share", "rate", "ratio", "density"}
        subset_keywords = {"top", "highest", "best", "rankings", "leaders", "most"}

        name_words = set(ctx.group_name.split())

        # Flags
        is_agg_context = any(k in name_words for k in agg_keywords)
        is_pct_context = any(k in name_words for k in percent_keywords)
        is_subset_context = any(k in name_words for k in subset_keywords)

        header_is_agg = any(any(k in col.lower() for k in agg_keywords) for col in valid_num_cols)
        header_is_pct = any(any(k in col.lower() for k in percent_keywords) for col in valid_num_cols)

        # Check if headers contain "Year", "Rank", "ID" which makes them bad for aggregation
        header_is_bad_agg = any(any(k in col.lower() for k in non_summable_keywords) for col in valid_num_cols)

        # 3. SCORING LOGIC

        # 1. Calculate Outlier/Extreme Factor
        # If the max is much larger than the mean, 'EXTREME' is a great strategy.
        extreme_boost = 0
        for col in valid_num_cols:
            vals = [float(row[col]) for row in ctx.data if row.get(col) is not None]
            if vals:
                mean_val = sum(vals) / len(vals)
                # If the max is > 2x the mean, we have a clear "Extreme" candidate
                if max(vals) > (mean_val * 2):
                    extreme_boost = 0.15
                    break

        # 2. REBALANCED SCORING

        # A. EXTREME_NUMERICAL (The "Outlier" Specialist)
        # Base: 0.75. Boosted to 0.90 if there's a significant outlier.
        scores["EXTREME_NUMERICAL"] = {"score": 0.75 + extreme_boost, "headers": valid_num_cols}

        # B. FILTER_COUNT (The "Clump" Specialist)
        # Boosted to 0.85 if data repeats.
        scores["FILTER_COUNT_NUMERICAL"] = {"score": 0.85 if min_uniqueness <= 0.8 else 0.65, "headers": valid_num_cols}

        # C. TOP_K_RANKING (The "Unique" Specialist)
        # Boosted to 0.86 if data is highly unique (cleaner ranking).
        if ctx.row_count > 4:
            scores["TOP_K_RANKING"] = {"score": 0.86 if max_uniqueness > 0.9 else 0.76, "headers": valid_num_cols}

        # D. AGGREGATION (The "Sum" Specialist)
        # Keywords: 0.95 | Safe Sum: 0.84 | Bad context: 0.3
        if (is_agg_context or header_is_agg) and not (is_pct_context or header_is_pct or is_subset_context):
            scores["AGGREGATION"] = {"score": 0.95, "headers": valid_num_cols}
        elif not (is_pct_context or header_is_pct or is_subset_context or header_is_bad_agg):
            scores["AGGREGATION"] = {"score": 0.84, "headers": valid_num_cols}
        else:
            scores["AGGREGATION"] = {"score": 0.3, "headers": valid_num_cols}

        # E. COMPARATIVE_DIFFERENCE (The "Duo" Specialist)
        # Small tables: 0.94 | Entity present: 0.83
        if ctx.row_count <= 5:
            scores["COMPARATIVE_DIFFERENCE"] = {"score": 0.94, "headers": valid_num_cols}
        elif ctx.has_entity:
            scores["COMPARATIVE_DIFFERENCE"] = {"score": 0.83, "headers": valid_num_cols}
        else:
            scores["COMPARATIVE_DIFFERENCE"] = {"score": 0.5, "headers": valid_num_cols}

    @staticmethod
    def _evaluate_temporal_strategies(ctx, scores):
        """
        EXTREME_TEMPORAL
        - Goal: Find the "Earliest" or "Latest" event.
        - Conditions: Requires a time column with at least two different years. Blocked
          if the title already specifies the date range (to avoid triviality).

        FILTER_COUNT_TEMPORAL
        - Goal: Count events occurring in a specific year or era.
        - Conditions: Triggered when multiple events are clustered within the same
          timeframe, allowing questions like "How many events happened in 1995?".
        """
        if not ctx.has_time: return
        time_cols = ctx.cols_by_type["temporal"]

        # Check variance
        all_years = set()
        for col in time_cols:
            all_years.update([str(row.get(col))[:4] for row in ctx.data if row.get(col)])

        if len(all_years) < 2: return

        scores["EXTREME_TEMPORAL"] = {"score": 0.85, "headers": time_cols}
        scores["FILTER_COUNT_TEMPORAL"] = {"score": 0.8, "headers": time_cols}

    # =========================================================================
    # Wrapper and Data Slicing
    # =========================================================================

    def get_top_strategies(self, data_sample):
        ctx = self.StrategyContext(data_sample)
        # scores look like: {"STRATEGY": {"score": 0.9, "headers": ["ColA", "ColB"]}}
        scores = {}
        threshold = 0.45

        if ctx.row_count < 2:
            return {}

        if not ctx.is_valid_table:
            return {}

        if ctx.row_count <= 3:
            self._handle_small_tables(ctx, scores)
        else:
            self._evaluate_entity_strategies(ctx, scores)
            self._evaluate_categorical_strategies(ctx, scores)
            self._evaluate_numerical_strategies(ctx, scores)
            self._evaluate_temporal_strategies(ctx, scores)

            # 1. Filter by threshold
            valid = {k: v for k, v in scores.items() if v['score'] >= threshold}

            # 2. DATA COMPLETENESS & CONDITIONAL PRUNING
            null_markers = {None, "", "n/a", "nan", "null", "none"}
            final_valid = {}

            for strategy, info in valid.items():
                original_headers = info['headers']

                # Split headers into Entity vs Data columns
                data_cols = [c for c in original_headers if c != ctx.entity_col]
                has_initial_data_cols = len(data_cols) > 0

                # If there are data columns, we must prune the ones with nulls
                clean_data_cols = []
                if has_initial_data_cols:
                    for col in data_cols:
                        is_col_full = True
                        for row in ctx.data:
                            val = str(row.get(col, "")).lower().strip()
                            if not val or val in null_markers:
                                is_col_full = False
                                break
                        if is_col_full:
                            clean_data_cols.append(col)

                    # DISCARD CHECK: If we started with data columns but now have zero, discard strategy
                    if not clean_data_cols:
                        continue

                # Reconstruct the headers (Entity + whatever clean data columns remain)
                final_headers = []
                if ctx.entity_col in original_headers:
                    final_headers.append(ctx.entity_col)
                final_headers.extend(clean_data_cols)

                final_valid[strategy] = {
                    'score': info['score'],
                    'headers': final_headers
                }

            # 3. Sort by score and take top 3
            sorted_keys = sorted(final_valid, key=lambda k: final_valid[k]['score'], reverse=True)[:4]

            return {k: final_valid[k]['headers'] for k in sorted_keys}

    def get_strategies(self):
        for table in tqdm_lib.tqdm(self.data, desc="Slicing strategy views"):
            # strat_map: {"FREQUENCY_MODE": ["Genre"], "COUNT_TOTAL": ["Entity"]}
            strat_map = self.get_top_strategies(table)

            if not strat_map:
                continue

            entity_col = next((k for k, v in table['Column class'].items() if v == 'entity'), None)
            strategy_views = {}

            for strat, headers in strat_map.items():
                # 1. Define the exact order: Entity first, then the rest
                ordered_cols = []
                if entity_col:
                    ordered_cols.append(entity_col)

                # Add headers that aren't the entity column to avoid duplicates
                for h in headers:
                    if h != entity_col:
                        ordered_cols.append(h)

                # 2. Build the sliced row using the defined order
                # Because Python dicts preserve insertion order, 'entity_col'
                # will always be the first key.
                strategy_views[strat] = [
                    {col: row.get(col) for col in ordered_cols}
                    for row in table["Table data - cleaned"]
                ]

            self.updated_data.append({
                "Group name": table["Group name"],
                "Wikipedia link": table["Wikipedia link"],
                "Top strategies": list(strat_map.keys()),
                "Strategy Views": strategy_views,
                "Entity column": entity_col
            })

        self._print_summary()

    # =========================================================================
    # Summary & Data Quality Validation
    # =========================================================================

    def _print_summary(self):
        print(f"\n✅ Strategy Selection Complete.")
        passed = len(self.updated_data)
        initial_count = len(self.data)

        # 1. Calculate Strategy Distribution
        # We flatten the list of 'Top strategies' from every passed table
        all_assigned_strats = [s for sample in self.updated_data for s in sample["Top strategies"]]
        strat_counts = Counter(all_assigned_strats)

        print("\n" + "═" * 55)
        print(f"📊 STRATEGY SELECTION SUMMARY")
        print("-" * 55)
        print(f"✅ Groups Passed:     {passed}")
        print(f"❌ Groups Discarded:  {initial_count - passed}")
        print(f"📈 Success Rate:      {(passed / initial_count * 100) if initial_count else 0:.1f}%")
        print("-" * 55)
        print("Strategy Distribution (Top 4 per table):")

        # Sort for the text print
        for strat, count in strat_counts.most_common():
            print(f"  - {strat: <26}: {count}")
        print("-" * 55)

        density_counts = Counter([len(s['Strategy Views'].keys()) for s in self.updated_data])
        total_groups = sum(density_counts.values())

        print(f"{'POSSIBLE STRATEGIES':<20} | {'GROUP COUNT':<12} | {'PERCENTAGE'}")
        print("-" * 55)
        for num_strats in sorted(density_counts.keys()):
            count = density_counts[num_strats]
            pct = (count / total_groups) * 100
            print(f"#{num_strats:<19} | {count:<12} | {pct:>9.1f}%")
        print("═" * 55 + "\n")

    def debug_strategy_samples(self, target_strategy, n=5):
        """
        Filters the updated_data for a specific strategy and displays n random samples,
        including the formal definition from question_strategies.json.
        """
        # 1. Load Strategy Definition
        strat_desc = "No formal definition found."
        requirements = "N/A"
        unacceptable_if = []

        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, "question_strategies.json")

            with open(json_path, 'r') as f:
                strategies_ref = json.load(f)['strategies']
                if target_strategy in strategies_ref:
                    config = strategies_ref[target_strategy]
                    strat_desc = config.get("definition", strat_desc)
                    requirements = config.get("requirements", "No specific requirements listed.")
                    u_if = config.get("unacceptable_if", [])
                    unacceptable_if = u_if if isinstance(u_if, list) else [u_if]
        except Exception as e:
            strat_desc = f"Could not load definition: {e}"

        # 2. UI Header & Merged Logic Box
        unacceptable_bullets = "".join([f"<li style='margin-bottom:6px;'>{u}</li>" for u in unacceptable_if])

        display(HTML(f"""
            <div style="background-color: #ffffff; border: 1px solid #dee2e6; border-radius: 12px; overflow: hidden; font-family: sans-serif; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                <div style="background-color: #2e7d32; padding: 15px 20px;">
                    <h2 style="margin:0; color: #ffffff; font-size: 1.5em; letter-spacing: 0.5px;">
                        🔍 Strategy Audit: {target_strategy}
                    </h2>
                </div>

                <div style="display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 0px;">

                    <div style="background-color: #e8f5e9; padding: 20px; border-right: 1px solid #c8e6c9;">
                        <h4 style="margin-top: 0; color: #1b5e20; text-transform: uppercase; font-size: 0.85em; letter-spacing: 1px;">📜 Objective & Logic</h4>
                        <p style="font-size: 1.1em; color: #1b5e20; line-height: 1.5; font-weight: 500; margin-bottom: 15px;">
                            {strat_desc}
                        </p>
                        <div style="background: rgba(255,255,255,0.4); padding: 12px; border-radius: 6px; border: 1px dashed #2e7d32;">
                            <strong style="color: #1b5e20; font-size: 0.9em;">REQUISITES:</strong>
                            <p style="margin: 5px 0 0 0; font-size: 0.95em; color: #2e7d32; line-height: 1.4;">{requirements}</p>
                        </div>
                    </div>

                    <div style="background-color: #fff5f5; padding: 20px;">
                        <h4 style="margin-top: 0; color: #b71c1c; text-transform: uppercase; font-size: 0.85em; letter-spacing: 1px;">⚠️ Unacceptable If</h4>
                        <ul style="margin-top: 10px; margin-bottom: 0; font-size: 0.9em; color: #c62828; line-height: 1.4; padding-left: 18px;">
                            {unacceptable_bullets if unacceptable_if else "<li>No specific blockers listed.</li>"}
                        </ul>
                    </div>

                </div>
            </div>
        """))

        # 3. Fina eligible samples:
        eligible_samples = [s for s in self.updated_data if target_strategy in s['Strategy Views']]

        if not eligible_samples:
            print(f"❌ No samples found for strategy: {target_strategy}")
            return
        print(
            f"Found {len(eligible_samples)} total tables with this strategy. Showing {min(n, len(eligible_samples))} random samples.\n")

        # 4. Sampling & Display
        num_to_show = min(n, len(eligible_samples))
        samples = random.sample(eligible_samples, num_to_show)

        for i, sample in enumerate(samples):
            group_name = sample['Group name']
            filtered_data = sample['Strategy Views'][target_strategy]

            display(
                HTML(f"<div style='border: 1px solid #ddd; padding: 10px; margin-bottom: 20px; border-radius: 5px;'>"))
            display(HTML(f"<strong>Sample {i + 1}:</strong> {group_name}"))
            print(f"🔗 Source: {sample.get('Wikipedia link', 'N/A')}")

            # Display the DataFrame
            display(pd.DataFrame(filtered_data))
            display(HTML(f"</div>"))


class QuestionGenerator:
    def __init__(self, input_tables, strategy_definitions):
        self.input_tables = input_tables
        self.definitions = strategy_definitions.get("strategies", strategy_definitions)
        self.model = gptAPI(QUESTION_PROMPT,
                            output_format='string',
                            temperature=0.2,
                            max_concurrent_requests=100,
                            max_rpm=4900)

        # Core state tracking
        self.global_strategy_counts = Counter()  # Successful generations
        self.attempted_strategy_counts = Counter()  # Total attempts (the "Leveler")

        self.valid_results = []
        self.failed_logs = []

        self.stats = {
            "initial_tables": len(input_tables),
            "final_success": 0,
            "failed_all_strategies": 0,
        }

    def _prepare_strat_list(self, table):
        """
        Ensures uniform distribution of attempts by prioritizing 'starved' strategies.
        """
        # Pull from 'Top strategies' (now a dict from the selector)
        strat_map = table.get('Strategy Views', {})
        strat_list = list(strat_map.keys())

        # 1. Shuffle to ensure different tables don't dogpile the same strategy in a batch
        random.shuffle(strat_list)

        # 2. Sort by GLOBAL ATTEMPTS (Lowest attempts first)
        # This pushes under-tried strategies to index 0
        strat_list.sort(key=lambda s: self.attempted_strategy_counts[s])

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

        bad_shots = ""
        for i, ex in enumerate(strat_def.get('bad_examples', []), 1):
            bad_shots += f"BAD EXAMPLE {i}:\nInput Group: {ex.get('input_group', 'N/A')}\nOutput: {ex.get('output', 'N/A')}\n\n"

        return f"""
        ### STRATEGY TO APPLY: {strategy_name}
        DEFINITION: {strat_def.get('definition', 'N/A')}
        HARD REQUIREMENTS: {strat_def.get('requirements', 'N/A')}
        UNACCEPTABLE IF: {strat_def.get('unacceptable_if', 'N/A')}
        RECOMMENDED FOCUS COLUMN: {focus_col}

        ### FEW-SHOT DEMONSTRATIONS
        {good_shots if good_shots else "No good examples provided."}
        {bad_shots if bad_shots else "No bad examples provided."}

        ### CURRENT TASK
        DATA TO ANALYZE:
        Wikipedia Page Title: {table.get('Wikipedia link', 'N/A').split("/wiki/")[1].split("#")[0].replace("_", " ")}
        Group Name: {table.get('Group name')}
        Table Data: {table.get('Strategy Views')[strategy_name]}

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
                strat_list = self._prepare_strat_list(table)

                attempt_idx = remaining_tables[idx]

                if attempt_idx < len(strat_list):
                    selected_strat = strat_list[attempt_idx]

                    # Mark that we are trying this strategy
                    self.attempted_strategy_counts[selected_strat] += 1

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
                strat_used = batch_strategies[i]
                table = self.input_tables[table_idx]

                if parsed and parsed.get('question'):
                    self._store_valid(self.input_tables[table_idx], parsed, strat_used)
                    del remaining_tables[table_idx]
                else:
                    # DEBUG LOG: LLM REJECTION
                    self.failed_logs.append({
                        "reason": parsed['rejection_reason'] if parsed else None,
                        "strategy": strat_used,
                        "table_name": table.get('Group name'),
                        "table_data": table.get('Strategy Views')[strat_used],
                        "wiki_link": table.get('Wikipedia link')
                    })

                    # FAILURE: The model returned NULL or bad JSON.
                    # We MUST undo the increment so the stats stay clean!
                    self.global_strategy_counts[strat_used] -= 1
                    if table_idx in remaining_tables:
                        remaining_tables[table_idx] += 1

            round_num += 1

        self.stats["final_success"] = len(self.valid_results)
        self._print_summary()

    def _store_valid(self, table, result, strategy_used):
        # Use the provided 'Entity column' key directly
        entity_col = table.get('Entity column')

        # Handle data_column as a single string OR a list (for multi-col strategies)
        target_cols = result.get('data_column')
        if isinstance(target_cols, str):
            target_cols = [target_cols]

        relevant_data = []
        for row in table.get('Strategy Views', [])[strategy_used]:
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
        except (KeyError, IndexError, JSONDecodeError):
            return None

    def _print_summary(self):
        # Define the master list of all 11 strategies
        all_strategies = [
            "AGGREGATION", "EXTREME_NUMERICAL", "TOP_K_RANKING", "COMPARATIVE_DIFFERENCE",
            "FILTER_COUNT_NUMERICAL", "FILTER_COUNT_TEMPORAL", "UNIQUE_DIVERSITY",
            "FREQUENCY_MODE", "CATEGORICAL_FILTER_LOOKUP", "EXTREME_TEMPORAL", "COUNT_TOTAL"
        ]

        print("\n" + "═" * 65)
        print("📊 QUESTION GENERATION SUMMARY".center(65))
        print("═" * 65)
        print(f"Initial Tables:      {self.stats['initial_tables']}")
        print(f"Successfully Gen:    {self.stats['final_success']}")
        print(f"Exhausted Strats:    {self.stats['failed_all_strategies']}")
        print(f"Success Rate:        {(self.stats['final_success'] / self.stats['initial_tables']) * 100:.1f}%")
        print("-" * 65)

        # Table Header
        print(f"{'STRATEGY':<28} | {'SUCCESS':<8} | {'TRIED':<8} | {'RATE'}")
        print("-" * 65)

        # Calculate stats directly from the final successful list
        final_stats = Counter([sample['Selected strategy'] for sample in self.valid_results])
        sorted_strategies = sorted(all_strategies, key=lambda sx: final_stats.get(sx, 0), reverse=True)

        # Print the sorted rows
        total_tried, total_created = 0, 0
        for s in sorted_strategies:
            created = final_stats.get(s, 0)
            tried = self.attempted_strategy_counts.get(s, 0)
            rate_pct = (100 * created / tried) if tried > 0 else 0
            print(f"{s:<28} | {created:<8} | {tried:<8} | {rate_pct:>3.0f}%")

            total_tried += tried
            total_created += created

        total_rate_pct = (100 * total_created / total_tried) if total_tried > 0 else 0

        print("-" * 65)
        print(f"{'Total':<28} | {total_created:<8} | {total_tried:<8} | {total_rate_pct:>3.0f}%")
        print("═" * 65)

        if self.failed_logs:
            print("\n" + "🔍 SAMPLES OF FAILED DATA VIEWS:")
            print("-" * 65)

            # Display the first 3 failures for inspection
            for i, fail in enumerate(self.failed_logs[:3]):
                strat = fail['strategy']
                table_name = fail['table_name']
                reason = fail.get('llm_explanation', fail['reason'])
                wiki_link = fail.get('wiki_link')

                print(f"\n[{i + 1}] Table: {table_name}")
                print(f"    Original Data: {wiki_link}")
                print(f"    Strategy Attempted: {strat}")
                print(f"    Rejection Reason:   {reason}")
                print(f"    Table Data:")

                # Find the original table in input_tables to show its 'Strategy View'
                # (Assuming the original data is accessible via strategy_views)
                target_table = next((t for t in self.input_tables if t.get('Group name') == table_name), None)

                if target_table and 'Strategy Views' in target_table:
                    view_data = target_table['Strategy Views'].get(strat, [])
                    if view_data:
                        display(pd.DataFrame(view_data))
                    else:
                        print("    (No data available for this strategy view)")

                print("-" * 65)
