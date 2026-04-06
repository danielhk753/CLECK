"""
Automated Journalistic Question Generation Pipeline

This module provides a structured framework for extracting, analyzing, and
transforming tabular data into natural language questions. It utilizes a
multi-stage pipeline to ensure data integrity, strategic diversity, and
linguistic quality.

Core Components:
    1. TableSanitizer: Performs data type induction and structural cleaning.
       It identifies "trash" columns (indices, redundant rankings) and
       categorizes useful data into String, Numerical, or Temporal types.

    2. StrategySelector: Analyzes the sanitized schema to determine the
       most "journalistic" approach. It maps data shapes to specific
       logical frameworks like Aggregation, Comparison, or Extremes.

    3. QuestionGenerator: The LLM-interface layer that synthesizes the
       final natural language question using a strict "Journalistic Mandate"
       to avoid "AI-speak" and robotic framing.

    4. QuestionFilter: A final quality-gate that applies "Auto-Fail"
       criteria. It strips out common LLM hallucinations, redundant
       phrasing, and questions that provide the answer within the prompt.

Workflow:
    The pipeline accepts a raw table and a 'Group Name' (context), passing
    them through the Sanitizer. The Selector identifies valid columns for
    specific strategies, which the Generator uses to prompt an LLM.
    Finally, the Filter ensures only the "cleanest" questions reach the user.
"""

import os
import re
import json
import random
import difflib
import pandas as pd
from rapidfuzz import fuzz
import tqdm.auto as tqdm_lib
from collections import Counter
from json import JSONDecodeError
from IPython.display import display, HTML

from knowledge_maze.tools.utils import save_json
from knowledge_maze.tools.api_tools import OpenRouterAPI

QUESTION_PROMPT = """
### ROLE
You are an expert Data Journalist. Your task is to generate a natural, professional question based on a dataset.

### NATURAL LANGUAGE CONSTRAINTS
- **THE HIERARCHY RULE**: Always prioritize the "Group Name", it is the specific subject you are reporting on.
- **SCOPE DEFINITION**: You MUST include the timeframe, category, or specific condition mentioned in the Group Name.
- **ENTITY MATCHING**: Ensure the subject (e.g., "universities", "players") matches the actual entities in the rows.
- **IMPLICIT SUBJECT**: Ensure the question is self-contained. Name the specific entity if the table focuses on one (e.g., "The US National Team").
- **NO SPOILERS & NO POINTERS**: Do not mention "the table" and do not include the answer in the question.

### CATEGORICAL ACCURACY
- **Verify the Noun**: Look at the actual names in the "Entity" column. If the entities are "Babe Ruth" and "Barry Bonds," 
do NOT call them "pitchers" just because the Group Name is vague. Ensure the noun used (e.g., "cities," "athletes," "films") accurately describes every row provided.
- **Identify the Subset**: If the table contains a specific number of rows (e.g., 6), do not just say "these six." Instead, 
describe the shared trait that makes them a group (e.g., "The six MLB players who held the single-season home run record...").

### TASK GUIDELINES
1. **STRATEGY**: Apply the {strategy_name} logic.
2. **DATA INTEGRATION**: Use the specific metrics provided in the columns to ground the question (e.g., if the column is 'Revenue (Billion USD)', ensure the question asks about 'revenue in billions' rather than just 'earnings').
3. **SELF-CONTAINED**: The question must make sense on its own.
4. **FALLBACK PROTOCOL**: If the data provided is insufficient or the strategy is impossible to apply accurately, do NOT 
return a question. Instead, set the "question" field to null and provide a detailed explanation in the "rejection_reason" field.

### THE DIRECTNESS MANDATE
- **ASSUME UNIVERSAL TRUTH**: Do not cite the source inside the question. Instead of "Who is the oldest according to the records?", ask "Who is the oldest?" or "In the 2016 census, who was the oldest?".
- **JOURNALISTIC FLOW**: A journalist doesn't say "Which four buildings are the tallest based on their height in meters?" They say "Which four Ankara buildings currently dominate the skyline?" or "Which four buildings in Ankara are the tallest?"
- **NO DATA POINTERS**: If the table is about "Russian Soups," the question should just ask about "Russian soups," never "soups in this collection."

### THE PROHIBITED LIST (AI-SPEAK)
You MUST NOT use the following phrases or their variations. If you use them, the question is a failure:
- "According to the [records/data/table/census/list/timeline]"
- "In this collection" or "Listed in this group"
- "Based on the [provided data/height/metrics]"
- "As of the recorded [year]" (Use "In [year]" instead)
- "Identified in this..."
- "Pointer words": Never use "these", "those", "the following", or "the listed" to refer to the entities in the table.
- The "Radio Test": If a person hearing this question on the radio cannot identify who or what you are talking about, the question is a failure.

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
            '%', 'mph', 'kph', 's', 'sec', 'min', 'hr', 'h', 'days', 'years', 'yr', 'points', 'pts'
        }
        self.keywords_blacklist = {
            'rank', 'no.', 'no', '#', 'index', 'sl no', 'notes',
            'note', 'comments', 'comment', 'remarks', 'remark', 'description', 'ordinal'
        }
        self.tables = tables
        self.sanitized_tables = []

    def _classify_value(self, val, col_header=""):
        """
        Determines if a cell value is empty, temporal, numerical, or string.
        Includes logic to strip fuzzy prefixes (~, >) and validate units.
        """
        val = str(val).strip()
        if not val or val in ["—", "-", "N/A", "n/a", "None", "nan"]:
            return "empty", None

        # 0. STRIP NOISE (Wikipedia refs, fuzzy prefixes)
        # Removes: [1], [a], ~, >, <, approx.
        val_clean = re.sub(r'\[[^]]*]', '', val).strip()
        val_clean = re.sub(r'^[\s~><]+|(?<=[\d])[\s~><]+', '', val_clean).strip()
        val_clean = re.sub(r'\b(approx|est|estimated)\.?\b', '', val_clean, flags=re.I).strip()

        # Keywords that suggest this column MUST be numerical
        num_keywords = ["gross", "income", "revenue", "gva", "gdp", "profit", "capacity",
                        "attendance", "population", "area", "total", "budget", "salary"]
        header_is_num = any(k in col_header for k in num_keywords)

        # 1. TEMPORAL CHECK
        is_date_header = "date" in col_header.lower() or "year" in col_header.lower()
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

        # 1.5 RANKING CHECK (e.g., 1st, 2nd) -> String
        if re.search(r'^\d+(st|nd|rd|th)$', val_clean, re.I):
            return "string", val_clean

        # 2. STRICT NUMERICAL CHECK
        # Supports: -100, $50, 50%, 50kg, 1,000.00
        # Regex Breakdown:
        # ^[£$€...]?       Optional currency prefix
        # -?               Optional negative sign
        # ([\d,.]+)        The number (digits, commas, dots)
        # \s* : Optional space
        # ([a-zA-Z/%]+)?$ : Optional unit suffix
        num_match = re.match(r'^[£$€¥₹₽₩฿₫₪]?\s*(-?[\d,.]+)\s*([a-zA-Z/%]+)?$', val_clean)

        if num_match:
            number_part = num_match.group(1).replace(',', '')
            unit_part = num_match.group(2).lower() if num_match.group(2) else None

            # 2.1 Unit Validation
            # If a unit exists, it MUST be in VALID_METRICS (prevents "5 dogs" -> Numerical)
            if unit_part and unit_part not in self.VALID_METRICS:
                # Fallback: If header demands numbers, ignore the weird unit
                if not header_is_num:
                    return "string", val

            # 2.2 Sanity Check (Are there actually digits?)
            if not any(c.isdigit() for c in number_part):
                return "string", val

            try:
                # Handle "1.2.3" edge cases that regex accepts but float rejects
                if number_part.count('.') > 1:
                    return "string", val

                num_float = float(number_part)

                # Apply scaling words if they were detached from the regex or inside the string
                # (Simple check on original string for context)
                multiplier = 1
                if 'billion' in val.lower():
                    multiplier = 1e9
                elif 'million' in val.lower():
                    multiplier = 1e6

                return "numerical", num_float * multiplier
            except ValueError:
                pass

        # 3. FORCED NUMERICAL FALLBACK (Header Driven)
        if header_is_num:
            only_digits = re.sub(r'[^\d.]', '', val_clean)
            if only_digits:
                try:
                    return "numerical", float(only_digits)
                except ValueError:
                    pass

        return "string", val

    @staticmethod
    def _count_truly_distinct(values):
        """
        Counts distinct values using fuzzy matching logic.
        - Ignores case.
        - Groups substrings (e.g., 'USA' and 'USA (est)' count as 1).
        - Groups high-similarity strings.
        """
        if not values:
            return 0

        # 1. Normalize
        clean_vals = sorted([str(v).strip().lower() for v in values], key=len)
        clusters = []

        for v in clean_vals:
            matched = False
            for i, cluster_rep in enumerate(clusters):
                # Substring check: "apple" in "pineapples" (simple containment)
                # or Fuzzy match > 0.85
                if v in cluster_rep or cluster_rep in v:
                    matched = True
                    # Keep the longer/more descriptive one as rep?
                    # Doesn't matter for counting, just need to not add a new cluster.
                    break

                # Similarity Check (SequenceMatcher is fast enough for column-size lists)
                ratio = difflib.SequenceMatcher(None, v, cluster_rep).ratio()
                if ratio > 0.85:
                    matched = True
                    break

            if not matched:
                clusters.append(v)

        return len(clusters)

    def _process_table(self, sample):
        group_name = sample['Group name']
        table_data = sample['Table data']
        wiki_link = sample['Wiki url']
        ent_col = sample['Entity column']

        if not table_data:
            return None

        num_rows = len(table_data)
        col_headers = list(table_data[0].keys())
        column_class = {}

        for col in col_headers:
            clean_col = col.lower().strip()

            if clean_col in self.keywords_blacklist:
                column_class[col] = "trash"
                continue

            if col == ent_col:
                column_class[col] = "entity"
                continue

            # Classify all cells
            classifications = [self._classify_value(row.get(col), col_header=col) for row in table_data]
            valid_entries = [c for c in classifications if c[0] != "empty"]

            # 1. DENSITY CHECK
            if len(valid_entries) < (num_rows * 0.5):
                column_class[col] = "trash"
                continue

            # 2. INVARIANCE CHECK (Exact duplicates)
            raw_values = [str(c[1]) for c in valid_entries]
            unique_vals = set(raw_values)
            if len(unique_vals) <= 1:
                column_class[col] = "trash"
                continue

            # 3. ORDINAL/SEQUENCE CHECK
            try:
                # 1. Normalize values: strip 'st/nd/rd/th' and convert to int
                # This handles both "1" and "1st"
                seq_vals = []
                for v in raw_values:
                    clean_v = re.sub(r'(?i)st|nd|rd|th', '', str(v).strip())
                    if re.match(r'^\d+(\.0+)?$', clean_v):
                        seq_vals.append(int(float(clean_v)))
                    else:
                        break  # Not a pure numeric/ordinal column

                if len(seq_vals) == len(valid_entries):
                    # 2. Check for monotonic increment of exactly 1
                    # We sort them to see if they form a perfect contiguous range
                    sorted_seq = sorted(seq_vals)
                    start_val = sorted_seq[0]
                    expected_range = list(range(start_val, start_val + len(valid_entries)))

                    if sorted_seq == expected_range:
                        column_class[col] = "trash"
                        continue
            except (ValueError, TypeError):
                pass

            types = [c[0] for c in valid_entries]
            dominant_type = Counter(types).most_common(1)[0][0]

            # 4. FINAL CLASSIFICATION LOGIC
            if dominant_type == "string":
                is_ranking = any(re.search(r'\d+(st|nd|rd|th)', str(v), re.I) for v in raw_values)

                # --- Check if these are actually misclassified numbers ---
                # Example: "5 (est)", "6", "7" -> dominant string, but content is numeric
                # We try to parse the raw_values purely as numbers
                numeric_parsing_success = 0
                for v in raw_values:
                    # Loose check: digits present, no complex letters
                    if re.search(r'\d', v) and len(re.findall(r'[a-zA-Z]', v)) < 3:
                        numeric_parsing_success += 1

                if numeric_parsing_success > len(raw_values) * 0.9 and not is_ranking:
                    # Get the actual parsed numbers to check variance
                    potential_nums = []
                    for v in raw_values:
                        num_match = re.search(r'[\d.]+', v)
                        if num_match:
                            potential_nums.append(num_match.group())

                    # Force rescue ONLY if it's diverse enough
                    if len(set(potential_nums)) >= 4:
                        column_class[col] = "numerical"
                    else:
                        column_class[col] = "trash"

                # --- Stricter Categorical (List) Check ---
                elif self._count_truly_distinct(raw_values) >= 3 and len(set(raw_values)) <= len(raw_values) - 2:
                    column_class[col] = "list"
                else:
                    # Clean Short String Check
                    is_clean_short = all(
                        len(v.split()) <= 3 and all(c.isalnum() or c in " -()" for c in v)
                        for v in raw_values
                    )
                    column_class[col] = "trash"  # "other" if needed, but usually trash for QA

            elif dominant_type == "numerical":
                vals = [c[1] for c in valid_entries if c[0] == "numerical"]

                # --- VARIANCE CHECK ---
                # If there are fewer than 4 unique values, it's too static for numerical analysis
                if len(set(vals)) < 4:
                    column_class[col] = "trash"

                # Temporal Heuristic: All values between 1000 and 2100 (likely years)
                # But allow small variance if most look like years
                if all(1000 <= v <= 2100 for v in vals):
                    column_class[col] = "temporal"
                else:
                    column_class[col] = "numerical"
            else:
                column_class[col] = dominant_type

        # 5. CONSTRUCT CLEANED TABLE
        cleaned_table = []
        for row in table_data:
            # Only keep columns that aren't trash
            new_row = {col: row.get(col) for col, cls in column_class.items() if cls != "trash"}
            cleaned_table.append(new_row)

        final_classes = {k: v for k, v in column_class.items() if v != "trash"}

        return {
            "Group name": group_name,
            "Table data - cleaned": cleaned_table,
            "Table data - original": table_data,
            "Wiki url": wiki_link,
            "Column class": final_classes
        }

    def process_tables(self):
        for table in tqdm_lib.tqdm(self.tables, desc="Sanitizing tables"):
            res = self._process_table(table)
            if res:
                self.sanitized_tables.append(res)


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
                "Wiki url": table["Wiki url"],
                "Top strategies": list(strat_map.keys()),
                "Strategy Views": strategy_views,
                "Entity column": entity_col
            })

        self._print_summary()

    # =========================================================================
    # Summary & Data Quality Validation
    # =========================================================================

    def _print_summary(self):
        passed = len(self.updated_data)
        initial_count = len(self.data)
        discarded = initial_count - passed
        success_rate = (passed / initial_count * 100) if initial_count else 0

        # 1. Strategy Popularity (Flattened and Sorted)
        all_assigned_strats = [s for sample in self.updated_data for s in sample.get("Top strategies", [])]
        strat_counts = Counter(all_assigned_strats).most_common()

        # 2. Strategy Density (# of strategies per group)
        density_counts = Counter([len(s.get('Strategy Views', {}).keys()) for s in self.updated_data])
        total_groups = sum(density_counts.values())

        # Metric Cards (Top Row)
        stats_html = f"""
        <div style="display: flex; gap: 12px; margin-bottom: 25px;">
            <div style="flex: 1; background: #1e293b; padding: 15px; border-radius: 12px; border: 1px solid #334155; text-align: center;">
                <div style="color: #94a3b8; font-size: 10px; font-weight: 700; text-transform: uppercase;">Total Initial</div>
                <div style="font-size: 24px; font-weight: 800; color: #f8fafc;">{initial_count}</div>
            </div>
            <div style="flex: 1; background: #064e3b; padding: 15px; border-radius: 12px; border: 1px solid #059669; text-align: center;">
                <div style="color: #34d399; font-size: 10px; font-weight: 700; text-transform: uppercase;">Groups Passed</div>
                <div style="font-size: 24px; font-weight: 800; color: #34d399;">{passed}</div>
            </div>
            <div style="flex: 1; background: #450a0a33; padding: 15px; border-radius: 12px; border: 1px solid #991b1b66; text-align: center;">
                <div style="color: #fca5a5; font-size: 10px; font-weight: 700; text-transform: uppercase;">Discarded</div>
                <div style="font-size: 24px; font-weight: 800; color: #f87171;">{discarded}</div>
            </div>
            <div style="flex: 1; background: #1e293b; padding: 15px; border-radius: 12px; border: 1px solid #334155; text-align: center;">
                <div style="color: #38bdf8; font-size: 10px; font-weight: 700; text-transform: uppercase;">Success Rate</div>
                <div style="font-size: 24px; font-weight: 800; color: #38bdf8;">{success_rate:.1f}%</div>
            </div>
        </div>
        """

        # Popularity Rows (Left Table - Sorted)
        popularity_rows = ""
        for strat, count in strat_counts:
            popularity_rows += f"""
            <tr style="border-bottom: 1px solid #0f172a; background: #1e293b;">
                <td style="padding: 12px 15px; color: #f8fafc; text-align: left; font-size: 13px;">{strat}</td>
                <td style="padding: 12px 15px; color: #38bdf8; text-align: left; font-weight: 700;">{count}</td>
            </tr>
            """

        # Strategy Density Rows (Right Table)
        density_rows = ""
        for num_strats in sorted(density_counts.keys()):
            count = density_counts[num_strats]
            pct = (count / total_groups * 100) if total_groups else 0
            density_rows += f"""
            <tr style="border-bottom: 1px solid #0f172a; background: #1e293b;">
                <td style="padding: 12px 15px; color: #f8fafc; text-align: left; font-weight: 600;">{num_strats} Strategies</td>
                <td style="padding: 12px 15px; color: #34d399; text-align: left;">{count} <span style="color: #64748b; font-size: 11px; margin-left: 8px;">({pct:.1f}%)</span></td>
            </tr>
            """

        dashboard_html = f"""
        <div style="width: 1100px; background-color: #0f172a; padding: 30px; border-radius: 24px; font-family: sans-serif; color: #f8fafc; border: 1px solid #1e293b; margin-left: 0; text-align: left; box-sizing: border-box;">
            <h3 style="margin: 0 0 25px 0; font-size: 18px; color: #38bdf8; text-align: center; text-transform: uppercase; font-weight: 800; letter-spacing: 1px;">📊 Strategy Selection Insights</h3>
            {stats_html}

            <div style="display: flex; gap: 20px; align-items: stretch;">

                <div style="flex: 1.2; background: #1e293b; border-radius: 18px; border: 1px solid #334155; overflow: hidden; display: flex; flex-direction: column;">
                    <div style="padding: 15px 20px; background: #1e293b; border-bottom: 1px solid #334155; color: #94a3b8; font-size: 11px; font-weight: 700; text-transform: uppercase;">
                        Strategy Popularity (Sorted)
                    </div>
                    <div style="max-height: 400px; overflow-y: auto;">
                        <table style="width: 100%; border-collapse: collapse; background: #1e293b;">
                            <thead>
                                <tr style="background: #0f172a; color: #64748b; font-size: 10px; text-transform: uppercase;">
                                    <th style="padding: 10px 15px; text-align: left;">Strategy Name</th>
                                    <th style="padding: 10px 15px; text-align: left;">Selection Count</th>
                                </tr>
                            </thead>
                            <tbody>
                                {popularity_rows}
                            </tbody>
                        </table>
                    </div>
                </div>

                <div style="flex: 0.8; background: #1e293b; border-radius: 18px; border: 1px solid #334155; overflow: hidden; display: flex; flex-direction: column;">
                    <div style="padding: 15px 20px; background: #1e293b; border-bottom: 1px solid #334155; color: #94a3b8; font-size: 11px; font-weight: 700; text-transform: uppercase;">
                        Strategy Density
                    </div>
                    <table style="width: 100%; border-collapse: collapse; background: #1e293b;">
                        <thead>
                            <tr style="background: #0f172a; color: #64748b; font-size: 10px; text-transform: uppercase;">
                                <th style="padding: 10px 15px; text-align: left;">Available Views</th>
                                <th style="padding: 10px 15px; text-align: left;">Group Count</th>
                            </tr>
                        </thead>
                        <tbody>
                            {density_rows}
                        </tbody>
                    </table>
                    <div style="padding: 20px; color: #64748b; font-size: 12px; font-style: italic; line-height: 1.4; border-top: 1px solid #334155; margin-top: auto; background: #1e293b;">
                        Analyzes how many unique, valid prompt strategies were identified for each data group.
                    </div>
                </div>
            </div>
        </div>
        """
        display(HTML(dashboard_html))

    def debug_strategy_samples(self, target_strategy, n=5):
        """
        Displays strategy definition and samples inside a single, unified dark-mode dashboard.
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

        # 2. Filtering and Sampling
        eligible_samples = [s for s in self.updated_data if target_strategy in s.get('Strategy Views', {})]
        if not eligible_samples:
            display(HTML(
                f"<div style='color: #f87171; padding: 20px;'>❌ No samples found for strategy: {target_strategy}</div>"))
            return

        num_to_show = min(n, len(eligible_samples))
        samples = random.sample(eligible_samples, num_to_show)

        # 3. Build Unified Dashboard Header
        unacceptable_bullets = "".join([f"<li style='margin-bottom:4px;'>{u}</li>" for u in unacceptable_if])

        # Start the Master Container
        html = f"""
        <div style="width: 1150px; background-color: #0f172a; padding: 30px; border-radius: 24px; font-family: sans-serif; color: #f8fafc; border: 1px solid #1e293b; margin-left: 0;">

            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; border-bottom: 1px solid #1e293b; padding-bottom: 15px;">
                <h2 style="margin: 0; color: #38bdf8; font-size: 20px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px;">🔍 Strategy Audit: {target_strategy}</h2>
                <div style="text-align: right;">
                    <span style="background: #38bdf822; color: #38bdf8; padding: 4px 12px; border-radius: 99px; font-size: 11px; font-weight: 700; border: 1px solid #38bdf844;">
                        {len(eligible_samples)} Total Instances Found
                    </span>
                </div>
            </div>

            <div style="display: flex; gap: 20px; margin-bottom: 30px;">
                <div style="flex: 1.2; background: #1e293b; padding: 20px; border-radius: 15px; border: 1px solid #334155;">
                    <h4 style="margin: 0 0 10px 0; color: #34d399; font-size: 11px; text-transform: uppercase;">📜 Objective & Logic</h4>
                    <p style="font-size: 14px; color: #f8fafc; line-height: 1.6; margin-bottom: 15px;">{strat_desc}</p>
                    <div style="background: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #334155;">
                        <span style="color: #38bdf8; font-size: 10px; font-weight: 800; text-transform: uppercase; display: block; margin-bottom: 4px;">Requisites:</span>
                        <span style="font-size: 12px; color: #94a3b8; line-height: 1.4;">{requirements}</span>
                    </div>
                </div>

                <div style="flex: 0.8; background: #450a0a33; padding: 20px; border-radius: 15px; border: 1px solid #991b1b66;">
                    <h4 style="margin: 0 0 10px 0; color: #f87171; font-size: 11px; text-transform: uppercase;">⚠️ Unacceptable If</h4>
                    <ul style="margin: 0; padding-left: 18px; font-size: 12px; color: #fca5a5; line-height: 1.5;">
                        {unacceptable_bullets if unacceptable_if else "<li>No specific blockers listed.</li>"}
                    </ul>
                </div>
            </div>

            <div style="color: #94a3b8; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center;">
                <span>Audit Gallery (Sampled {num_to_show})</span>
                <span style="font-size: 10px; color: #475569;">Scroll within box to view more samples</span>
            </div>

            <div style="max-height: 600px; overflow-y: auto; padding-right: 10px; border-radius: 15px;">
        """

        # 4. Generate Sample Cards
        for i, sample in enumerate(samples):
            group_name = sample.get('Group name', 'Untitled Group')
            entity_col = sample.get('Entity column', 'N/A')
            wiki_url = sample.get('Wiki url', '#')
            view_data = sample['Strategy Views'][target_strategy]

            # Convert DataFrame to Styled HTML
            df_html = pd.DataFrame(view_data).style.set_table_styles([
                {'selector': 'th',
                 'props': [('background-color', '#0f172a'), ('color', '#38bdf8'), ('text-align', 'left'),
                           ('padding', '10px'), ('font-size', '12px')]},
                {'selector': 'td', 'props': [('background-color', '#1e293b'), ('color', '#cbd5e1'), ('padding', '10px'),
                                             ('border-bottom', '1px solid #0f172a'), ('font-size', '12px')]}
            ]).hide(axis='index').to_html()

            html += f"""
            <div style="background-color: #1e293b; padding: 20px; border-radius: 18px; border: 1px solid #334155; margin-bottom: 25px;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 15px;">
                    <div>
                        <div style="color: #64748b; font-size: 10px; font-weight: 700; text-transform: uppercase;">Sample #{i + 1}</div>
                        <div style="color: #f8fafc; font-size: 16px; font-weight: 700;">{group_name}</div>
                    </div>
                    <div style="text-align: right;">
                        <a href="{wiki_url}" target="_blank" style="color: #38bdf8; font-size: 11px; text-decoration: none; font-weight: 600;">Wiki Source ↗</a>
                        <div style="margin-top: 5px; color: #94a3b8; font-size: 11px;">
                            <span style="color: #475569;">Entity Col:</span> <span style="color: #38bdf8;">{entity_col}</span>
                        </div>
                    </div>
                </div>
                <div style="border: 1px solid #0f172a; border-radius: 10px; overflow: hidden; background: #1e293b;">
                    {df_html}
                </div>
            </div>
            """

        # Close the master containers
        html += "</div></div>"

        display(HTML(html))


class QuestionGenerator:
    def __init__(self, input_tables, strategy_definitions):
        self.input_tables = input_tables
        self.definitions = strategy_definitions.get("strategies", strategy_definitions)
        self.model = OpenRouterAPI(
            QUESTION_PROMPT,
            model='google/gemini-3-flash-preview',
            output_format='string',
            temperature=0.2,
            max_concurrent_requests=100,
            max_rpm=500
        )

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

        good_shots = ""
        for i, ex in enumerate(strat_def.get('good_examples', []), 1):
            good_shots += f"GOOD EXAMPLE {i}:\nInput Group: {ex.get('input_group', 'N/A')}\nOutput: {ex.get('output', 'N/A')}\n\n"

        bad_shots = ""
        for i, ex in enumerate(strat_def.get('bad_examples', []), 1):
            bad_shots += f"BAD EXAMPLE {i}:\nInput Group: {ex.get('input_group', 'N/A')}\nOutput: {ex.get('output', 'N/A')}\n\n"

        unacceptable = strat_def.get("unacceptable_if") or []
        unacceptable_text = "\n".join(f"- {a}" for a in unacceptable) or "N/A"

        return f"""
### STRATEGY TO APPLY: {strategy_name}
DEFINITION: {strat_def.get('definition', 'N/A')}
HARD REQUIREMENTS: {strat_def.get('requirements', 'N/A')}
UNACCEPTABLE IF:
{unacceptable_text}

### FEW-SHOT DEMONSTRATIONS
{good_shots if good_shots else "No good examples provided."}
{bad_shots if bad_shots else "No bad examples provided."}

### CURRENT TASK
DATA TO ANALYZE:
Group Name: {table.get('Group name')}
Table Data: {table.get('Strategy Views')[strategy_name]}

**FINAL EVALUATION**: Apply the strategy. If the data meets all "HARD REQUIREMENTS" and avoids all "UNACCEPTABLE IF" conditions, generate the JSON.
"""

    async def generate_questions(self):
        # table_idx -> how many strategies we have tried for it
        remaining_tables = {i: 0 for i in range(len(self.input_tables))}
        round_num = 1

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

            raw_responses = await self.model.async_run(batch_prompts, desc=f"Round {round_num} - Processing {len(active_batch_indices)} tables to questions")

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
                        "wiki_link": table.get('Wiki url')
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
            "Wiki link": table.get('Wiki url'),
            "Relevant table data": relevant_data
        })

    @staticmethod
    def _parse_json(text):
        try:
            if "{" in text:
                return json.loads(text[text.find("{"):text.rfind("}") + 1])
            return None
        except (KeyError, IndexError, JSONDecodeError, TypeError):
            return None

    def _print_summary(self):
        # 1. Logic & Calculations
        initial = self.stats['initial_tables']
        success = self.stats['final_success']
        exhausted = self.stats['failed_all_strategies']
        success_rate = (success / initial * 100) if initial > 0 else 0

        all_strategies = [
            "AGGREGATION", "EXTREME_NUMERICAL", "TOP_K_RANKING", "COMPARATIVE_DIFFERENCE",
            "FILTER_COUNT_NUMERICAL", "FILTER_COUNT_TEMPORAL", "UNIQUE_DIVERSITY",
            "FREQUENCY_MODE", "CATEGORICAL_FILTER_LOOKUP", "EXTREME_TEMPORAL", "COUNT_TOTAL"
        ]

        final_stats = Counter([sample['Selected strategy'] for sample in self.valid_results])
        sorted_strategies = sorted(all_strategies, key=lambda sx: final_stats.get(sx, 0), reverse=True)

        # 2. Metric Cards
        stats_html = f"""
        <div style="display: flex; gap: 12px; margin-bottom: 25px;">
            <div style="flex: 1; background: #1e293b; padding: 15px; border-radius: 12px; border: 1px solid #334155; text-align: center;">
                <div style="color: #94a3b8; font-size: 10px; font-weight: 700; text-transform: uppercase;">Initial Tables</div>
                <div style="font-size: 24px; font-weight: 800; color: #f8fafc;">{initial}</div>
            </div>
            <div style="flex: 1; background: #064e3b; padding: 15px; border-radius: 12px; border: 1px solid #059669; text-align: center;">
                <div style="color: #34d399; font-size: 10px; font-weight: 700; text-transform: uppercase;">Generated</div>
                <div style="font-size: 24px; font-weight: 800; color: #34d399;">{success}</div>
            </div>
            <div style="flex: 1; background: #450a0a33; padding: 15px; border-radius: 12px; border: 1px solid #991b1b66; text-align: center;">
                <div style="color: #fca5a5; font-size: 10px; font-weight: 700; text-transform: uppercase;">Exhausted</div>
                <div style="font-size: 24px; font-weight: 800; color: #f87171;">{exhausted}</div>
            </div>
            <div style="flex: 1; background: #1e293b; padding: 15px; border-radius: 12px; border: 1px solid #334155; text-align: center;">
                <div style="color: #38bdf8; font-size: 10px; font-weight: 700; text-transform: uppercase;">Success Rate</div>
                <div style="font-size: 24px; font-weight: 800; color: #38bdf8;">{success_rate:.1f}%</div>
            </div>
        </div>
        """

        # 3. Strategy Performance Table
        strategy_rows = ""
        total_tried, total_created = 0, 0
        for s in sorted_strategies:
            created = final_stats.get(s, 0)
            tried = self.attempted_strategy_counts.get(s, 0)
            rate = (100 * created / tried) if tried > 0 else 0
            total_tried += tried
            total_created += created

            strategy_rows += f"""
                <tr style="border-bottom: 1px solid #0f172a; background: #1e293b;">
                    <td style="padding: 12px 15px; color: #f8fafc; text-align: left; font-size: 11px; font-weight: 700; white-space: nowrap;">{s}</td>
                    <td style="padding: 12px 15px; color: #34d399; text-align: left; width: 60px;">{created}</td>
                    <td style="padding: 12px 15px; color: #94a3b8; text-align: left; width: 60px;">{tried}</td>
                    <td style="padding: 12px 15px; text-align: left; width: 60px;">
                        <span style="color: #38bdf8; font-weight: 700;">{rate:.0f}%</span>
                    </td>
                </tr>
                """

        # 4. Failed Logs Gallery
        failure_cards = ""
        for fail in self.failed_logs[:3]:  # Top 3 failures
            strat = fail['strategy']
            table_name = fail['table_name']
            reason = fail.get('llm_explanation', fail['reason'])
            wiki_link = fail.get('wiki_link', '#')

            # Get data view for the failure
            target_table = next((t for t in self.input_tables if t.get('Group name') == table_name), None)
            view_data = target_table['Strategy Views'].get(strat, []) if target_table else []

            df_html = ""
            if view_data:
                df_html = pd.DataFrame(view_data).style.set_table_styles([
                    {'selector': 'th',
                     'props': [('background-color', '#0f172a'), ('color', '#38bdf8'), ('font-size', '11px')]},
                    {'selector': 'td',
                     'props': [('background-color', '#1e293b'), ('color', '#cbd5e1'), ('font-size', '11px'),
                               ('border-bottom', '1px solid #0f172a')]}
                ]).hide(axis='index').to_html()

            failure_cards += f"""
            <div style="background: #1e293b; padding: 15px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span style="color: #f87171; font-weight: 800; font-size: 11px;">● FAILED: {strat}</span>
                    <a href="{wiki_link}" target="_blank" style="color: #38bdf8; font-size: 11px; text-decoration: none;">Source ↗</a>
                </div>
                <div style="color: #f8fafc; font-weight: 700; font-size: 14px; margin-bottom: 5px;">{table_name}</div>
                <div style="color: #94a3b8; font-size: 12px; font-style: italic; background: #0f172a44; padding: 8px; border-radius: 6px; border-left: 3px solid #f87171; margin-bottom: 10px;">
                    {reason}
                </div>
                <div style="border-radius: 6px; overflow: hidden; border: 1px solid #0f172a;">{df_html}</div>
            </div>
            """

        # 5. Assembly (Updated Flex Ratios)
        dashboard_html = f"""
        <div style="width: 1150px; background-color: #0f172a; padding: 30px; border-radius: 24px; font-family: sans-serif; color: #f8fafc; border: 1px solid #1e293b; margin-left: 0; text-align: left; box-sizing: border-box;">
            <h3 style="margin: 0 0 25px 0; font-size: 18px; color: #38bdf8; text-align: center; text-transform: uppercase; font-weight: 800; letter-spacing: 1px;">📊 Question Generation Summary</h3>
            {stats_html}

            <div style="display: flex; gap: 20px; align-items: stretch;">
                <div style="flex: 1.6; background: #1e293b; border-radius: 18px; border: 1px solid #334155; overflow: hidden; display: flex; flex-direction: column;">
                    <div style="padding: 15px 20px; background: #1e293b; border-bottom: 1px solid #334155; color: #94a3b8; font-size: 11px; font-weight: 700; text-transform: uppercase;">Strategy Success Rates</div>
                    <div style="max-height: 550px; overflow-y: auto;">
                        <table style="width: 100%; border-collapse: collapse; table-layout: auto;">
                            <thead>
                                <tr style="background: #0f172a; color: #64748b; font-size: 10px; text-transform: uppercase;">
                                    <th style="padding: 12px 15px; text-align: left;">Strategy Name</th>
                                    <th style="padding: 12px 15px; text-align: left; width: 60px;">Success</th>
                                    <th style="padding: 12px 15px; text-align: left; width: 60px;">Tried</th>
                                    <th style="padding: 12px 15px; text-align: left; width: 60px;">Rate</th>
                                </tr>
                            </thead>
                            <tbody>{strategy_rows}</tbody>
                        </table>
                    </div>
                </div>

                <div style="flex: 1.4; background: #0f172a; border-radius: 18px; border: 1px solid #1e293b; display: flex; flex-direction: column; padding: 15px;">
                    <div style="color: #94a3b8; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 15px;">🔍 Failure Deep-Dive</div>
                    <div style="max-height: 550px; overflow-y: auto; padding-right: 5px;">
                        {failure_cards}
                    </div>
                </div>
            </div>
        </div>
        """
        display(HTML(dashboard_html))


class QuestionFilter:
    def __init__(self, input_data):
        self.input_data = input_data

        # Internal state
        self.approved_data = []  # Passed everything (Schema + Dedup)
        self.rejected_data = []  # All failures

        # Intermediate state for the pipeline
        self._phase1_passed = []

        # Formalized Definitions (Unified)
        self.fail_definitions = {
            "MISSING_QUESTION": "Question field was empty or null.",
            "MISSING_TABLE": "No table data was found for this sample.",
            "EXTRA_COLUMNS": "Table contains more than one attribute column (schema violation).",
            "MISSING_COLUMNS": "Table lacks an attribute column for this strategy.",
            "EMPTY_DATA": "Attribute column exists but has no data rows.",
            "DUPLICATE_DATA": "Question is too similar to another question in the dataset.",
            "FORBIDDEN_KEYWORD": "A forbidden keyword was found in the question text.",
        }

    def run_pipeline(self):
        """Executes the full filtering pipeline sequentially."""
        print(f"Starting post-processing with {len(self.input_data)} samples...")

        self._manual_filter()
        self._deduplicate_filter()

        save_json("question_dataset", self.approved_data)

        self._display_summary()

    def _manual_filter(self):
        """Phase 1: Structure and Schema checks."""
        forb_keywords = ["these", "the following", "as shown", "below", "above", "the above", "shown above",
                         "shown below", "this table", "in this table", "this list", "in this list", "listed below",
                         "listed above", "from the table", "from the list", "given", "notable", "miscellaneous",
                         "such as", "this group", "this collection",
                         ]
        for sample in self.input_data:
            strat = sample.get('Selected strategy', 'UNKNOWN')
            reason = None

            q = str(sample.get('Question', '')).strip().lower()
            table = sample.get('Relevant table data', [])

            # --- Basic structure checks ---
            if q in ['null', 'null.', 'none', '']:
                reason = "MISSING_QUESTION"
            elif not table:
                reason = "MISSING_TABLE"
            else:
                cols = list(table[0].keys())
                target_cols = cols[1:]  # Skip first (entity) col

                if len(target_cols) > 1:
                    reason = "EXTRA_COLUMNS"
                elif len(target_cols) < 1 and strat != "COUNT_TOTAL":
                    reason = "MISSING_COLUMNS"
                elif any(all(row.get(col) is None for row in table) for col in target_cols):
                    reason = "EMPTY_DATA"

            # --- Forbidden keyword check ---
            if not reason:
                for kw in forb_keywords:
                    if kw in q:
                        reason = f"FORBIDDEN_KEYWORD"
                        break  # stop at first matching keyword

            if reason:
                # Format to match the "tuple" style of the model results for unified stats
                sample['raw_rules'] = {reason: [False, self.fail_definitions[reason]]}
                sample['rejection_reasons'] = [f"{reason}: {self.fail_definitions[reason]}"]
                sample['source'] = 'MANUAL'
                self.rejected_data.append(sample)
            else:
                self._phase1_passed.append(sample)

    def _deduplicate_filter(self, similarity_threshold=85):
        """Phase 2: Cluster similar questions and keep the one with the rarest strategy."""
        if not self._phase1_passed:
            return

        # 1. Calculate strategy frequencies to determine "rarity"
        # We calculate this based on Phase 1 survivors to get an accurate reflection of valid data
        strategy_counts = Counter(s.get('Selected strategy', 'UNKNOWN') for s in self._phase1_passed)

        clusters = []

        # 2. Greedy Clustering using rapidfuzz
        for sample in self._phase1_passed:
            placed = False
            q_text = str(sample.get('Question', '')).lower()

            for cluster in clusters:
                # Compare against the first item in the cluster (the cluster 'centroid')
                cluster_q = str(cluster[0].get('Question', '')).lower()

                # fuzz.ratio is fast and excellent for overall structural similarity
                score = fuzz.ratio(q_text, cluster_q)

                if score >= similarity_threshold:
                    cluster.append(sample)
                    placed = True
                    break

            if not placed:
                clusters.append([sample])

        # 3. Resolve Clusters (Pick rarest, reject the rest)
        for cluster in clusters:
            if len(cluster) == 1:
                # No duplicates for this question
                self.approved_data.append(cluster[0])
            else:
                # Sort the cluster by the frequency of their strategy (Ascending = rarest first)
                cluster.sort(key=lambda x: strategy_counts[x.get('Selected strategy', 'UNKNOWN')])

                # Keep the rarest one
                kept_sample = cluster[0]
                self.approved_data.append(kept_sample)

                # Reject the rest
                for rejected_sample in cluster[1:]:
                    reason = "DUPLICATE_DATA"
                    rejected_sample['raw_rules'] = {reason: [False, self.fail_definitions[reason]]}
                    rejected_sample['rejection_reasons'] = [f"{reason}: {self.fail_definitions[reason]}"]
                    rejected_sample['source'] = 'DEDUP'

                    # Optional: Attach what it was a duplicate of for your own debugging
                    rejected_sample['duplicate_of'] = kept_sample.get('Question', '')

                    self.rejected_data.append(rejected_sample)

    def _display_summary(self):
        """HTML Dashboard for QuestionFilter with extended duplicate samples."""
        from IPython.display import display, HTML
        from collections import Counter

        # 1. Calculations
        initial = len(self.input_data)
        approved = len(self.approved_data)
        rejected = len(self.rejected_data)
        pass_rate = (approved / initial * 100) if initial > 0 else 0

        # 2. Count Rejections and Collect Multi-Exhibits
        reason_counts = Counter()
        exhibits = {}  # Standard rules -> 1 sample
        duplicate_exhibits = []  # Special list for 5 duplicate samples

        for sample in self.rejected_data:
            reasons = sample.get('rejection_reasons', [])
            if not reasons: continue
            rule = reasons[0].split(':')[0].strip()
            reason_counts[rule] += 1

            if rule == "DUPLICATE_DATA":
                if len(duplicate_exhibits) < 5:
                    duplicate_exhibits.append(sample)
            elif rule not in exhibits:
                exhibits[rule] = sample

        # 3. Metric Cards (Slate & Emerald Palette)
        stats_html = f"""
        <div style="display: flex; gap: 12px; margin-bottom: 25px;">
            <div style="flex: 1; background: #1e293b; padding: 15px; border-radius: 12px; border: 1px solid #334155; text-align: center;">
                <div style="color: #94a3b8; font-size: 10px; font-weight: 700; text-transform: uppercase;">Total Input</div>
                <div style="font-size: 24px; font-weight: 800; color: #f8fafc;">{initial}</div>
            </div>
            <div style="flex: 1; background: #064e3b; padding: 15px; border-radius: 12px; border: 1px solid #059669; text-align: center;">
                <div style="color: #34d399; font-size: 10px; font-weight: 700; text-transform: uppercase;">Approved</div>
                <div style="font-size: 24px; font-weight: 800; color: #34d399;">{approved}</div>
            </div>
            <div style="flex: 1; background: #450a0a33; padding: 15px; border-radius: 12px; border: 1px solid #991b1b66; text-align: center;">
                <div style="color: #fca5a5; font-size: 10px; font-weight: 700; text-transform: uppercase;">Rejected</div>
                <div style="font-size: 24px; font-weight: 800; color: #f87171;">{rejected}</div>
            </div>
            <div style="flex: 1; background: #1e293b; padding: 15px; border-radius: 12px; border: 1px solid #334155; text-align: center;">
                <div style="color: #38bdf8; font-size: 10px; font-weight: 700; text-transform: uppercase;">Pass Rate</div>
                <div style="font-size: 24px; font-weight: 800; color: #38bdf8;">{pass_rate:.1f}%</div>
            </div>
        </div>
        """

        # 4. Strategy/Rule Inventory
        rejection_rows = ""
        all_rules = sorted(self.fail_definitions.keys(), key=lambda x: reason_counts.get(x, 0), reverse=True)
        for rule in all_rules:
            count = reason_counts.get(rule, 0)
            desc = self.fail_definitions.get(rule, "Logic/Language Check")
            rejection_rows += f"""
            <tr style="border-bottom: 1px solid #0f172a; background: #1e293b;">
                <td style="padding: 12px 15px; color: #f8fafc; font-size: 11px; font-weight: 700;">{rule}</td>
                <td style="padding: 12px 15px; color: #94a3b8; font-size: 11px;">{desc}</td>
                <td style="padding: 12px 15px; color: #f87171; text-align: right; font-weight: 800;">{count}</td>
            </tr>
            """

        # 5. Build Gallery Cards
        failure_cards = ""

        # Add the 5 Duplicate Samples First
        for i, sample in enumerate(duplicate_exhibits):
            q = sample.get('Question', 'N/A')
            match_q = sample.get('duplicate_of', 'Unknown Original')
            strat = sample.get('Selected strategy', 'N/A')
            failure_cards += f"""
            <div style="background: #1e293b; padding: 12px; border-radius: 12px; border: 1px solid #38bdf833; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #38bdf8; font-weight: 800; font-size: 10px;">● DUPLICATE CLASH {i + 1}/5</span>
                    <span style="color: #64748b; font-size: 9px;">{strat}</span>
                </div>
                <div style="color: #fca5a5; font-size: 11px; margin-bottom: 4px;"><b>REJECTED:</b> "{q}"</div>
                <div style="color: #34d399; font-size: 11px; padding-left: 8px; border-left: 2px solid #059669;"><b>KEPT:</b> "{match_q}"</div>
            </div>
            """

        # Add 1 sample for every other rejection type
        for rule, sample in exhibits.items():
            q = sample.get('Question', 'N/A')
            reason = sample.get('rejection_reasons', ['Unknown'])[0]
            failure_cards += f"""
            <div style="background: #1e293b; padding: 12px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 12px;">
                <div style="color: #f87171; font-weight: 800; font-size: 10px; margin-bottom: 4px;">● {rule}</div>
                <div style="color: #f8fafc; font-size: 12px; margin-bottom: 4px;">"{q}"</div>
                <div style="color: #94a3b8; font-size: 10px; font-style: italic;">{reason}</div>
            </div>
            """

        # 6. Final Dashboard Assembly
        dashboard_html = f"""
        <div style="width: 1100px; background-color: #0f172a; padding: 25px; border-radius: 20px; font-family: sans-serif; color: #f8fafc; border: 1px solid #1e293b; box-sizing: border-box;">
            <h3 style="margin: 0 0 20px 0; font-size: 16px; color: #38bdf8; text-align: center; text-transform: uppercase; font-weight: 800; letter-spacing: 2px;">🛡️ Data Quality Filter Pipeline</h3>
            {stats_html}
            <div style="display: flex; gap: 20px; align-items: flex-start;">
                <div style="flex: 1.4; background: #1e293b; border-radius: 15px; border: 1px solid #334155; overflow: hidden;">
                    <div style="padding: 12px 15px; border-bottom: 1px solid #334155; color: #94a3b8; font-size: 10px; font-weight: 700; text-transform: uppercase;">Rejection Audit Table</div>
                    <table style="width: 100%; border-collapse: collapse;">
                        <thead><tr style="background: #0f172a; color: #64748b; font-size: 9px; text-transform: uppercase;">
                            <th style="padding: 10px 15px; text-align: left;">Rule</th>
                            <th style="padding: 10px 15px; text-align: left;">Definition</th>
                            <th style="padding: 10px 15px; text-align: right;">Count</th>
                        </tr></thead>
                        <tbody>{rejection_rows}</tbody>
                    </table>
                </div>
                <div style="flex: 1; display: flex; flex-direction: column;">
                    <div style="color: #94a3b8; font-size: 10px; font-weight: 700; text-transform: uppercase; margin-bottom: 12px;">🔍 Deep-Dive: Duplicates & Errors</div>
                    <div style="max-height: 500px; overflow-y: auto; padding-right: 8px;">{failure_cards}</div>
                </div>
            </div>
        </div>
        """
        display(HTML(dashboard_html))