"""
Extreme Operations Module
-------------------------
This module provides strategies for identifying boundary values in numerical
and temporal datasets (e.g., Maximum, Minimum, Top K, Earliest/Latest).

It leverages centralized utilities to calculate ground-truth extremes,
generate scaled distractors, and produce 'lied' evidence by shuffling
column values and injecting slight nudges to change the resulting extreme.
"""

import re
import random
import dateparser
import pandas as pd
from collections import Counter
from datetime import timedelta

from knowledge_maze.create_dataset.create_answers.utils import math_ops, table_ops
from knowledge_maze.create_dataset.create_answers.utils.text_ops import TemporalTableParser


def _is_seeking_minimum(question_text):
    """Helper to detect if the question is asking for a minimum/earliest value."""
    q = question_text.lower()
    min_keywords = ['min', 'lowest', 'least', 'smallest', 'fewest', 'earliest', 'first', 'oldest']
    return any(word in q for word in min_keywords)


def run_extreme_numerical(sample):
    df = pd.DataFrame(sample.get('Relevant table data', []))
    if df.empty: return None

    entity_col = list(df.columns)[0]
    target_col = list(df.columns)[1] if len(df.columns) > 1 else df.columns[0]
    is_min = _is_seeking_minimum(sample.get('Question', ''))

    # Resolution Detection (e.g., 1000 for population, 0.1 for ratings)
    res = math_ops.get_column_resolution(df[target_col])

    # 1. TRUE ANSWER
    true_idx = df[target_col].idxmin() if is_min else df[target_col].idxmax()
    true_ans_entity = df.at[true_idx, entity_col]

    # 2. COUNTER ANSWER 1 (Original 2nd place)
    sorted_df = df.sort_values(by=target_col, ascending=is_min).reset_index(drop=True)
    counter_1_entity = sorted_df.at[1, entity_col] if len(sorted_df) > 1 else "N/A"

    # 3. COUNTER ANSWER 2 (The Structural Lie)
    forbidden_entities = {true_ans_entity, counter_1_entity}
    available_indices = df[~df[entity_col].isin(forbidden_entities)].index
    winner_idx = random.choice(available_indices) if len(available_indices) > 0 else df.index[0]
    counter_2_entity = df.at[winner_idx, entity_col]

    # --- THE MANIPULATION (Memory Wipe + Distribution Matching) ---
    lied_df = df.copy()

    # Memory Wipe: Shift the entire column's distribution slightly
    global_shift = random.uniform(0.7, 1.3)

    def mutate(val):
        try:
            return math_ops.round_to_resolution(float(val) * global_shift, res)
        except:
            return val

    lied_df[target_col] = lied_df[target_col].apply(mutate)

    # Force the New Winner
    # We find the current extreme in the mutated column and nudge it by exactly one resolution unit
    current_extreme = lied_df[target_col].min() if is_min else lied_df[target_col].max()
    new_extreme_val = current_extreme - res if is_min else current_extreme + res
    lied_df.at[winner_idx, target_col] = new_extreme_val

    return {
        "true_answer": true_ans_entity,
        "counter_answer_1": counter_1_entity,
        "counter_answer_2": counter_2_entity,
        "new_evidence": lied_df.to_dict(orient='records')
    }


def extract_k(question, table_len):
    question_clean = question.lower()

    # 1. Mapping for word-based numbers
    word_to_num = {
        'one': 1, 'first': 1, 'two': 2, 'second': 2, 'three': 3, 'third': 3,
        'four': 4, 'fourth': 4, 'five': 5, 'fifth': 5, 'six': 6, 'sixth': 6,
        'seven': 7, 'seventh': 7, 'eight': 8, 'eighth': 8, 'nine': 9, 'ninth': 9,
        'ten': 10, 'tenth': 10
    }

    # 2. Check for WORD matches first (e.g., "three")
    # This avoids the "2018" problem entirely
    match_word = re.search(r'\b(' + '|'.join(word_to_num.keys()) + r')\b', question_clean)
    if match_word:
        return min(word_to_num[match_word.group(1)], table_len)

    # 3. Check for RANK digits (e.g., "3rd", "3rd-largest")
    match_rank_digit = re.search(r'\b(\d+)(st|nd|rd|th)\b', question_clean)
    if match_rank_digit:
        return min(int(match_rank_digit.group(1)), table_len)

    # 4. Fallback: Search for any digits BUT filter out years (4-digit numbers starting with 19 or 20)
    # This regex ignores 19XX and 20XX
    all_digits = re.findall(r'\b(\d+)\b', question_clean)
    for digit in all_digits:
        val = int(digit)
        if not (1900 <= val <= 2100):  # It's likely not a year
            return min(val, table_len)

    return min(3, table_len)  # Default fallback


def run_top_k(sample):
    df = pd.DataFrame(sample.get('Relevant table data', []))
    if df.empty: return None

    entity_col = list(df.columns)[0]
    target_col = list(df.columns)[1] if len(df.columns) > 1 else df.columns[0]
    question = sample.get('Question', '')
    is_min = _is_seeking_minimum(question)

    res = math_ops.get_column_resolution(df[target_col])

    # 1. TRUE ANSWER Calculation
    k = extract_k(question, len(df))
    sorted_df = df.sort_values(by=target_col, ascending=is_min).reset_index(drop=True)
    true_entities = sorted_df.iloc[:k][entity_col].tolist()
    true_ans = " ".join([f"{i + 1}. {ent}" for i, ent in enumerate(true_entities)])

    # 2. COUNTER 1 (Set difference check)
    all_entities = df[entity_col].tolist()
    c1_entities = random.sample(all_entities, k)
    while set(c1_entities) == set(true_entities) and len(all_entities) > k:
        c1_entities = random.sample(all_entities, k)
    counter_1 = " ".join([f"{i + 1}. {ent}" for i, ent in enumerate(c1_entities)])

    # 3. COUNTER 2 (Evidence Lie)
    lied_df = df.copy()

    # Select new winner indices (at least one must be a 'loser')
    all_idx = list(df.index)
    true_idx = set(df.sort_values(by=target_col, ascending=is_min).head(k).index)
    loser_idx = [i for i in all_idx if i not in true_idx]

    new_win_idx = random.sample(loser_idx, 1) + random.sample([i for i in all_idx if i not in loser_idx], k - 1)

    # Memory Wipe: Mutate column based on distribution
    global_scale = random.uniform(0.8, 1.2)
    lied_df[target_col] = lied_df[target_col].apply(
        lambda x: math_ops.round_to_resolution(float(x) * global_scale, res))

    # Get all values from the mutated column and sort them to assign back
    available_values = sorted(lied_df[target_col].tolist(), reverse=not is_min)

    # Assign the top values to our chosen indices
    top_values = available_values[:k]
    other_values = available_values[k:]

    other_idx = [i for i in all_idx if i not in new_win_idx]
    random.shuffle(other_values)

    for i, idx in enumerate(new_win_idx):
        lied_df.at[idx, target_col] = top_values[i]
    for i, idx in enumerate(other_idx):
        lied_df.at[idx, target_col] = other_values[i]

    # Boundary Tie-Breaking using Resolution
    lied_sorted = lied_df.sort_values(by=target_col, ascending=is_min)
    idx_k = lied_sorted.index[k - 1]

    if len(lied_sorted) > k:
        idx_k_plus_1 = lied_sorted.index[k]
        val_k = lied_df.at[idx_k, target_col]
        val_next = lied_df.at[idx_k_plus_1, target_col]

        if val_k == val_next:
            # Nudge the k-th winner to be strictly better by one resolution step
            lied_df.at[idx_k, target_col] = val_k - res if is_min else val_k + res

    # Final C2 List extraction
    final_sorted = lied_df.sort_values(by=target_col, ascending=is_min).head(k)
    c2_entities = final_sorted[entity_col].tolist()
    counter_2 = " ".join([f"{i + 1}. {ent}" for i, ent in enumerate(c2_entities)])

    return {
        "true_answer": true_ans,
        "counter_answer_1": counter_1,
        "counter_answer_2": counter_2,
        "new_evidence": lied_df.to_dict(orient='records')
    }


# TODO: make sure that dates are generated in the boundaries and distribution of the original data.
# TODO: look at why the generated tables are not in the parsed data format.
def run_extreme_temporal(sample):
    """Generates synthetic counterfactual evidence using the TemporalSpecialist parser."""
    df = pd.DataFrame(sample.get("Relevant table data", []))
    parser = TemporalTableParser()
    if df.empty:
        return None

    entity_col = list(df.columns)[0]
    target_col = (
        list(df.columns)[1] if len(df.columns) > 1 else df.columns[0]
    )
    question = sample.get("Question", "").lower()

    # Determine sorting polarity
    is_min = any(
        word in question for word in ["earliest", "first", "oldest", "min"]
    )

    # =========================================================
    # --- 1. HARNESS THE CLASS PARSER FOR GROUND TRUTH ---
    # =========================================================
    dtypes_found = []
    parsed_datetimes = {}

    for idx, row in df.iterrows():
        raw_val = row[target_col]
        # Use the class waterfall directly!
        parsed_val, dtype = parser._run_waterfall(raw_val)
        dtypes_found.append(dtype)

        # We need actual datetime objects to compute counterfactual shifts
        if parsed_val and dtype not in ["FMT_NONE", "Other"]:
            # If it's a span, extract the boundary based on is_min
            if "-" in parsed_val:
                parts = parsed_val.split("-")
                target_part = parts[0] if is_min else parts[-1]
                dt = dateparser.parse(target_part)
            else:
                dt = dateparser.parse(parsed_val)

            if dt:
                parsed_datetimes[idx] = dt

    # Quality Gate: Return N/A if we don't have enough evaluable dates
    if len(parsed_datetimes) < 2:
        return {
            "true_answer": "N/A",
            "counter_answer_1": "N/A",
            "counter_answer_2": "N/A",
            "new_evidence": [],
        }

    # Determine Majority Format (mimicking the class voting system)
    valid_votes = [d for d in dtypes_found if d not in ["FMT_NONE", "Other"]]
    mapped_votes = [
        "FMT_YEAR" if v in ["FMT_DECADE", "FMT_CENTURY"] else v
        for v in valid_votes
    ]

    if mapped_votes:
        target_format = Counter(mapped_votes).most_common(1)[0][0]
    else:
        target_format = "FMT_YEAR"

    # Map the format to a generation template
    if target_format == "FMT_FULL":
        template = "verbose"
    elif any("-" in str(df.at[idx, target_col]) for idx in df.index):
        template = "span"
    else:
        template = "year"

    # =========================================================
    # --- 2. TRUE & COUNTER 1 (Extracted from Real Data) ---
    # =========================================================
    sorted_idx = sorted(
        parsed_datetimes.keys(),
        key=lambda i: parsed_datetimes[i],
        reverse=not is_min,
    )
    true_idx = sorted_idx[0]
    true_ans = str(df.at[true_idx, entity_col])

    counter_1 = "N/A"
    for idx in sorted_idx[1:]:
        cand = str(df.at[idx, entity_col])
        if cand != true_ans:
            counter_1 = cand
            break

    # =========================================================
    # --- 3. COUNTER 2 (The Lie) ---
    # =========================================================
    potential_c2 = [
        i
        for i in df.index
        if str(df.at[i, entity_col]) not in [true_ans, counter_1]
    ]
    winner_idx = random.choice(potential_c2) if potential_c2 else df.index[0]
    counter_2 = str(df.at[winner_idx, entity_col])

    # =========================================================
    # --- 4. SYNTHETIC EVIDENCE (The "No-Tie" Generator) ---
    # =========================================================
    # Create a clean copy. The original df remains completely untouched.
    lied_df = df.copy()

    # Global Shift to break parametric memory
    year_shift = random.randint(5, 20) * random.choice([-1, 1])
    base_dt = min(parsed_datetimes.values()) + timedelta(days=year_shift * 365)

    # Step size must be > 1 year if we are only displaying years
    step_days = 400 if template in ["year", "span"] else 45

    # Generate a pool of strictly unique dates
    pool = [base_dt + timedelta(days=i * step_days) for i in range(len(df))]
    pool.sort(reverse=not is_min)  # Winner gets the extreme

    # Assign the extreme to our winner_idx
    final_dates = {winner_idx: pool.pop(0)}

    # Shuffle and assign the rest
    remaining_indices = [i for i in df.index if i != winner_idx]
    random.shuffle(remaining_indices)
    for idx in remaining_indices:
        final_dates[idx] = pool.pop(0)

    # =========================================================
    # --- 5. FORMATTING THE LIED TABLE ---
    # =========================================================
    for idx, dt in final_dates.items():
        if template == "year":
            val = dt.strftime("%Y")
        elif template == "span":
            # Ensure the span doesn't overlap the next row's start year
            val = f"{dt.year}–{dt.year + random.randint(1, 2)}"
        else:
            val = dt.strftime("%B %d, %Y")

        # Directly overwrite the targeted original column
        lied_df.at[idx, target_col] = val

    return {
        "true_answer": true_ans,
        "counter_answer_1": counter_1,
        "counter_answer_2": counter_2,
        "new_evidence": lied_df.to_dict(orient="records"),
    }
