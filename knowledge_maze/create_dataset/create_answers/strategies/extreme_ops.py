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
import copy
import random
import dateparser
import pandas as pd
from datetime import timedelta, datetime

from knowledge_maze.create_dataset.create_answers.utils import math_ops, table_ops
from knowledge_maze.create_dataset.create_answers.utils.text_ops import TemporalTableParser

# ==========================================
# EXTREME_NUMERICAL:
# ==========================================
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

# ==========================================
#TOP_K_FILTERING:
# ==========================================
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

# ==========================================
# EXTREME_TEMPORAL:
# ==========================================
def process_single_table(orig_sample, parsed_sample):
    """Processes a single table to generate counterfactuals based on parsed data."""
    df_orig = pd.DataFrame(orig_sample.get("Relevant table data", []))
    df_parsed = pd.DataFrame(parsed_sample.get("Relevant table data", []))

    if df_orig.empty or df_parsed.empty:
        return None

    entity_col = list(df_orig.columns)[0]
    target_col = list(df_orig.columns)[1] if len(df_orig.columns) > 1 else df_orig.columns[0]
    question = orig_sample.get("Question", "").lower()
    is_min = any(word in question for word in ["earliest", "first", "oldest", "min"])

    # Extract the new parsed column generated by the class
    original_cols = list(df_orig.columns)
    parsed_cols = [c for c in df_parsed.columns if c not in original_cols]
    if not parsed_cols:
        return None
    parsed_col = parsed_cols[-1]

    parsed_spans = {}  # To track full (start, end) tuples
    parsed_datetimes = {}  # The target date used for sorting
    format_type = "FMT_YEAR"
    is_span = False

    # --- Intent Detector ---
    start_keywords = ['join', 'start', 'begin', 'member', 'born', 'found', 'create', 'became']
    end_keywords = ['leave', 'end', 'death', 'die', 'until', 'finish', 'resign', 'retire']
    prefers_start = any(kw in question for kw in start_keywords)
    prefers_end = any(kw in question for kw in end_keywords)

    def parse_bound(b):
        """Helper to catch 'present' strings before hitting dateparser."""
        if any(x in b.lower() for x in ["present", "now", "current"]):
            return datetime.now()
        return dateparser.parse(b)

    # --- 1. EXTRACT AND PARSE ---
    for idx, row in df_parsed.iterrows():
        val = str(row[parsed_col])
        if pd.isna(row[parsed_col]) or val == "None" or not val.strip():
            continue

        # Format Detection
        if len(val.split("/")) == 3:
            format_type = "FMT_FULL"
        elif len(val.split("/")) == 2:
            format_type = "FMT_MONTH_YEAR"

        # Robust Span Handling: split on hyphens or en-dashes
        parts = re.split(r'\s*[\–\-]\s*', val)

        if len(parts) >= 2:
            is_span = True
            start_dt = parse_bound(parts[0])
            end_dt = parse_bound(parts[-1])

            if start_dt and end_dt:
                parsed_spans[idx] = (start_dt, end_dt)

                # Assign dates based on question intent
                if prefers_start:
                    parsed_datetimes[idx] = start_dt
                elif prefers_end:
                    parsed_datetimes[idx] = end_dt
                else:
                    parsed_datetimes[idx] = start_dt if is_min else end_dt
        else:
            dt = parse_bound(val)
            if dt:
                parsed_datetimes[idx] = dt

    if len(parsed_datetimes) < 2:
        return None

    # --- 2. TRUE & COUNTER 1 ---
    sorted_idx = sorted(parsed_datetimes.keys(), key=lambda i: parsed_datetimes[i], reverse=not is_min)
    true_idx = sorted_idx[0]
    true_ans = str(df_orig.at[true_idx, entity_col])

    counter_1 = "N/A"
    for idx in sorted_idx[1:]:
        cand = str(df_orig.at[idx, entity_col])
        if cand != true_ans:
            counter_1 = cand
            break

    # --- 3. COUNTER 2 (The Lie) ---
    potential_c2 = [i for i in df_orig.index if str(df_orig.at[i, entity_col]) not in [true_ans, counter_1]]
    if not potential_c2:
        return None

    winner_idx = random.choice(potential_c2)
    counter_2 = str(df_orig.at[winner_idx, entity_col])

    # --- 4. STRUCTURAL CHECKS ---
    is_sequential_chain = False

    if is_span and len(parsed_spans) >= 2:
        sorted_spans = sorted(parsed_spans.values(), key=lambda x: x[0])

        # Check sequential links (overlapping is fine, but we see if they chain)
        links = sum(1 for i in range(len(sorted_spans) - 1)
                    if abs((sorted_spans[i][1] - sorted_spans[i + 1][0]).days) <= 366)

        if links >= len(sorted_spans) // 2:
            is_sequential_chain = True

    # --- 5. SYNTHETIC EVIDENCE GENERATION ---
    lied_df = df_orig.copy()
    lied_df[target_col] = lied_df[target_col].astype(object)

    min_dt = min(parsed_datetimes.values())
    max_dt = max(parsed_datetimes.values())
    N = len(df_orig)

    # BRANCH A: Continuous Sequential Chains
    if is_span and is_sequential_chain:
        pool_spans = []
        curr_dt = min_dt
        for _ in range(N):
            dur = timedelta(days=random.randint(1 * 365, 5 * 365))
            pool_spans.append((curr_dt, curr_dt + dur))
            curr_dt = curr_dt + dur

        pool_spans.sort(key=lambda x: x[0], reverse=not is_min)

        lied_df.at[winner_idx, target_col] = f"{pool_spans[0][0].year}–{pool_spans[0][1].year}"
        rem_idx = [i for i in df_orig.index if i != winner_idx]
        random.shuffle(rem_idx)
        for idx in rem_idx:
            st, en = pool_spans.pop()
            lied_df.at[idx, target_col] = f"{st.year}–{en.year}"

    # BRANCH B: Non-sequential Spans -> Swap values to make winner the new extreme!
    elif is_span:
        extreme_idx = sorted_idx[0]

        # Pull the raw text representations from the original dataframe
        val_extreme = df_orig.at[extreme_idx, target_col]
        val_winner = df_orig.at[winner_idx, target_col]

        # Swap them
        lied_df.at[winner_idx, target_col] = val_extreme
        lied_df.at[extreme_idx, target_col] = val_winner

    # BRANCH C: Single Dates -> Prevent ties using a buffered boundary
    else:
        total_seconds = int((max_dt - min_dt).total_seconds())
        if total_seconds < 3600 * 24 * 365:
            max_dt = min_dt + timedelta(days=365)
            total_seconds = int((max_dt - min_dt).total_seconds())

        buffer_seconds = int(total_seconds * 0.1)  # 10% safety buffer
        rem_idx = [i for i in df_orig.index if i != winner_idx]
        random.shuffle(rem_idx)

        final_dates = {}

        if is_min:
            # Generate extremum strictly at the earliest edge
            final_dates[winner_idx] = min_dt
            # Shift the remaining range slightly later
            sub_min_dt = min_dt + timedelta(seconds=buffer_seconds)
            sub_total_seconds = int((max_dt - sub_min_dt).total_seconds())

            unique_offsets = set()
            while len(unique_offsets) < len(rem_idx):
                unique_offsets.add(random.randint(0, sub_total_seconds))

            sorted_offsets = sorted(list(unique_offsets))
            pool = [sub_min_dt + timedelta(seconds=off) for off in sorted_offsets]
        else:
            # Generate extremum strictly at the latest edge
            final_dates[winner_idx] = max_dt
            # Shift the remaining range slightly earlier
            sub_max_dt = max_dt - timedelta(seconds=buffer_seconds)
            sub_total_seconds = int((sub_max_dt - min_dt).total_seconds())

            unique_offsets = set()
            while len(unique_offsets) < len(rem_idx):
                unique_offsets.add(random.randint(0, sub_total_seconds))

            sorted_offsets = sorted(list(unique_offsets), reverse=True)
            pool = [min_dt + timedelta(seconds=off) for off in sorted_offsets]

        for idx in rem_idx:
            final_dates[idx] = pool.pop(0)

        # Write results to the lied table
        for idx, dt in final_dates.items():
            if format_type == "FMT_FULL":
                val = dt.strftime("%d/%m/%Y")
            elif format_type == "FMT_MONTH_YEAR":
                val = dt.strftime("%m/%Y")
            else:
                val = dt.strftime("%Y")
            lied_df.at[idx, target_col] = str(val)

    # Final payload formatting
    sample_out = orig_sample.copy()
    sample_out.update({
        "true_answer": true_ans,
        "counter_answer_1": counter_1,
        "counter_answer_2": counter_2,
        "new_evidence": lied_df.to_dict(orient="records"),
    })
    return sample_out


def run_extreme_temporal(samples, verbose=False):
    """
    Takes a list of table dictionaries, runs the TemporalTableParser on all of them,
    and returns a filtered list of fully processed synthetic counterfactual samples.
    """
    parser = TemporalTableParser()
    working_samples = copy.deepcopy(samples)

    # Run the batch process (assuming your parser supports mutating samples in-place)
    parser.process_and_normalize_tables(working_samples, verbose=False)

    successful_samples = []
    rejected_count = 0

    for orig_sample, parsed_sample in zip(samples, working_samples):
        result = process_single_table(orig_sample, parsed_sample)

        if result is not None:
            successful_samples.append(result)
        else:
            rejected_count += 1

    if verbose:
        print("=" * 40)
        print("TEMPORAL GENERATION SUMMARY")
        print("=" * 40)
        print(f"Total Initial Tables : {len(samples)}")
        print(f"Successfully Created : {len(successful_samples)}")
        print(f"Rejected / Deleted   : {rejected_count}")
        print("=" * 40)

    return successful_samples