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
import pandas as pd

from knowledge_maze.create_dataset.create_answers.utils import math_ops, table_ops


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

    # 1. TRUE ANSWER
    true_idx = df[target_col].idxmin() if is_min else df[target_col].idxmax()
    true_ans_entity = df.at[true_idx, entity_col]

    # 2. COUNTER ANSWER 1 (The original 2nd place)
    sorted_df = df.sort_values(by=target_col, ascending=is_min).reset_index(drop=True)
    counter_1_entity = sorted_df.at[1, entity_col] if len(sorted_df) > 1 else "N/A"

    # 3. COUNTER ANSWER 2 (The Structural Lie)
    # Pick a random entity that is NOT the true winner AND NOT the counter_1 runner-up
    forbidden_entities = {true_ans_entity, counter_1_entity}
    available_indices = df[~df[entity_col].isin(forbidden_entities)].index

    if len(available_indices) > 0:
        winner_idx = random.choice(available_indices)
    else:
        # Fallback if table is too small: just ensure it's not the true winner
        winner_idx = df[df[entity_col] != true_ans_entity].index[0]

    counter_2_entity = df.at[winner_idx, entity_col]

    # --- THE MANIPULATION ---
    # Step A: Swap the values of the True Winner and the New Winner
    # This effectively "moves" the original extreme value to the new entity
    lied_df = table_ops.swap_values(df, true_idx, winner_idx, target_col)

    # Step B: Scale the entire column so every value is technically a "lie"
    # This uses the precision-aware scaling we built earlier
    lied_df = table_ops.scale_column_values(lied_df, target_col)

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
    """
    Executes TOP_K strategy with rank-preserving swaps and space-separated
    numbered lists. Ensures unique entity sets for all three answers.
    """
    df = pd.DataFrame(sample.get('Relevant table data', []))
    if df.empty: return None

    entity_col = list(df.columns)[0]
    target_col = list(df.columns)[1] if len(df.columns) > 1 else df.columns[0]
    question = sample.get('Question', '')
    is_min = _is_seeking_minimum(question)

    # 1. Extract K and identify True Winners
    k = extract_k(question, len(df))
    sorted_df = df.sort_values(by=target_col, ascending=is_min).reset_index(drop=True)

    true_entities = sorted_df.iloc[:k][entity_col].tolist()
    true_ans = " ".join([f"{i + 1}. {ent}" for i, ent in enumerate(true_entities)])

    # --- 2. COUNTER ANSWER 1 (Randomly Sampled Set) ---
    all_entities = df[entity_col].tolist()

    # Loop to ensure the SET of entities is different from True Answer
    attempts = 0
    while attempts < 100:
        c1_entities = random.sample(all_entities, k)
        if set(c1_entities) != set(true_entities):
            break
        attempts += 1

    counter_1 = " ".join([f"{i + 1}. {ent}" for i, ent in enumerate(c1_entities)])

    # --- 3. COUNTER ANSWER 2 (The Structural/Evidence Lie) ---
    # To ensure set difference, we must force at least one "loser" into the new top K
    true_indices = set(sorted_df.index[:k])
    all_indices = list(df.index)
    loser_indices = [i for i in all_indices if i not in true_indices]

    if loser_indices and k > 0:
        # Guarantee at least one entity from outside the true winners
        num_losers = random.randint(1, min(k, len(loser_indices)))
        new_winner_indices = random.sample(loser_indices, num_losers)

        # Fill remaining spots with anyone else
        remaining_needed = k - num_losers
        remaining_pool = [i for i in all_indices if i not in new_winner_indices]
        new_winner_indices += random.sample(remaining_pool, remaining_needed)
    else:
        # Fallback for tiny tables: just shuffle indices
        new_winner_indices = random.sample(all_indices, k)

    # SWAP: Move original top values to the new winner indices
    lied_df = df.copy()
    original_top_values = sorted_df.iloc[:k][target_col].tolist()
    for idx, val in zip(new_winner_indices, original_top_values):
        lied_df.at[idx, target_col] = val

    # SCALE: Use a tighter scale to help preserve the intended ranks after swapping
    lied_df = table_ops.scale_column_values(lied_df, target_col)

    # --- Ensure no tie at the K-th boundary ---
    if len(lied_df) > k:
        # Determine the precision nudge (1 for integers, 0.01 for decimals)
        sample_val = lied_df[target_col].iloc[0]
        nudge = 1 if math_ops.get_precision(sample_val) == 0 else 0.1 ** math_ops.get_precision(sample_val)

        # Sort indices to find the boundary rows
        boundary_indices = lied_df[target_col].sort_values(ascending=is_min).index
        idx_k = boundary_indices[k - 1]
        idx_k_plus_1 = boundary_indices[k]

        # If the k-th and (k+1)-th values are identical, break the tie
        if lied_df.at[idx_k, target_col] == lied_df.at[idx_k_plus_1, target_col]:
            if is_min:
                # For 'Smallest', make the k-th value slightly smaller
                lied_df.at[idx_k, target_col] -= nudge
            else:
                # For 'Largest', make the k-th value slightly larger
                lied_df.at[idx_k, target_col] += nudge

    # D. Identify NEW top K and check uniqueness
    lied_sorted = lied_df.sort_values(by=target_col, ascending=is_min).reset_index(drop=True)
    c2_entities = lied_sorted.iloc[:k][entity_col].tolist()

    # Final uniqueness check: Ensure C2 is a different set than True and C1
    if set(c2_entities) == set(true_entities) or set(c2_entities) == set(c1_entities):
        # If random noise ruined the set difference, we force a shuffle of c2
        # to at least make the rank order different, or re-run selection.
        random.shuffle(c2_entities)

    counter_2 = " ".join([f"{i + 1}. {ent}" for i, ent in enumerate(c2_entities)])

    return {
        "true_answer": true_ans,
        "counter_answer_1": counter_1,
        "counter_answer_2": counter_2,
        "new_evidence": lied_df.to_dict(orient='records')
    }


def run_extreme_temporal(sample):
    """
    Executes the EXTREME_TEMPORAL strategy (Earliest or Latest Date).

    Converts text dates to datetime objects to find the true extreme.
    Counter 1 shifts the year/month. Counter 2 shuffles the dates and
    injects a fake extreme date into the evidence.
    """
    df = pd.DataFrame(sample.get('Relevant table data', []))
    if df.empty: return None

    target_col = list(df.columns)[1] if len(df.columns) > 1 else df.columns[0]
    is_min = _is_seeking_minimum(sample.get('Question', ''))

    # Safely parse dates using Pandas
    parsed_dates = pd.to_datetime(df[target_col], errors='coerce')

    # 1. True Answer
    true_date = parsed_dates.min() if is_min else parsed_dates.max()
    true_ans = true_date.strftime('%Y-%m-%d')  # Standardize format

    # 2. Counter Answer 1 (Math-only nudge)
    # Shift by a random number of days (e.g., 1 to 5 years)
    shift_days = random.randint(365, 1825)
    fake_date_1 = true_date - pd.Timedelta(days=shift_days) if is_min else true_date + pd.Timedelta(days=shift_days)
    counter_1 = fake_date_1.strftime('%Y-%m-%d')

    # 3. Counter Answer 2 (Evidence-backed)
    lied_df = table_ops.shuffle_column(df, target_col)

    # Generate the fake date for the evidence
    shift_days_2 = random.randint(100, 300)
    fake_date_2 = true_date - pd.Timedelta(days=shift_days_2) if is_min else true_date + pd.Timedelta(days=shift_days_2)
    counter_2 = fake_date_2.strftime('%Y-%m-%d')

    # Inject it into a random row, replacing the original string
    random_idx = random.choice(lied_df.index)
    lied_df.at[random_idx, target_col] = counter_2

    return {
        "true_answer": true_ans,
        "counter_answer_1": counter_1,
        "counter_answer_2": counter_2,
        "new_evidence": lied_df.to_dict(orient='records')
    }