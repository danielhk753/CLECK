"""
Aggregation Operations Module
-----------------------------
This module provides strategies for generating ground-truth answers and
distractors for aggregation-based questions (e.g., Sum, Average, Count).

Each strategy produces:
1. A 'True' answer derived from the original table.
2. 'Counter Answer 1': A deterministic numerical nudge (math-only).
3. 'Counter Answer 2': An answer derived from manipulated evidence (e.g., dropped rows
   or scaled values), including the modified table data to facilitate
   knowledge-conflict testing in LLMs.
"""
import random
import pandas as pd
from knowledge_maze.create_dataset.create_answers.utils import math_ops, table_ops

# ==========================================
# COUNT_TOTAL:
# ==========================================
def run_count_total(sample):
    """
    Executes the COUNT_TOTAL strategy to determine the number of entities in a table.

    Calculates the true row count, generates a scaled distractor, and creates
    a 'lied' version of the evidence by randomly dropping rows. The second
    counter answer is the count of the remaining rows in the manipulated table.

    Args:
        sample (dict): A dictionary containing 'Relevant table data' (list of dicts).

    Returns:
        dict: Contains 'true_answer', 'counter_answer_1', 'counter_answer_2',
              and 'new_evidence' (the table with dropped rows).
    """
    df = pd.DataFrame(sample.get('Relevant table data', []))
    if df.empty: return None

    # 1. True Answer
    true_ans = len(df)

    # 2. Counter Answer 2 (Evidence-backed)
    # Drop rows to change the count physically
    lied_df = table_ops.drop_random_rows(df, max_drop=3)
    counter_2 = len(lied_df)

    # 3. Counter Answer 1 (Claim-backed)
    # Define the range: from (True - 5, capped at 2) to (True + 3)
    low_bound = max(true_ans - 5, 2)
    high_bound = true_ans + 3

    # Create the pool of integers in this range
    potential_pool = list(range(low_bound, high_bound + 1))

    # Remove the two "occupied" answers (Truth and Evidence)
    valid_candidates = [c for c in potential_pool if c != true_ans and c != counter_2]

    # Separate the valid candidates into those below and above the true answer
    lower_pool = [c for c in valid_candidates if c < true_ans]
    upper_pool = [c for c in valid_candidates if c > true_ans]

    # 50/50 chance to pick from either pool to eliminate directional bias
    if lower_pool and upper_pool:
        if random.choice([True, False]):
            counter_1 = random.choice(lower_pool)
        else:
            counter_1 = random.choice(upper_pool)
    elif valid_candidates:
        # Fallback if only one side of the true_ans has valid integers
        counter_1 = random.choice(valid_candidates)
    else:
        # Emergency fallback: random jump of 2, ensuring we stay >= 2
        shift = random.choice([2, -2])
        counter_1 = max(2, true_ans + shift)

    return {
        "true_answer": true_ans,
        "counter_answer_1": counter_1,
        "counter_answer_2": counter_2,
        "new_evidence": lied_df.to_dict(orient='records')
    }

# ==========================================
# AGGREGATION:
# ==========================================
def run_aggregation(sample):
    """
    Executes AGGREGATION strategy (Sum/Average) with global memory-wipe
    and resolution-consistent rounding.
    """
    df = pd.DataFrame(sample.get('Relevant table data', []))
    if df.empty: return None

    target_col = list(df.columns)[1] if len(df.columns) > 1 else df.columns[0]
    question = sample.get('Question', '').lower()
    is_avg = any(word in question for word in ["average", "mean", "avg"])

    # 1. Detect Column Resolution (e.g., 1000 for population, 0.1 for ratings)
    res = math_ops.get_column_resolution(df[target_col])

    # For Averages, we often need one level of finer precision than the resolution
    # to be realistic (e.g., avg of [10, 21] is 15.5), but we respect the 'round numbers'
    # rule by capping it to a logical increment.
    ans_res = res if not is_avg else (0.1 if res >= 1 else res)

    # 2. TRUE ANSWER Calculation
    # We drop NaNs but keep the rest for a "Raw" calculation before rounding to resolution
    valid_series = pd.to_numeric(df[target_col], errors='coerce').dropna()
    if valid_series.empty: return None

    raw_true = valid_series.mean() if is_avg else valid_series.sum()
    true_ans = math_ops.round_to_resolution(raw_true, ans_res)

    # 3. COUNTER ANSWER 2 (The Structural/Evidence Lie)
    lied_df = df.copy()

    # Global Memory Wipe:
    # Scale the whole column's distribution to invalidate parametric memory.
    global_shift = random.uniform(0.7, 1.3)

    def mutate_val(val):
        try:
            original_float = float(val)
            # 1. Calculate the shifted and rounded value
            mutated = math_ops.round_to_resolution(original_float * global_shift, res)

            # 2. Safety check: If rounding or a shift near 1.0 resulted in no change,
            # force a nudge by exactly one resolution unit.
            if mutated == original_float:
                # Shift in the direction of the global_shift (up if > 1, down if < 1)
                change_direction = 1 if global_shift >= 1 else -1
                mutated = math_ops.round_to_resolution(mutated + (change_direction * res), res)

            return mutated
        except (ValueError, TypeError):
            # Keep non-numeric strings (like "N/A") as they are
            return val

    lied_df[target_col] = lied_df[target_col].apply(mutate_val)

    # Recalculate on the mutated data
    mutated_series = pd.to_numeric(lied_df[target_col], errors='coerce').dropna()
    raw_counter_2 = mutated_series.mean() if is_avg else mutated_series.sum()
    counter_2 = math_ops.round_to_resolution(raw_counter_2, ans_res)

    # 4. COUNTER ANSWER 1 (Scale nudge derived from Evidence/Counter 2)
    # Nudge by a random percentage (e.g., 5% to 15%) of counter_2's magnitude
    scale_shift = random.uniform(0.05, 0.15)
    direction = random.choice([-1, 1])

    raw_counter_1 = counter_2 + (direction * abs(counter_2) * scale_shift)
    counter_1 = math_ops.round_to_resolution(raw_counter_1, ans_res)

    # Safety loop: ensure counter_1 doesn't collide with true_ans or counter_2 due to rounding.
    # If counter_2 was exactly 0, fallback to shifting by the resolution amount.
    shift_step = ans_res if ans_res > 0 else 1
    while counter_1 in {true_ans, counter_2}:
        counter_1 = math_ops.round_to_resolution(counter_1 + (direction * shift_step), ans_res)

    # Final uniqueness check: ensure counter_2 didn't accidentally land on true_ans originally
    while counter_2 in {true_ans, counter_1}:
        counter_2 = math_ops.round_to_resolution(counter_2 + shift_step, ans_res)

    # Formatting helper to clean up trailing .0 and handle precision
    def format_final(val):
        # If it's effectively an integer and our resolution is >= 1, return int
        if float(val).is_integer() and ans_res >= 1:
            return str(int(val))
        # Otherwise, round to appropriate decimal places based on resolution
        precision = 0 if ans_res >= 1 else len(str(ans_res).split('.')[1])
        return f"{val:.{precision}f}"

    return {
        "true_answer": format_final(true_ans),
        "counter_answer_1": format_final(counter_1),
        "counter_answer_2": format_final(counter_2),
        "new_evidence": lied_df.to_dict(orient='records')
    }
