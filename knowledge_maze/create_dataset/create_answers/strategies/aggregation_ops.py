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

    # 2. Counter Answer 1 (Math-only nudge)
    counter_1 = math_ops.random_scale(true_ans)
    if counter_1 == true_ans: counter_1 += 1

    # 3. Counter Answer 2 (Evidence-backed)
    # Drop rows to change the count physically
    lied_df = table_ops.drop_random_rows(df, max_drop=3)
    counter_2 = len(lied_df)

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

    # 3. COUNTER ANSWER 1 (Math-only nudge)
    # Nudge by a random multiple of the resolution to keep the "vibe" consistent
    nudge = res * random.choice([-3, -2, -1, 1, 2, 3])
    counter_1 = math_ops.round_to_resolution(true_ans + nudge, ans_res)

    # Safety check: if nudge didn't change the value (e.g. true_ans was 0)
    if counter_1 == true_ans:
        counter_1 = math_ops.round_to_resolution(true_ans * 1.2, ans_res)

    # 4. COUNTER ANSWER 2 (The Structural/Evidence Lie)
    lied_df = df.copy()

    # Global Memory Wipe:
    # Scale the whole column's distribution to invalidate parametric memory.
    global_shift = random.uniform(0.7, 1.3)

    def mutate_val(val):
        try:
            return math_ops.round_to_resolution(float(val) * global_shift, res)
        except (ValueError, TypeError):
            return val

    lied_df[target_col] = lied_df[target_col].apply(mutate_val)

    # Recalculate on the mutated data
    mutated_series = pd.to_numeric(lied_df[target_col], errors='coerce').dropna()
    raw_counter_2 = mutated_series.mean() if is_avg else mutated_series.sum()
    counter_2 = math_ops.round_to_resolution(raw_counter_2, ans_res)

    # Final uniqueness check across all answers
    if counter_2 in [true_ans, counter_1]:
        counter_2 = math_ops.round_to_resolution(counter_2 + (res * 5), ans_res)

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
