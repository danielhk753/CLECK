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


def run_aggregation(sample):
    """
    Executes the AGGREGATION strategy for Sum or Average operations on numerical columns.

    Automatically detects column precision (decimal depth) to ensure mathematical
    consistency. Generates a true sum/average, a scaled numerical distractor,
    and a 'lied' evidence package where every value in the target column is
    individually scaled.

    Args:
        sample (dict): A dictionary containing 'Relevant table data' and the
                       'Question' text to determine the aggregation type (sum vs. avg).

    Returns:
        dict: Contains 'true_answer', 'counter_answer_1', 'counter_answer_2',
              and 'new_evidence' (the table with scaled numerical values).
    """
    df = pd.DataFrame(sample.get('Relevant table data', []))
    if df.empty: return None

    target_col = list(df.columns)[1] if len(df.columns) > 1 else df.columns[0]
    is_avg = "average" in sample.get('Question', '').lower()

    # --- NEW: Detect Column Precision ---
    # Find the maximum number of decimal places in the column
    # We use this to ensure the SUM or AVG doesn't lose data
    col_precisions = [math_ops.get_precision(x) for x in df[target_col]]
    max_p = max(col_precisions)

    # 1. True Answer
    raw_true = df[target_col].mean() if is_avg else df[target_col].sum()

    # Use the max precision of the column (or 2 for averages)
    true_ans = round(float(raw_true), max_p if not is_avg else max(max_p, 2))
    # Convert to int ONLY if it's truly a whole number
    if true_ans.is_integer():
        true_ans = int(true_ans)

    # 2. Counter Answer 1 (Math-only nudge)
    counter_1 = math_ops.random_scale(true_ans)

    # 3. Counter Answer 2 (Evidence-backed)
    lied_df = table_ops.scale_column_values(df, target_col)

    raw_counter_2 = lied_df[target_col].mean() if is_avg else lied_df[target_col].sum()

    # Match the precision of the true_ans
    counter_2 = round(float(raw_counter_2), max_p if not is_avg else max(max_p, 2))
    if counter_2.is_integer():
        counter_2 = int(counter_2)

    return {
        "true_answer": true_ans,
        "counter_answer_1": counter_1,
        "counter_answer_2": counter_2,
        "new_evidence": lied_df.to_dict(orient='records')
    }
