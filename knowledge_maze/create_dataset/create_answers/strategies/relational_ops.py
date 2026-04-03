"""
Comparative, Diversity, and Frequency Strategies Module
------------------------------------------------------
This module provides strategies for constructing QA pairs from tabular data
by computing ground-truth answers and generating adversarial counter-answers.

It includes methods for:
- Comparative Difference: identifying two entities in a table, computing their
  absolute difference, and producing distractors via arithmetic traps and
  table perturbations.
- Unique Diversity: counting unique values in a column and generating
  counter-answers through alternative counts and value reassignment.
- Frequency Mode: identifying the most frequent value in a column and
  constructing distractors by manipulating value distributions to alter the mode.

Each strategy also produces a modified ("lied") version of the original table
by applying controlled transformations that invalidate the original reasoning
path while preserving plausibility.
"""

import re
import random
import numpy as np
import pandas as pd

from knowledge_maze.create_dataset.create_answers.utils import math_ops

# ==========================================
# COMPARATIVE_DIFFERENCE:
# ==========================================
def _extract_target_entities(question, entities):
    """
    Finds the two target entities by first checking for exact substring matches,
    and falling back to word-overlap depletion for non-standard orderings.
    """
    stop_words = {
        'the', 'a', 'an', 'and', 'of', 'in', 'to', 'for', 'with', 'on', 'at',
        'by', 'from', 'is', 'exceed', 'difference', 'between', 'that', 'proposed', 'currently'
    }

    results = []
    q_lower = str(question).lower()

    # Sort by length descending to match full names before shorter subsets
    # e.g., prioritizing "Orchid Turf View A" before "Orchid"
    sorted_ents = sorted(list(set(entities)), key=lambda x: len(str(x)), reverse=True)

    # --- PHASE 1: Exact Containment Check ---
    for ent in sorted_ents:
        cleaned_ent = str(ent).strip('"\'*^ ')
        cleaned_ent = re.sub(r'\s+\d+$', '', cleaned_ent)
        ent_str = cleaned_ent.lower()

        if ent_str and ent_str in q_lower:
            results.append(ent)
            # Remove it from the question so it doesn't trigger again or steal words
            q_lower = q_lower.replace(ent_str, " ")
            if len(results) == 2:
                return results

    # --- PHASE 2: Token Depletion Fallback ---
    # Extract remaining meaningful words from the mutated question string
    q_words = set(re.findall(r'\w+', q_lower)) - stop_words

    while len(results) < 2:
        best_score = 0
        best_ent = None
        best_overlap = set()

        for ent in entities:
            if ent in results:
                continue

            cleaned_ent = str(ent).strip('"\'*^ ')
            cleaned_ent = re.sub(r'\s+\d+$', '', cleaned_ent)
            ent_words = set(re.findall(r'\w+', cleaned_ent.lower())) - stop_words

            if not ent_words:
                continue

            overlap = ent_words.intersection(q_words)
            score = len(overlap)

            if score > best_score:
                best_score = score
                best_ent = ent
                best_overlap = overlap
            elif score == best_score and score > 0:
                if best_ent is None or len(str(ent)) < len(str(best_ent)):
                    best_ent = ent
                    best_overlap = overlap

        if best_ent is not None and best_score > 0:
            results.append(best_ent)
            q_words = q_words - best_overlap
        else:
            # If we run out of valid matches, stop
            break

    return results if len(results) == 2 else []


def run_comparative_difference(sample):
    df = pd.DataFrame(sample.get('Relevant table data', []))
    if df.empty or len(df.columns) < 2: return None

    entity_col = list(df.columns)[0]
    target_col = list(df.columns)[1]
    question = sample.get('Question', '')

    all_entities = df[entity_col].tolist()
    targets = _extract_target_entities(question, all_entities)
    if len(targets) < 2:
        return None

    ent_A, ent_B = targets[0], targets[1]

    try:
        val_A = float(df.loc[df[entity_col] == ent_A, target_col].values[0])
        val_B = float(df.loc[df[entity_col] == ent_B, target_col].values[0])
    except (ValueError, IndexError, TypeError):
        return None

    # Determine table resolution (e.g., are we dealing with 1000s or 0.1s?)
    resolution = math_ops.get_column_resolution(df[target_col])

    # --- TRUE ANSWER ---
    true_diff = abs(val_A - val_B)
    true_ans = math_ops.round_to_resolution(true_diff, resolution)

    # --- COUNTER ANSWER 2 (The Structural Lie) & MEMORY WIPE ---
    lied_df = df.copy()
    valid_series = pd.to_numeric(df[target_col], errors='coerce').dropna()

    # Step A: Complete Memory Wipe using the native distribution bounds
    def mutate_row(val):
        try:
            return math_ops.generate_from_distribution(valid_series, resolution, scale_factor=1.0)
        except (ValueError, TypeError):
            return val

    lied_df[target_col] = lied_df[target_col].apply(mutate_row)

    # Step B: Dynamically extract the new Fake Difference
    new_val_A = float(lied_df.loc[lied_df[entity_col] == ent_A, target_col].values[0])
    new_val_B = float(lied_df.loc[lied_df[entity_col] == ent_B, target_col].values[0])

    counter_2 = math_ops.round_to_resolution(abs(new_val_A - new_val_B), resolution)

    # Step C: Safety Reroll
    # If the random generation accidentally hit the true answer or 0, just reroll Entity A
    while counter_2 == true_ans or counter_2 == 0:
        new_val_A = math_ops.generate_from_distribution(valid_series, resolution, scale_factor=1.0)
        lied_df.loc[lied_df[entity_col] == ent_A, target_col] = new_val_A
        counter_2 = math_ops.round_to_resolution(abs(new_val_A - new_val_B), resolution)

    # --- COUNTER ANSWER 1 (Distractor: Alternative pair difference) ---
    ans_diff = np.abs(counter_2 - true_ans)
    offset_range = np.arange(int(ans_diff / 5), int(ans_diff / 2))
    if len(offset_range) == 0:
        offset_range = list(range(int(ans_diff + 4)))
    counter_offset = random.choice([-1,1]) * random.choice(offset_range)
    counter_1 = counter_2 + counter_offset

    # Ensure it's unique
    while counter_1 in [true_ans, counter_2] or counter_1 == 0:
        counter_1 += resolution

    # Formatting helper to remove '.0' for integers
    def format_ans(val):
        return str(int(val)) if resolution >= 1 and float(val).is_integer() else str(val)

    return {
        "true_answer": format_ans(true_ans),
        "counter_answer_1": format_ans(counter_1),
        "counter_answer_2": format_ans(counter_2),
        "new_evidence": lied_df.to_dict(orient='records')
    }

# ==========================================
# UNIQUE_DIVERSITY:
# ==========================================
def run_unique_diversity(sample):
    df = pd.DataFrame(sample.get('Relevant table data', []))
    if df.empty: return None

    target_col = list(df.columns)[1] if len(df.columns) > 1 else df.columns[0]

    # We treat all non-null values as valid for this count
    valid_df = df.dropna(subset=[target_col])
    if valid_df.empty: return None

    valid_vals = valid_df[target_col].tolist()
    unique_vals = list(set(valid_vals))

    # --- TRUE ANSWER ---
    true_ans = len(unique_vals)
    total_valid = len(valid_vals)

    # We need at least 3 original uniques to delete 1 or more safely
    # and still maintain variety (since reducing 2 to 1 is boring).
    if true_ans < 3:
        return None

    # --- COUNTER ANSWER 2 (The Structural Lie) ---
    # Delete 1 or more unique values (must be strictly less than true_ans).
    # We start at 2 to avoid returning '1' unique value.
    possible_c2 = list(range(2, true_ans))
    if not possible_c2:
        return None

    counter_2 = random.choice(possible_c2)
    target_c2 = counter_2

    # --- COUNTER ANSWER 1 (The Close Nudge) ---
    # Must be close to true_ans (+/- 3) and cannot exceed total valid rows (unnatural)
    possible_c1 = [
        x for x in range(max(1, true_ans - 3), min(total_valid, true_ans + 3) + 1)
        if x != true_ans and x != counter_2
    ]

    if not possible_c1:
        return None

    counter_1 = random.choice(possible_c1)

    # --- THE MANIPULATION (Perfect Memory Wipe without New Values) ---
    lied_df = df.copy()

    # We strictly subset our existing unique values to exactly target_c2 count
    forced_uniques = random.sample(unique_vals, target_c2)

    valid_indices = valid_df.index.tolist()

    # We use a greedy assignment loop with retries to guarantee:
    # 1. No row has its original value (Perfect Memory Wipe).
    # 2. Every value in forced_uniques appears at least once (Guaranteed Count).
    attempts = 0
    success = False

    while attempts < 100:
        temp_assignments = {}
        unused_values = list(forced_uniques)
        random.shuffle(valid_indices)

        round_success = True
        for idx in valid_indices:
            orig_val = df.at[idx, target_col]

            # Candidates are from forced_uniques that aren't the original value
            candidates = [v for v in forced_uniques if v != orig_val]

            # We prioritize values that haven't been used yet to guarantee target_c2 count
            priority_candidates = [v for v in unused_values if v != orig_val]

            if priority_candidates:
                chosen = random.choice(priority_candidates)
                unused_values.remove(chosen)
            elif candidates:
                chosen = random.choice(candidates)
            else:
                # No safe candidate available for this row (all equal to orig_val)
                round_success = False
                break

            temp_assignments[idx] = chosen

        if round_success and not unused_values:
            # We assigned everything and used all forced_uniques!
            for idx, val in temp_assignments.items():
                lied_df.at[idx, target_col] = val
            success = True
            break

        attempts += 1

    # If it's impossible to find a valid arrangement within 100 tries, skip the sample
    if not success:
        return None

    return {
        "true_answer": str(true_ans),
        "counter_answer_1": str(counter_1),
        "counter_answer_2": str(counter_2),
        "new_evidence": lied_df.to_dict(orient='records')
    }

# ==========================================
# FREQUENCY_MODE:
# ==========================================
def run_frequency_mode(sample):
    df = pd.DataFrame(sample.get('Relevant table data', []))
    if df.empty: return None

    target_col = list(df.columns)[1] if len(df.columns) > 1 else df.columns[0]
    valid_df = df.dropna(subset=[target_col])
    if valid_df.empty: return None

    counts = valid_df[target_col].value_counts()
    if len(counts) == 0: return None

    # --- TRUE ANSWER ---
    true_ans = str(counts.index[0])
    true_max_count = counts.iloc[0]

    # --- COUNTER ANSWER 1 (Runner-Up) ---
    counter_1 = str(counts.index[1]) if len(counts) > 1 else f"Not {true_ans}"

    # --- COUNTER ANSWER 2 (The Structural Lie) ---
    lied_df = df.copy()

    # Memory Wipe: Randomly swap the values for different ones in the column
    # to ensure the "Mode" isn't what the LLM remembers.
    unique_vals = valid_df[target_col].unique().tolist()
    shuffled_map = dict(zip(unique_vals, random.sample(unique_vals, len(unique_vals))))
    lied_df[target_col] = lied_df[target_col].map(shuffled_map).fillna(lied_df[target_col])

    # Recalculate mode after wipe to pick a NEW loser to promote
    new_counts = lied_df[target_col].value_counts()
    current_leader_count = new_counts.iloc[0]

    potential_fakes = [val for val in new_counts.index if str(val) != true_ans]
    fake_mode_val = random.choice(potential_fakes) if potential_fakes else f"New_{true_ans}"

    counter_2 = str(fake_mode_val)

    # Promotion: Force fake_mode_val to be the undisputed winner
    target_fake_count = current_leader_count + 1
    overwrite_pool = lied_df[lied_df[target_col] != fake_mode_val].index.tolist()

    num_to_overwrite = min(target_fake_count, len(overwrite_pool))
    indices_to_change = random.sample(overwrite_pool, num_to_overwrite)

    for idx in indices_to_change:
        lied_df.at[idx, target_col] = fake_mode_val

    return {
        "true_answer": true_ans,
        "counter_answer_1": counter_1,
        "counter_answer_2": counter_2,
        "new_evidence": lied_df.to_dict(orient='records')
    }