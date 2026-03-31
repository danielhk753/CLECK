import random
import pandas as pd

from knowledge_maze.create_dataset.create_answers.utils import math_ops, table_ops

# ==========================================
# COMPARATIVE_DIFFERENCE:
# ==========================================
def _extract_target_entities(question, entities):
    """Attempts to find which two entities from the table are being compared."""
    found = []
    q_lower = str(question).lower()
    for ent in sorted(entities, key=lambda x: len(str(x)), reverse=True):
        if str(ent).lower() in q_lower:
            found.append(ent)
            q_lower = q_lower.replace(str(ent).lower(), "")
            if len(found) == 2: break
    return found


def run_comparative_difference(sample):
    df = pd.DataFrame(sample.get('Relevant table data', []))
    if df.empty or len(df.columns) < 2: return None

    entity_col = list(df.columns)[0]
    target_col = list(df.columns)[1]
    question = sample.get('Question', '')

    all_entities = df[entity_col].tolist()
    targets = _extract_target_entities(question, all_entities)
    if len(targets) < 2:
        targets = random.sample(all_entities, 2)

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

    # --- COUNTER ANSWER 1 (The Mathematical Trap - Sum instead of Diff) ---
    c1_raw = val_A + val_B
    counter_1 = math_ops.round_to_resolution(c1_raw, resolution)
    if counter_1 == true_ans:
        counter_1 = math_ops.round_to_resolution(true_ans * random.uniform(1.5, 2.5), resolution)

    # --- COUNTER ANSWER 2 (The Structural Lie) ---
    lied_df = df.copy()

    # Step A: Generate a believable fake difference that fits the data's scale
    fake_diff_raw = math_ops.generate_from_distribution(df[target_col], resolution, scale_factor=0.5)
    counter_2 = math_ops.round_to_resolution(abs(fake_diff_raw), resolution)

    # Ensure it's unique
    while counter_2 in (true_ans, counter_1) or counter_2 == 0:
        counter_2 += resolution

    # Step B: Memory Wipe - Alter ALL numerical entries in the table
    # We apply a gentle scale factor to the whole column to invalidate memory,
    # but round it so it looks natural (e.g., 55000 -> 61000, not 61382)
    global_shift_factor = random.uniform(0.7, 1.3)

    def mutate_row(val):
        try:
            return math_ops.round_to_resolution(float(val) * global_shift_factor, resolution)
        except (ValueError, TypeError):
            return val

    lied_df[target_col] = lied_df[target_col].apply(mutate_row)

    # Step C: Hardcode Entity A and B to guarantee the new Fake Difference
    # We base B off the newly mutated distribution, then force A.
    new_val_B = math_ops.generate_from_distribution(lied_df[target_col], resolution)
    new_val_A = new_val_B + counter_2

    lied_df.loc[lied_df[entity_col] == ent_A, target_col] = new_val_A
    lied_df.loc[lied_df[entity_col] == ent_B, target_col] = new_val_B

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

    # --- COUNTER ANSWER 1 (Total Rows Trap) ---
    total_valid = len(valid_vals)
    counter_1 = total_valid if total_valid != true_ans else (true_ans + 1)

    # --- COUNTER ANSWER 2 (The Structural Lie) ---
    lied_df = df.copy()

    # Determine target unique count for the lie
    target_c2 = true_ans - 1 if true_ans > 1 else true_ans + 1
    if target_c2 == counter_1 or target_c2 == true_ans:
        target_c2 = true_ans + 2

    counter_2 = target_c2

    # Memory Wipe: If we are dealing with strings/categories, shuffle them globally
    # so the LLM doesn't recognize the specific entries.
    all_target_values = valid_df[target_col].unique().tolist()
    random.shuffle(all_target_values)

    # Selection of forced uniques for the lie
    forced_uniques = all_target_values[:target_c2]
    while len(forced_uniques) < target_c2:
        forced_uniques.append(f"Variant_{random.randint(100, 999)}")

    # Overwrite to guarantee exact unique count
    valid_indices = valid_df.index.tolist()
    for i, idx in enumerate(valid_indices):
        if i < len(forced_uniques):
            lied_df.at[idx, target_col] = forced_uniques[i]
        else:
            lied_df.at[idx, target_col] = random.choice(forced_uniques)

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