import re
import torch
import random
import pandas as pd
from sentence_transformers import SentenceTransformer, util

from knowledge_maze.create_dataset.create_answers.utils import math_ops
from knowledge_maze.create_dataset.create_answers.utils.table_ops import swap_values
from knowledge_maze.create_dataset.create_answers.utils.text_ops import TemporalTableParser, fuzzy_match


# ==========================================
# 1. CATEGORICAL FILTER LOOKUP
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

def process_categorical_count(sample):
    """
    Handles: "How many of the entities in col 0 have the value in col 1 be equal to X"
    Strategically finds 'X' by locating the longest matching phrase from Col 1 in the Question.
    """
    df = pd.DataFrame(sample.get("Relevant table data", []))
    if df.empty or len(df.columns) < 2:
        return None

    target_col = df.columns[1]
    question = sample.get("Question", "")

    # --- 1. NLP EXTRACTION: FIND THE LONGEST PHRASE FROM COL 1 IN THE QUESTION ---
    # Get all unique valid strings from column 1
    unique_vals = df[target_col].dropna().astype(str).unique().tolist()

    # Strategy A: Find the longest phrase from the table that exists in the question
    matching_phrases = [v for v in unique_vals if v.lower() in question.lower()]

    if matching_phrases:
        target_val = max(matching_phrases, key=len)
    else:
        # Strategy B: Semantic Fallback (Spain vs ESP, etc.)
        # This only runs if the loop above finds nothing.
        # Lazy-load model to keep it from being "heavy" on every call
        val_embeddings = MODEL.encode(unique_vals, convert_to_tensor=True)
        q_embedding = MODEL.encode(question, convert_to_tensor=True)

        # Find the value with the highest cosine similarity to the question
        hits = util.cos_sim(q_embedding, val_embeddings)[0]
        best_idx = torch.argmax(hits).item()

        # Threshold check to ensure it's actually relevant
        target_val = unique_vals[best_idx] if torch.max(hits) > 0.3 else None

    if not target_val:
        return None

    # Tie-breaker: Grab the longest phrase to avoid "US" matching inside "USA"
    if matching_phrases:
        target_val = max(matching_phrases, key=len)

    # --- 2. FUZZY COUNTING ---
    valid_df = df.dropna(subset=[target_col])

    # Using the fuzzy_match we built earlier
    matching_indices = [
        idx for idx, val in valid_df[target_col].items()
        if fuzzy_match(val, target_val, threshold=0.85)
    ]

    true_count = len(matching_indices)
    true_ans = str(true_count)

    # --- 3. COUNTER ANSWER 1 (Scaled) ---
    c1_int = math_ops.random_scale(true_count)
    if c1_int == true_count:
        c1_int = true_count + random.choice([-1, 1])
    counter_1 = str(max(0, c1_int))

    # --- 4. COUNTER ANSWER 2 (The Lie) & VOCAB PRESERVATION ---
    lied_df = df.copy()

    # Analyze current distribution to protect singletons
    other_indices = [idx for idx in df.index if idx not in matching_indices]
    other_counts = {}
    for idx in other_indices:
        val = df.at[idx, target_col]
        other_counts[val] = other_counts.get(val, 0) + 1

    num_others = len(other_counts)

    # Calculate the strict "Safe Range" to preserve vocabulary size
    # Total rows - count of unique other values = max rows target_val can take
    min_c2 = 1
    max_c2 = len(df) - num_others

    # Generate pool of valid counts (must be unique from other answers)
    valid_c2_pool = [
        x for x in range(min_c2, max_c2 + 1)
        if x != true_count and x != int(counter_1)
    ]

    # Emergency Fallback: If safe range is empty, allow vocab shrinking
    if not valid_c2_pool:
        valid_c2_pool = [
            x for x in range(0, len(df) + 1)
            if x != true_count and x != int(counter_1)
        ]

    c2_val = random.choice(valid_c2_pool)
    counter_2 = str(c2_val)

    # --- 5. EXECUTE SAFE SUBSTITUTION ---
    # This replaces target_val count to match c2_val exactly
    if true_count < c2_val:
        # INFLATE: Overwrite only 'surplus' instances of other values
        needed = c2_val - true_count
        safe_indices = []
        temp_counts = other_counts.copy()

        # Shuffle indices to vary which values get cannibalized
        random.shuffle(other_indices)
        for idx in other_indices:
            val = lied_df.at[idx, target_col]
            if temp_counts[val] > 1:
                safe_indices.append(idx)
                temp_counts[val] -= 1

        to_flip = random.sample(safe_indices, min(needed, len(safe_indices)))
        for idx in to_flip:
            lied_df.at[idx, target_col] = target_val

    elif true_count > c2_val:
        # DEFLATE: Change excess target_vals into other existing values
        excess = true_count - c2_val
        to_remove = random.sample(matching_indices, excess)

        other_vals_list = list(other_counts.keys())
        # If no other values exist in table, use a generic placeholder
        replacement_pool = other_vals_list if other_vals_list else [f"Not_{target_val}"]

        for idx in to_remove:
            lied_df.at[idx, target_col] = random.choice(replacement_pool)

    return {
        "true_answer": true_ans,
        "counter_answer_1": counter_1,
        "counter_answer_2": counter_2,
        "new_evidence": lied_df.to_dict(orient="records")
    }

# ==========================================
# 2. FILTER COUNT TEMPORAL
# ==========================================
# TODO: Unfinished function.
def process_filter_temporal(orig_sample, parsed_sample):
    """
    Uses TemporalTableParser.
    Strategy: Identifies 'Passing' strings and 'Failing' strings,
    then resamples them into a new DF to meet fake_count.
    """
    df_orig = pd.DataFrame(orig_sample.get("Relevant table data", []))
    df_parsed = pd.DataFrame(parsed_sample.get("Relevant table data", []))
    if df_orig.empty or df_parsed.empty: return None

    target_col = list(df_orig.columns)[1] if len(df_orig.columns) > 1 else df_orig.columns[0]
    question = orig_sample.get("Question", "").lower()

    try:
        true_count = int(float(orig_sample.get("true_answer")))
        fake_count = int(float(orig_sample.get("counter_answer_2")))
    except (ValueError, TypeError):
        return None

    if fake_count > len(df_orig) or true_count == fake_count: return None

    parsed_col = [c for c in df_parsed.columns if c not in df_orig.columns][-1]

    # Intent
    start_keywords = ['join', 'start', 'begin', 'member', 'born', 'found', 'became']
    prefers_start = any(kw in question for kw in start_keywords)
    is_after = any(w in question for w in ["after", "since", "newer", "later"])

    parsed_datetimes = {}
    for idx, val in df_parsed[parsed_col].items():
        if pd.isna(val): continue
        # Handle the (start, end) tuple from your TemporalTableParser
        if isinstance(val, tuple):
            parsed_datetimes[idx] = val[0] if prefers_start else val[1]
        else:
            parsed_datetimes[idx] = val

    if len(parsed_datetimes) < len(df_orig): return None

    # Sort to find who originally passed
    sorted_idx = sorted(parsed_datetimes.keys(), key=lambda i: parsed_datetimes[i])
    if is_after:
        passing_indices = sorted_idx[-true_count:] if true_count > 0 else []
        failing_indices = sorted_idx[:-true_count] if true_count > 0 else sorted_idx
    else:
        passing_indices = sorted_idx[:true_count] if true_count > 0 else []
        failing_indices = sorted_idx[true_count:] if true_count > 0 else sorted_idx

    # Resample strings to guarantee the count
    passing_strs = [str(df_orig.at[i, target_col]) for i in passing_indices]
    failing_strs = [str(df_orig.at[i, target_col]) for i in failing_indices]

    lied_df = df_orig.copy()
    all_indices = list(df_orig.index)
    random.shuffle(all_indices)

    for i, idx in enumerate(all_indices):
        source_pool = passing_strs if i < fake_count else failing_strs
        lied_df.at[idx, target_col] = random.choice(source_pool)

    sample_out = orig_sample.copy()
    sample_out.update({"new_evidence": lied_df.to_dict(orient="records")})
    return sample_out


# ==========================================
# 3. FILTER COUNT NUMERICAL
# ==========================================
# TODO: Unfinished function.
def process_filter_numerical(sample):
    df = pd.DataFrame(sample.get('Relevant table data', []))
    if df.empty or len(df.columns) < 2:
        return None

    entity_col = df.columns[0]
    target_col = df.columns[1]
    question = sample.get("Question", "")

    # Resolution Detection (e.g., 1000 for population, 0.1 for ratings)
    res = math_ops.get_column_resolution(df[target_col])

    # --- 1. NLP EXTRACTION: IDENTIFY RANGE/BOUNDS FROM QUESTION ---
    # Find all floats/ints in the question
    numbers = [float(n) for n in re.findall(r'[-+]?\d*\.?\d+', question)]

    if len(numbers) >= 2:
        # Case: "between 10 and 20"
        lower_bound, upper_bound = min(numbers[:2]), max(numbers[:2])
    elif len(numbers) == 1:
        # Case: "greater than 10" or "below 5"
        q_lower = question.lower()
        if any(w in q_lower for w in ['greater', 'above', 'more', 'over', '>', 'longer', 'heavier']):
            lower_bound, upper_bound = numbers[0], float('inf')
        else:
            lower_bound, upper_bound = float('-inf'), numbers[0]
    else:
        return None  # No numerical criteria found in the question

    def is_in_range(val):
        try:
            fval = float(val)
            return lower_bound <= fval <= upper_bound
        except:
            return False

    # --- 2. TRUE ANSWER ---
    matching_indices = [idx for idx, val in df[target_col].items() if is_in_range(val)]
    true_count = len(matching_indices)
    true_ans = str(true_count)

    # --- 3. COUNTER ANSWER 1 (Scaled) ---
    c1_int = math_ops.random_scale(true_count)
    if c1_int == true_count:
        c1_int = true_count + random.choice([-1, 1])
    counter_1 = str(max(0, c1_int))

    # --- 4. COUNTER ANSWER 2 (The Structural Lie) ---
    # Pool of valid counts excluding true answer and counter 1 to avoid collisions
    valid_c2_pool = [
        x for x in range(0, len(df) + 1)
        if x != true_count and x != int(counter_1)
    ]
    if not valid_c2_pool:
        return None

    c2_val = random.choice(valid_c2_pool)
    counter_2 = str(c2_val)

    # --- 5. THE MANIPULATION (Memory Wipe + Syncing the Count) ---
    lied_df = df.copy()

    # Memory Wipe: Shift the entire column's distribution slightly
    global_shift = random.uniform(0.7, 1.3)

    def mutate(val):
        try:
            return math_ops.round_to_resolution(float(val) * global_shift, res)
        except:
            return val

    lied_df[target_col] = lied_df[target_col].apply(mutate)

    # Recalculate who falls in range *after* the memory wipe
    current_lied_indices = [
        idx for idx, val in lied_df[target_col].items()
        if is_in_range(val)
    ]
    current_count = len(current_lied_indices)

    # Force the count to reach exact c2_val
    if current_count < c2_val:
        # We need MORE items in the range. Move random outliers in.
        needed = c2_val - current_count
        outside_indices = [idx for idx in lied_df.index if idx not in current_lied_indices]
        to_move_in = random.sample(outside_indices, min(needed, len(outside_indices)))

        for idx in to_move_in:
            if lower_bound != float('-inf') and upper_bound != float('inf'):
                # Bounded range: pick a random float inside it
                new_val = random.uniform(lower_bound, upper_bound)
            elif lower_bound != float('-inf'):
                new_val = lower_bound + res
            else:
                new_val = upper_bound - res
            lied_df.at[idx, target_col] = math_ops.round_to_resolution(new_val, res)

    elif current_count > c2_val:
        # We need FEWER items in the range. Nudge items just outside.
        excess = current_count - c2_val
        to_move_out = random.sample(current_lied_indices, excess)

        for idx in to_move_out:
            current_val = float(lied_df.at[idx, target_col])
            if lower_bound != float('-inf') and upper_bound != float('inf'):
                # Push out to whichever edge is closer
                if abs(current_val - lower_bound) < abs(current_val - upper_bound):
                    new_val = lower_bound - res
                else:
                    new_val = upper_bound + res
            elif lower_bound != float('-inf'):
                new_val = lower_bound - res
            else:
                new_val = upper_bound + res
            lied_df.at[idx, target_col] = math_ops.round_to_resolution(new_val, res)

    return {
        "true_answer": true_ans,
        "counter_answer_1": counter_1,
        "counter_answer_2": counter_2,
        "new_evidence": lied_df.to_dict(orient='records')
    }