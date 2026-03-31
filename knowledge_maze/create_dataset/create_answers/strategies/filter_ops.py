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
# TODO: Make sure that all the unique values in col_1 get at least 1 appearance in the new_evidences.
# TODO: Look if loading the small llm has a lot of latency that i would want it to be loaded once in the beginning.
def process_categorical_count(sample):
    """
    Handles: "How many of the entities in col 0 have the value in col 1 be equal to X"
    Strategically finds 'X' by locating the longest matching phrase from Col 1 in the Question.
    """
    df = pd.DataFrame(sample.get("Relevant table data", []))
    if df.empty or len(df.columns) < 2:
        return None

    entity_col = df.columns[0]
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
        model = SentenceTransformer('all-MiniLM-L6-v2')
        val_embeddings = model.encode(unique_vals, convert_to_tensor=True)
        q_embedding = model.encode(question, convert_to_tensor=True)

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

    # --- 4. COUNTER ANSWER 2 (The Lie) ---
    lied_df = df.copy()
    non_matching_indices = [idx for idx in df.index if idx not in matching_indices]

    strategy = random.choice(["inflate", "deflate"])

    if strategy == "inflate" and non_matching_indices:
        # Increase occurrences of target_val
        num_to_add = random.randint(1, len(non_matching_indices))
        indices_to_change = random.sample(non_matching_indices, num_to_add)
        for idx in indices_to_change:
            lied_df.at[idx, target_col] = target_val

    elif strategy == "deflate" and matching_indices:
        # Decrease occurrences of target_val
        num_to_remove = random.randint(1, len(matching_indices))
        indices_to_change = random.sample(matching_indices, num_to_remove)

        # Get pool of other values to replace target_val with
        other_vals = [val for idx, val in df[target_col].items() if idx not in matching_indices]
        if not other_vals:
            other_vals = [f"Not_{target_val}"]

        for idx in indices_to_change:
            lied_df.at[idx, target_col] = random.choice(other_vals)

    # Recalculate count for Counter 2
    new_matching = [
        idx for idx, val in lied_df[target_col].items()
        if fuzzy_match(val, target_val, threshold=0.85)
    ]
    counter_2 = str(len(new_matching))

    # --- 5. DYNAMIC COLLISION AVOIDANCE & TABLE FIXING ---
    # Ensure counter_2 is unique among the three answers
    used_answers = {int(true_ans), int(counter_1)}
    c2_val = int(counter_2)

    # If c2 collide, shift it until it's unique (and positive)
    while c2_val in used_answers or c2_val < 0:
        c2_val += 1 if c2_val < len(df) else -1

    counter_2 = str(c2_val)

    # Now, forcefully sync the table to match this specific c2_val
    current_lied_indices = [
        idx for idx, val in lied_df[target_col].items()
        if fuzzy_match(val, target_val)
    ]
    current_count = len(current_lied_indices)

    if current_count < c2_val:
        # We need more: Flip random non-matching rows to the target_val
        needed = c2_val - current_count
        potential_rows = [i for i in lied_df.index if i not in current_lied_indices]
        if potential_rows:
            to_flip = random.sample(potential_rows, min(needed, len(potential_rows)))
            for idx in to_flip:
                lied_df.at[idx, target_col] = target_val

    elif current_count > c2_val:
        # We have too many: Flip random matching rows to something else
        excess = current_count - c2_val
        to_remove = random.sample(current_lied_indices, excess)

        # Pick a replacement that isn't the target
        alt_pool = [v for v in unique_vals if not fuzzy_match(v, target_val)]
        replacement = alt_pool[0] if alt_pool else f"Not_{target_val}"

        for idx in to_remove:
            lied_df.at[idx, target_col] = replacement

    # Final Return (Guaranteed no None)
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
def process_filter_numerical(orig_sample, parsed_sample):
    """
    Relies on NumericalTableParser (which uses math_ops logic internally).
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

    parsed_col = [c for c in df_parsed.columns if c not in df_orig.columns][-1]
    is_greater = any(w in question for w in ["more", "greater", "above", "higher", "over"])

    parsed_vals = {idx: float(v) for idx, v in df_parsed[parsed_col].items() if not pd.isna(v)}
    if len(parsed_vals) < len(df_orig): return None

    sorted_idx = sorted(parsed_vals.keys(), key=lambda i: parsed_vals[i])
    if is_greater:
        passing_indices = sorted_idx[-true_count:] if true_count > 0 else []
        failing_indices = sorted_idx[:-true_count] if true_count > 0 else sorted_idx
    else:
        passing_indices = sorted_idx[:true_count] if true_count > 0 else []
        failing_indices = sorted_idx[true_count:] if true_count > 0 else sorted_idx

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