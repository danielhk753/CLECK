"""
Filter Counting Module
----------------------
This module provides strategies for identifying and counting entities based on
Categorical, Temporal, and Numerical criteria (e.g., Counts, Ranges, Decades).

It leverages NLP-driven bound extraction and semantic matching to calculate
ground-truth counts, while producing 'lied' evidence through global memory
wipes (timeline/distribution shifting) and targeted nudges to synchronize
the table with a specific structural counterfactual.
"""
import re
import copy
import torch
import random
import pandas as pd
import dateparser
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util

from knowledge_maze.create_dataset.create_answers.utils import math_ops
from knowledge_maze.create_dataset.create_answers.utils.table_ops import swap_values
from knowledge_maze.create_dataset.create_answers.utils.text_ops import TemporalTableParser, fuzzy_match


# ==========================================
# 1. CATEGORICAL_FILTER_LOOKUP
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

def run_categorical_count(sample):
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

    # --- 3. COUNTER ANSWER 2 (The Lie) & VOCAB PRESERVATION ---
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
    min_c2 = 2
    max_c2 = len(df) - num_others

    # Generate pool of valid counts (must be unique from other answers)
    valid_c2_pool = [
        x for x in range(min_c2, max_c2 + 1)
        if x != true_count
    ]

    # Emergency Fallback: If safe range is empty, allow vocab shrinking
    if not valid_c2_pool:
        valid_c2_pool = [
            x for x in range(2, len(df) + 1)
            if x != true_count
        ]

    c2_val = random.choice(valid_c2_pool)
    counter_2 = str(c2_val)

    # --- 4. EXECUTE SAFE SUBSTITUTION ---
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

    # --- 5. COUNTER ANSWER 1 (Scaled) ---
    c1_int = math_ops.random_scale(c2_val)
    while c1_int == true_count or c1_int == c2_val:
        c1_int += random.choice([-1, 1])
    counter_1 = str(max(2, c1_int))

    return {
        "true_answer": true_ans,
        "counter_answer_1": counter_1,
        "counter_answer_2": counter_2,
        "new_evidence": lied_df.to_dict(orient="records")
    }

# ==========================================
# 2. FILTER_COUNT_TEMPORAL
# ==========================================
def extract_temporal_bounds(question):
    """
    Extracts datetime lower and upper bounds from a question.
    Returns (datetime_min, datetime_max).
    """
    q = question.lower()

    # 1. Handle Centuries (e.g., "20th century")
    match = re.search(r'(\d+)(st|nd|rd|th)\s+century', q)
    if match:
        cent = int(match.group(1))
        # 20th century = 1901-01-01 to 2000-12-31
        return datetime((cent - 1) * 100 + 1, 1, 1), datetime(cent * 100, 12, 31)

    # 2. NEW: Handle Decades (e.g., "1970s")
    # We do this before the 4-digit year extractor so "1970s" isn't eaten as "1970"
    decade_match = re.search(r'\b(1[0-9]{3}|20[0-9]{2})s\b', q)
    if decade_match:
        decade_start = int(decade_match.group(1))
        # Returns Jan 1, 1970 to Dec 31, 1979
        return datetime(decade_start, 1, 1), datetime(decade_start + 9, 12, 31)

    # 3. Extract 4-digit years
    years = [int(y) for y in re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', q)]

    if len(years) >= 2:
        # "between 1990 and 2000"
        y1, y2 = min(years[:2]), max(years[:2])
        return datetime(y1, 1, 1), datetime(y2, 12, 31)

    elif len(years) == 1:
        y = years[0]
        # Check for directional keywords
        if any(w in q for w in ['after', 'since', 'post', 'following']):
            return datetime(y + 1, 1, 1), datetime.max
        elif any(w in q for w in ['before', 'until', 'pre', 'prior to']):
            return datetime.min, datetime(y - 1, 12, 31)
        else:
            # "in 1969"
            return datetime(y, 1, 1), datetime(y, 12, 31)

    return None, None


def process_single_table(orig_sample, parsed_sample):
    df_orig = pd.DataFrame(orig_sample.get("Relevant table data", []))
    df_parsed = pd.DataFrame(parsed_sample.get("Relevant table data", []))

    if df_orig.empty or df_parsed.empty or len(df_orig.columns) < 2:
        return None

    target_col = df_orig.columns[1]
    question = orig_sample.get("Question", "").lower()

    # Apply your manual dtype fixes
    df_orig[target_col] = df_orig[target_col].astype(str)
    # Note: Using target_col here as per your manual fix
    parsed_col = f"{target_col} (parsed)"
    if parsed_col not in df_parsed.columns:
        return None

    df_parsed[parsed_col] = df_parsed[parsed_col].astype(str)

    # --- 1. NLP EXTRACTION ---
    lower_bound, upper_bound = extract_temporal_bounds(question)
    if lower_bound is None or upper_bound is None:
        return None

    # --- 2. EXTRACT DATETIMES, NONEs, AND PRESENTs ---
    parsed_spans = {}
    parsed_datetimes = {}
    ended_in_present = {}
    none_indices = []

    format_type = "FMT_YEAR"
    is_span = False
    now = datetime.now()

    prefers_end = any(kw in question for kw in ['leave', 'end', 'death', 'die', 'retire'])

    def parse_bound(b):
        if any(x in b.lower() for x in ["present", "now", "current"]):
            return now
        return dateparser.parse(b)

    for idx, row in df_parsed.iterrows():
        val = str(row[parsed_col]).strip()

        if val.lower() in ["none", "nan", "", "nat"]:
            none_indices.append(idx)
            continue

        if len(val.split("/")) == 3:
            format_type = "FMT_FULL"
        elif len(val.split("/")) == 2:
            format_type = "FMT_MONTH_YEAR"

        if any(x in val.lower() for x in ["present", "now", "current"]):
            ended_in_present[idx] = True

        parts = re.split(r'\s*[\–\-]\s*', val)
        if len(parts) >= 2:
            is_span = True
            start_dt, end_dt = parse_bound(parts[0]), parse_bound(parts[-1])
            if start_dt and end_dt:
                parsed_spans[idx] = (start_dt, end_dt)
                parsed_datetimes[idx] = end_dt if prefers_end else start_dt
        else:
            dt = parse_bound(val)
            if dt: parsed_datetimes[idx] = dt

    # Fill 'None' rows with random dates from the existing spread
    valid_dts = list(parsed_datetimes.values())
    if len(valid_dts) < 2: return None

    data_min_dt, data_max_dt = min(valid_dts), max(valid_dts)
    has_future_dates = data_max_dt > now

    for idx in none_indices:
        span_days = max(1, (data_max_dt - data_min_dt).days)
        rand_dt = data_min_dt + timedelta(days=random.randint(0, span_days))
        parsed_datetimes[idx] = rand_dt
        if is_span:
            parsed_spans[idx] = (rand_dt, rand_dt + timedelta(days=365))

    def is_in_range(dat):
        return lower_bound <= dat <= upper_bound

    # --- 3. TRUE ANSWER & COUNTERS ---
    # Calculated on the ORIGINAL parsed state
    matching_indices = [idx for idx, dt in parsed_datetimes.items() if is_in_range(dt)]
    true_count = len(matching_indices)
    true_ans = str(true_count)

    c1_int = max(1, true_count + random.choice([-2, -1, 1, 2]))
    if c1_int == true_count: c1_int += 1
    counter_1 = str(c1_int)

    # Try to find a count that is NOT the true answer, NOT counter_1,
    # and NOT the total number of rows (leaving a buffer of at least 2).
    total_rows = len(df_orig)
    valid_c2_pool = [
        x for x in range(1, total_rows - 1)
        if x != true_count and x != int(counter_1)
    ]

    # Fallback: If the pool is empty (small table), allow any count except true/counter_1
    if not valid_c2_pool:
        valid_c2_pool = [
            x for x in range(1, total_rows + 1)
            if x != true_count and x != int(counter_1)
        ]

    # Final safety check: if still empty, just pick true_count + 3 (extreme outlier)
    if not valid_c2_pool:
        c2_val = true_count + 3
    else:
        c2_val = random.choice(valid_c2_pool)

    counter_2 = str(c2_val)

    # --- 4. THE MANIPULATION (Memory Wipe + Targeted Nudging) ---
    lied_df = df_parsed.copy()
    lied_df[target_col] = lied_df[target_col].astype(object)
    if parsed_col in lied_df.columns:
        del lied_df[parsed_col]

    global_shift = random.randint(-1095, 1095)
    for idx in parsed_datetimes:
        parsed_datetimes[idx] += timedelta(days=global_shift)
        if idx in parsed_spans:
            s, e = parsed_spans[idx]
            parsed_spans[idx] = (s + timedelta(days=global_shift), e + timedelta(days=global_shift))

    # Re-evaluate who is in range after the wipe
    current_lied_indices = [idx for idx, dt in parsed_datetimes.items() if is_in_range(dt)]
    current_count = len(current_lied_indices)

    wiped_dts = list(parsed_datetimes.values())
    w_min, w_max = min(wiped_dts), max(wiped_dts)
    spread = max(365, (w_max - w_min).days)
    max_drift = max(365, int(spread * 0.2))

    def get_natural_out_of_bounds_dt():
        if lower_bound != datetime.min and upper_bound != datetime.max:
            if random.choice([True, False]):
                return lower_bound - timedelta(days=random.randint(30, max_drift))
            return upper_bound + timedelta(days=random.randint(30, max_drift))
        return (lower_bound - timedelta(days=random.randint(30, max_drift))) if lower_bound != datetime.min else (
                    upper_bound + timedelta(days=random.randint(30, max_drift)))

    def get_natural_in_bounds_dt():
        s_low = max(w_min, lower_bound) if lower_bound != datetime.min else w_min
        s_high = min(w_max, upper_bound) if upper_bound != datetime.max else w_max
        if s_high < s_low: s_low, s_high = lower_bound, upper_bound
        return s_low + timedelta(days=random.randint(0, max(1, (s_high - s_low).days)))

    def write_dt_to_table(idx, target_dt):
        fmt = "%Y" if format_type == "FMT_YEAR" else ("%m/%Y" if format_type == "FMT_MONTH_YEAR" else "%d/%m/%Y")
        if is_span and idx in parsed_spans:
            o_start, o_end = parsed_spans[idx]
            dur = o_end - o_start
            new_s = target_dt if not prefers_end else target_dt - dur
            new_e = target_dt + dur if not prefers_end else target_dt

            s_str = new_s.strftime(fmt)
            if ended_in_present.get(idx, False) or (new_e > now and not has_future_dates):
                e_str = "present"
            else:
                e_str = new_e.strftime(fmt)
            lied_df.at[idx, target_col] = f"{s_str}–{e_str}"
        else:
            lied_df.at[idx, target_col] = target_dt.strftime(fmt)

    # --- 5. EXECUTE TARGETED NUDGING ---
    for idx in lied_df.index:
        if current_count < c2_val and idx not in current_lied_indices:
            new_dt = get_natural_in_bounds_dt()
            current_count += 1
            current_lied_indices.append(idx)
        elif current_count > c2_val and idx in current_lied_indices:
            new_dt = get_natural_out_of_bounds_dt()
            current_count -= 1
            current_lied_indices.remove(idx)
        else:
            new_dt = parsed_datetimes.get(idx, data_min_dt)

        write_dt_to_table(idx, new_dt)

    # Final payload formatting
    sample_out = orig_sample.copy()
    sample_out.update({
        "true_answer": true_ans,
        "counter_answer_1": counter_1,
        "counter_answer_2": counter_2,
        "new_evidence": lied_df.to_dict(orient="records"),
    })
    return sample_out


def run_filter_temporal(samples, verbose=False):
    """
    Driver function that handles batch parsing and processing.
    """
    parser = TemporalTableParser()
    working_samples = copy.deepcopy(samples)

    # Run the batch process to mutate samples in-place
    parser.process_and_normalize_tables(working_samples, verbose=False)

    successful_samples = []
    rejected_count = 0

    # This is exactly your loop structure!
    for orig_sample, parsed_sample in zip(samples, working_samples):
        result = process_single_table(orig_sample, parsed_sample)

        if result is not None:
            successful_samples.append(result)
        else:
            rejected_count += 1

    if verbose:
        print("=" * 40)
        print("TEMPORAL FILTER GENERATION SUMMARY")
        print("=" * 40)
        print(f"Total Initial Tables : {len(samples)}")
        print(f"Successfully Created : {len(successful_samples)}")
        print(f"Rejected / Failed    : {rejected_count}")
        print("=" * 40)

    return successful_samples


# ==========================================
# 3. FILTER_COUNT_NUMERICAL
# ==========================================
def extract_bounds_from_question(question):
    q = question.lower()

    # 1. Strip commas inside numbers (e.g., "15,000" -> "15000")
    q = re.sub(r'(?<=\d),(?=\d)', '', q)

    # 2. Map common spelled-out numbers
    word_to_num = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'twice': 2, 'single': 1, 'half a million': 500000
    }

    # Handle specific idioms first
    for word, val in word_to_num.items():
        if word in q and ('more than ' + word in q or 'at least ' + word in q):
            return val, float('inf')

    # 3. Look for comparison anchors
    greater_anchors = ['more than', 'greater than', 'above', 'higher than', 'exceeding', 'at least']
    lesser_anchors = ['less than', 'below', 'under', 'fewer than']

    all_anchors = greater_anchors + lesser_anchors

    for anchor in all_anchors:
        if anchor in q:
            # Grab the text immediately after the anchor
            after_anchor = q.split(anchor)[1].strip().split()
            if not after_anchor:
                continue

            candidate = after_anchor[0]

            # Extract digits
            digit_match = re.search(r'^\d*\.?\d+', candidate)
            if digit_match:
                val = float(digit_match.group())

                # Check the very next word for scale (e.g., "20 million")
                if len(after_anchor) > 1 and 'million' in after_anchor[1]:
                    val *= 1_000_000
                elif len(after_anchor) > 1 and 'billion' in after_anchor[1]:
                    val *= 1_000_000_000

                if anchor in greater_anchors:
                    return val, float('inf')
                else:
                    return float('-inf'), val

    # 4. Handle edge cases like "35 or more" (number comes before anchor)
    reverse_match = re.search(r'(\d+)\s+or\s+more', q)
    if reverse_match:
        return float(reverse_match.group(1)), float('inf')

    return None, None


def run_filter_numerical(sample):
    df = pd.DataFrame(sample.get('Relevant table data', []))
    if df.empty or len(df.columns) < 2:
        return None

    target_col = df.columns[1]
    question = sample.get("Question", "")

    # Resolution Detection (e.g., 1000 for population, 0.1 for ratings)
    res = math_ops.get_column_resolution(df[target_col])

    # --- 1. NLP EXTRACTION: IDENTIFY RANGE/BOUNDS FROM QUESTION ---
    # Since extract_bounds_from_question returns a tuple directly,
    # we can bypass list checks and unpack it safely!
    lower_bound, upper_bound = extract_bounds_from_question(question)

    if lower_bound is None and upper_bound is None:
        return None  # No numerical criteria found in the question

    def is_in_range(val):
        try:
            f_val = float(val)
            return lower_bound <= f_val <= upper_bound
        except:
            return False

    # --- 2. TRUE ANSWER ---
    matching_indices = [idx for idx, val in df[target_col].items() if is_in_range(val)]
    true_count = len(matching_indices)
    true_ans = str(true_count)

    # --- 3. COUNTER ANSWER 2 (The Structural Lie) ---
    # Pool of valid counts excluding true answer and counter 1 to avoid collisions
    valid_c2_pool = [
        x for x in range(2, len(df) + 1)
        if x != true_count
    ]
    if not valid_c2_pool:
        return None

    c2_val = random.choice(valid_c2_pool)
    counter_2 = str(c2_val)

    # --- 4. THE MANIPULATION (Memory Wipe + Syncing the Count) ---
    lied_df = df.copy()

    # Memory Wipe: Shift the entire column's distribution slightly
    global_shift = random.uniform(0.7, 1.3)

    def mutate(val):
        try:
            return math_ops.round_to_resolution(float(val) * global_shift, res)
        except:
            return val

    lied_df[target_col] = lied_df[target_col].apply(mutate)

    # Calculate realistic data bounds to make fake numbers look natural
    try:
        numeric_vals = pd.to_numeric(lied_df[target_col], errors='coerce').dropna()
        data_min = float(numeric_vals.min())
        data_max = float(numeric_vals.max())
        data_spread = data_max - data_min if data_max > data_min else 100
    except:
        data_min, data_max, data_spread = 0, 100, 100

    # Recalculate who falls in range AFTER the memory wipe
    current_lied_indices = [
        idx for idx, val in lied_df[target_col].items()
        if is_in_range(val)
    ]
    current_count = len(current_lied_indices)

    # Force the count to reach exact c2_val
    if current_count < c2_val:
        # We need MORE items in the range. Move random outliers IN.
        needed = c2_val - current_count
        outside_indices = [idx for idx in lied_df.index if idx not in current_lied_indices]
        to_move_in = random.sample(outside_indices, min(needed, len(outside_indices)))

        for idx in to_move_in:
            if lower_bound != float('-inf') and upper_bound != float('inf'):
                new_val = random.uniform(lower_bound, upper_bound)
            elif lower_bound != float('-inf'):
                # Pick a random number between the threshold and the highest natural number
                highest_natural = max(data_max, lower_bound + data_spread * 0.2)
                new_val = random.uniform(lower_bound + res, highest_natural)
            else:
                # Pick a random number between the lowest natural number and the threshold
                lowest_natural = min(data_min, upper_bound - data_spread * 0.2)
                new_val = random.uniform(lowest_natural, upper_bound - res)

            lied_df.at[idx, target_col] = math_ops.round_to_resolution(new_val, res)

    elif current_count > c2_val:
        # We need FEWER items in the range. Move items OUT.
        excess = current_count - c2_val
        to_move_out = random.sample(current_lied_indices, excess)

        for idx in to_move_out:
            if lower_bound != float('-inf') and upper_bound != float('inf'):
                # 50/50 chance to push it below the min bound or above the max bound
                if random.choice([True, False]):
                    lowest_natural = min(data_min, lower_bound - data_spread * 0.2)
                    new_val = random.uniform(lowest_natural, lower_bound - res)
                else:
                    highest_natural = max(data_max, upper_bound + data_spread * 0.2)
                    new_val = random.uniform(upper_bound + res, highest_natural)
            elif lower_bound != float('-inf'):
                # It's currently > threshold, push it BELOW the threshold naturally
                lowest_natural = min(data_min, lower_bound - data_spread * 0.2)
                new_val = random.uniform(lowest_natural, lower_bound - res)
            else:
                # It's currently < threshold, push it ABOVE the threshold naturally
                highest_natural = max(data_max, upper_bound + data_spread * 0.2)
                new_val = random.uniform(upper_bound + res, highest_natural)

            lied_df.at[idx, target_col] = math_ops.round_to_resolution(new_val, res)

    # --- 5. COUNTER ANSWER 1 (Scaled) ---
    c1_int = math_ops.random_scale(c2_val)
    while c1_int == true_count or c1_int == c2_val:
        c1_int += random.choice([-1, 1])
    counter_1 = str(max(2, c1_int))

    return {
        "true_answer": true_ans,
        "counter_answer_1": counter_1,
        "counter_answer_2": counter_2,
        "new_evidence": lied_df.to_dict(orient='records')
    }