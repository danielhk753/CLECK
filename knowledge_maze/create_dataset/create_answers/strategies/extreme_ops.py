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
import dateparser
import pandas as pd
from datetime import datetime, timedelta

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


def detect_template(series):
    """Identifies the dominant format: 'year', 'span', or 'verbose'."""
    sample = str(series.dropna().iloc[0])
    if re.fullmatch(r'\d{4}', sample): return 'year'
    if re.search(r'(\d{4})[–—-]', sample): return 'span'
    return 'verbose'


def clean_by_template(val, template, is_min):
    """Tailored cleaning based on the detected template."""
    if pd.isna(val): return None
    text = re.sub(r'\[.*?\]', '', str(val)).strip()  # Kill brackets

    if template == 'span':
        # Split 1901–1927; take start for Earliest, end for Latest
        parts = re.split(r'[–—-]', text)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts: return None
        # Handle 'present' or 'current' by using today's date
        target = parts[0] if is_min else parts[-1]
        if any(word in target.lower() for word in ['present', 'current', 'now']):
            return datetime.now()
        text = target

    return dateparser.parse(text, settings={'PREFER_DAY_OF_MONTH': 'first'})


def run_extreme_temporal(sample):
    df = pd.DataFrame(sample.get('Relevant table data', []))
    if df.empty: return None

    entity_col = list(df.columns)[0]
    target_col = list(df.columns)[1] if len(df.columns) > 1 else df.columns[0]
    question = sample.get('Question', '').lower()
    is_min = any(word in question for word in ['earliest', 'first', 'oldest', 'min'])

    # --- 1. DETECT TEMPLATE & CLEAN ---
    template = detect_template(df[target_col])
    date_map = {}
    for idx, val in df[target_col].items():
        dt = clean_by_template(val, template, is_min)
        if dt: date_map[idx] = dt

    # Quality Gate: Return an empty dict instead of None to avoid .pop() errors
    if len(date_map) < 2:
        return {"true_answer": "N/A", "counter_answer_1": "N/A", "counter_answer_2": "N/A", "new_evidence": []}

    # --- 2. TRUE & COUNTER 1 (Real Data) ---
    sorted_idx = sorted(date_map.keys(), key=lambda i: date_map[i], reverse=not is_min)
    true_idx = sorted_idx[0]
    true_ans = str(df.at[true_idx, entity_col])

    counter_1 = "N/A"
    for idx in sorted_idx[1:]:
        cand = str(df.at[idx, entity_col])
        if cand != true_ans:
            counter_1 = cand
            break

    # --- 3. COUNTER 2 (The Lie) ---
    potential_c2 = [i for i in df.index if str(df.at[i, entity_col]) not in [true_ans, counter_1]]
    winner_idx = random.choice(potential_c2) if potential_c2 else df.index[0]
    counter_2 = str(df.at[winner_idx, entity_col])

    # --- 4. SYNTHETIC EVIDENCE (The "No-Tie" Generator) ---
    lied_df = df.copy()

    # Global Shift to break parametric memory
    year_shift = random.randint(5, 20) * random.choice([-1, 1])
    base_dt = min(date_map.values()) + timedelta(days=year_shift * 365)

    # CRITICAL: Step size must be > 1 year if we are only displaying years
    step_days = 400 if template in ['year', 'span'] else 45

    # Generate a pool of strictly unique dates
    pool = [base_dt + timedelta(days=i * step_days) for i in range(len(df))]
    pool.sort(reverse=not is_min)  # Winner gets the extreme

    # Assign the extreme to our winner_idx
    final_dates = {winner_idx: pool.pop(0)}

    # Shuffle and assign the rest
    remaining_indices = [i for i in df.index if i != winner_idx]
    random.shuffle(remaining_indices)
    for idx in remaining_indices:
        final_dates[idx] = pool.pop(0)

    # --- 5. FORMATTING ---
    for idx, dt in final_dates.items():
        if template == 'year':
            val = dt.strftime('%Y')
        elif template == 'span':
            # Ensure the span doesn't overlap the next row's start year
            val = f"{dt.year}–{dt.year + random.randint(1, 2)}"
        else:
            val = dt.strftime('%B %d, %Y')
        lied_df.at[idx, target_col] = val

    return {
        "true_answer": true_ans,
        "counter_answer_1": counter_1,
        "counter_answer_2": counter_2,
        "new_evidence": lied_df.to_dict(orient='records')
    }


import pandas as pd
import re
import dateparser
from datetime import datetime


def clean_wiki_text(text):
    """Strips citations, footnotes, trailing text, and standardizes spans."""
    if pd.isna(text): return ""
    text = str(text).strip()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = text.replace('†', '').replace('*', '')

    # Standardize range separators to '–' (en-dash)
    text = re.sub(r'\s*—\s*|\s+-\s+(?!\d{2}\b)|\s+to\s+', '–', text)

    # Clean trailing text after a comma IF it doesn't contain a month or digit
    # Example: "1990, appointed by the board" -> "1990"
    # Protects: "May 12, 1994" or "1990, 1995"
    if ',' in text:
        parts = text.split(',', 1)
        after_comma = parts[1].lower()
        has_month_or_digit = bool(re.search(r'\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec', after_comma))
        if not has_month_or_digit:
            text = parts[0]

    return text.strip()


def parse_table_temporal_data(sample):
    df = pd.DataFrame(sample.get('Relevant table data', []))
    if df.empty: return []

    target_col = list(df.columns)[1] if len(df.columns) > 1 else df.columns[0]
    ent_col = list(df.columns)[0]
    parsed_rows = []

    for _, row in df.iterrows():
        raw_val = row[target_col]
        raw_ent = row[ent_col]
        cleaned = clean_wiki_text(raw_val)

        # 1. NONE
        if not cleaned or cleaned.lower() in ['tba', 'tbd', 'unknown', 'none', '?', '-', '_', 'unannounced']:
            parsed_rows.append({
                ent_col: raw_ent,
                target_col: raw_val, "data_type": "none",
                "start": None, "end": None, "_sort_key": datetime.max
            })
            continue

        # 2. MULTIPLE VALUES (Comma separated)
        # Identifies multiple distinct years/dates (e.g., "1990, 1995" or "1992, 1994, 1996")
        # Ignores standard formats like "May 12, 1994" by ensuring digits exist on both sides of the comma
        if re.search(r'\b\d{4}\b\s*,\s*\b\d{4}\b', cleaned):
            parsed_rows.append({
                ent_col: raw_ent,
                target_col: raw_val, "data_type": "other",
                "start": cleaned, "end": cleaned, "_sort_key": datetime.max
            })
            continue

        # 3. DECADE
        decade_match = re.search(r'(\d{3})0s', cleaned.lower())
        if decade_match and '–' not in cleaned:
            base_year = int(decade_match.group(1) + "0")
            parsed_rows.append({
                ent_col: raw_ent,
                target_col: raw_val, "data_type": "year",
                "start": str(base_year), "end": str(base_year + 9),
                "_sort_key": datetime(base_year, 1, 1)
            })
            continue

        # 4. CENTURY
        century_match = re.search(r'(\d{1,2})(?:st|nd|rd|th)\s*c', cleaned.lower())
        if century_match:
            cent = int(century_match.group(1))
            start_yr = (cent - 1) * 100 + 1
            end_yr = cent * 100
            parsed_rows.append({
                ent_col: raw_ent,
                target_col: raw_val, "data_type": "year",
                "start": str(start_yr), "end": str(end_yr),
                "_sort_key": datetime(start_yr, 1, 1)
            })
            continue

        # 5. SPAN SPLITTING
        if '–' in cleaned:
            parts = cleaned.split('–', 1)
            start_str = parts[0].strip()
            end_str = parts[1].strip()
            # Handle shorthand spans like "1990–95" -> "1990", "1995"
            if re.fullmatch(r'\d{4}', start_str) and re.fullmatch(r'\d{2}', end_str):
                end_str = start_str[:2] + end_str
        else:
            start_str = cleaned
            end_str = cleaned

        # 6. SEMANTIC CLASSIFICATION
        has_year = bool(re.search(r'\b\d{3,4}\b', start_str))
        has_month = bool(re.search(
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\b\d{1,2}/\d{4}\b|\b\d{4}-\d{1,2}\b',
            start_str.lower()))
        no_year_str = re.sub(r'\b\d{4}\b', '', start_str)
        has_day = bool(re.search(r'\b\d{1,2}(?:st|nd|rd|th)?\b', no_year_str))

        if re.search(r'\b\d{4}-\d{2}-\d{2}\b', start_str) or re.search(r'\b\d{1,2}/\d{1,2}/\d{4}\b', start_str):
            data_type = 'day_month_year'
        elif has_year and has_month and has_day:
            data_type = 'day_month_year'
        elif has_year and has_month and not has_day:
            data_type = 'month_year'
        elif not has_year and has_month and has_day:
            data_type = 'day_month'
        elif has_year and not has_month and not has_day:
            data_type = 'year'
        else:
            data_type = 'other'

        # Present/Current logic
        if end_str.lower() in ['present', 'current', 'now', 'ongoing']:
            end_str = str(datetime.now().year) if data_type == 'year' else datetime.now().strftime('%B %d, %Y')

        # 7. SORTING KEY GENERATION
        dt = dateparser.parse(start_str, settings={'PREFER_DAY_OF_MONTH': 'first'}) if start_str else None
        sort_key = dt if dt else datetime.max

        parsed_rows.append({
            ent_col: raw_ent,
            target_col: raw_val,
            "data_type": data_type,
            "start": start_str,
            "end": end_str,
            "_sort_key": sort_key
        })

    # --- FINAL SORT AND CLEANUP ---
    parsed_rows.sort(key=lambda x: x['_sort_key'])
    for row in parsed_rows:
        del row['_sort_key']

    return parsed_rows