import random
from itertools import combinations

from knowledge_maze.create_dataset.create_answers.utils import math_ops
from knowledge_maze.create_dataset.create_answers.utils.text_ops import light_match


def shuffle_column(df, col_name):
    """Shuffles the values in a specific column (Used in Extreme, Comparative)."""
    new_df = df.copy()
    shuffled_vals = new_df[col_name].sample(frac=1).reset_index(drop=True)
    new_df[col_name] = shuffled_vals
    return new_df

def drop_random_rows(df, max_drop=3):
    """Drops a random small number of rows (Used in COUNT_TOTAL)."""
    drop_count = random.randint(1, min(max_drop, len(df) - 2))
    drop_indices = random.sample(list(df.index), drop_count)
    return df.drop(drop_indices).reset_index(drop=True)

def merge_categories(df, col_name, target_cat, source_cats):
    """Switches values to make categories disappear or change mode (Used in Categorical)."""
    new_df = df.copy()
    new_df[col_name] = new_df[col_name].apply(lambda x: target_cat if x in source_cats else x)
    return new_df


def scale_column_values(df, col_name):
    """Applies random_scale to every row in a column, preserving precision."""
    new_df = df.copy()
    new_df[col_name] = new_df[col_name].apply(lambda x: math_ops.random_scale(x))
    return new_df


def swap_values(df, idx_a, idx_b, col_name):
    """Swaps the values of two specific rows in a given column."""
    new_df = df.copy()
    val_a = new_df.at[idx_a, col_name]
    val_b = new_df.at[idx_b, col_name]

    new_df.at[idx_a, col_name] = val_b
    new_df.at[idx_b, col_name] = val_a
    return new_df


def resolve_table_entities(table_data):
    """
    Identifies 'light matches' in Column 1 and collapses longer strings
    into their shorter, contained versions across the entire table.
    """
    if not table_data:
        return []

    # 1. Identify the key for Column 1
    # We assume the table is a list of dicts; index 0 is Col0, index 1 is Col1.
    sample_row = table_data[0]
    keys = list(sample_row.keys())
    if len(keys) < 2:
        return table_data  # Nothing to compare in Col1

    target_key = keys[1]

    # 2. Get all unique values from Col 1 to find overlap pairs
    unique_values = list(set(row[target_key] for row in table_data if row.get(target_key)))

    # 3. Build a replacement map (Long String -> Short String)
    replacement_map = {}

    # Use combinations to check every unique pair once
    for val1, val2 in combinations(unique_values, 2):
        if light_match(val1, val2):
            # Identify which is shorter
            short, long = (val1, val2) if len(val1) < len(val2) else (val2, val1)
            replacement_map[long] = short

    # 4. Apply the replacements to the entire table
    new_table = []
    for row in table_data:
        new_row = row.copy()
        current_val = new_row.get(target_key)

        # If the value is a 'long' version we found earlier, swap it
        if current_val in replacement_map:
            new_row[target_key] = replacement_map[current_val]

        new_table.append(new_row)

    return new_table