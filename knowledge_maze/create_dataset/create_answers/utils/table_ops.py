import random

from knowledge_maze.create_dataset.create_answers.utils import math_ops


def shuffle_column(df, col_name):
    """Shuffles the values in a specific column (Used in Extreme, Comparative)."""
    new_df = df.copy()
    shuffled_vals = new_df[col_name].sample(frac=1).reset_index(drop=True)
    new_df[col_name] = shuffled_vals
    return new_df

def drop_random_rows(df, max_drop=3):
    """Drops a random small number of rows (Used in COUNT_TOTAL)."""
    drop_count = random.randint(1, min(max_drop, len(df) - 1))
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