import random
import numpy as np
import pandas as pd


def get_precision(num):
    """Detects the number of decimal places in a number."""
    if isinstance(num, (int, np.integer)) or (isinstance(num, float) and num.is_integer()):
        return 0
    s = str(num)
    if '.' not in s:
        return 0
    return len(s.split('.')[-1])


def random_scale(value, min_scale=0.5, max_scale=1.5):
    """Scales a number, preserving original integer/decimal precision."""
    # Convert numpy types to python natives
    if hasattr(value, 'item'):
        value = value.item()

    precision = get_precision(value)
    scale = random.uniform(min_scale, max_scale)
    new_value = value * scale

    if precision == 0:
        return int(round(new_value))

    return round(new_value, precision)


def get_column_resolution(series):
    """
    Determines the rounding resolution of a numerical series.
    E.g., [1000, 2500, 50000] -> returns 1000
    E.g., [1.5, 2.0, 3.5] -> returns 0.5
    """
    valid_vals = pd.to_numeric(series, errors='coerce').dropna()
    if len(valid_vals) == 0: return 1

    # Check for decimals first
    precisions = [len(str(v).split('.')[1]) for v in valid_vals if '.' in str(v) and float(v) != int(float(v))]
    if precisions:
        return 10 ** -max(precisions)

    # For integers, find the minimum number of trailing zeros
    # e.g., '15000' has 3 trailing zeros (resolution 1000)
    int_vals = valid_vals.astype(int).astype(str)
    trailing_zeros = [len(v) - len(v.rstrip('0')) for v in int_vals if v != '0' and v.rstrip('0')]

    if trailing_zeros:
        min_zeros = min(trailing_zeros)
        return 10 ** min_zeros

    return 1


def round_to_resolution(value, resolution):
    """Rounds a value to the nearest step of the resolution."""
    if resolution == 0: return value
    return round(value / resolution) * resolution


def generate_from_distribution(series, resolution, scale_factor=1.0):
    """
    Generates a believable new value based on the column's actual distribution.
    Uses a normal distribution based on the original mean and std dev.
    """
    valid_vals = pd.to_numeric(series, errors='coerce').dropna()
    mean = valid_vals.mean() * scale_factor
    std = valid_vals.std() * scale_factor

    if pd.isna(std) or std == 0:
        std = abs(mean) * 0.1 if mean != 0 else 1

    new_val = np.random.normal(mean, std)
    return round_to_resolution(new_val, resolution)