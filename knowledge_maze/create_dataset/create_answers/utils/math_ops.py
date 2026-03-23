import random
import numpy as np


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