import re

def normalize_text(text):
    """Removes punctuation, lowers case, strips whitespace for soft equality."""
    if not isinstance(text, str):
        return text
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text.strip()

def get_fuzzy_counts(series):
    """Returns a frequency dict treating semantically similar strings as equal."""
    normalized_series = series.apply(normalize_text)
    return normalized_series.value_counts().to_dict()