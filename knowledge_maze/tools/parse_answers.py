"""parse numerical and date types answers."""

import re

NUMBERS = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
    'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
    'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80,
    'ninety': 90, 'one hundred': 100
}
QUANTITIES = {
    'hundred': 100, 'thousand': 1_000, 'million': 1_000_000, 'billion': 1_000_000_000
}
FRACTIONS = {
    'half': 0.5, 'quarter': 0.25, 'third': 1/3
}
MONTHS = [
    'January', 'February', 'March', 'April',
    'May', 'June', 'July', 'August', 'September',
    'October', 'November', 'December'
]
DAYS = [
    'Sunday', 'Monday', 'Tuesday', 'Wednesday',
    'Thursday', 'Friday', 'Saturday'
]
# Regex pattern for different year span formats, including shorthand years (e.g., "1933 -- 35")
YEAR_SPAN_PATTERN = re.compile(r'(\d{4})\s*[-–—]{1,2}\s*(\d{2,4})|(\d{4})\s+to\s+(\d{2,4})', re.IGNORECASE)

def extract_number(answer: str) -> int | float | str:
    """
    Extract numerical values from a string, handling words, ranges, and quantities.

    Args:
        answer (str): The input string containing numbers.

    Returns:
        int | float | str: Extracted number as int if possible, else float, or original string if no numbers exist.
    """
    # Remove commas inside numbers (e.g., "121,364" → "121364")
    answer = re.sub(r'(\d),(\d)', r'\1\2', answer)

    # Remove special characters but keep decimal points inside numbers
    answer = re.sub(r"[^\w\s.]|(?<=\d)\.(?=\D)|(?<=\D)\.(?=\d)", "", answer)

    # Handle ranges like "10-30" → take the midpoint
    range_match = re.search(r'(-?\d+\.?\d*)\s*-\s*(-?\d+\.?\d*)', answer)
    if range_match:
        num1, num2 = map(float, range_match.groups())
        midpoint = (num1 + num2) / 2
        return int(midpoint) if midpoint.is_integer() else midpoint

    # Replace number words with digits
    words = answer.lower().split()
    total = 0
    current_num = 0
    has_num = False

    for word in words:
        if word.replace('.', '', 1).isdigit():  # Check if it's a valid number (including floats)
            current_num = float(word)  # Update current number
            has_num = True
        elif word in NUMBERS:
            if current_num == 0:
                current_num = NUMBERS[word]  # Convert number words
            else:
                current_num += NUMBERS[word]
            has_num = True
        elif word in QUANTITIES:
            if not has_num:
                current_num = 1  # If "million" appears alone, assume "1 million"
            current_num *= QUANTITIES[word]  # Multiply the previous number by the quantity
            total += current_num
            current_num = 0  # Reset after applying multiplier
            has_num = True
        elif word in FRACTIONS:
            fraction_value = FRACTIONS[word]
            if has_num:
                current_num += fraction_value
            else:
                current_num = fraction_value
            has_num = True
        else:
            continue

    total += current_num  # Add any remaining value

    if total > 0:
        try:
            if total.is_integer():
                return int(total)
        except:
            return total

    # Extract numeric values including decimals
    num_match = re.findall(r'[-]?\d+\.?\d*', answer)
    if num_match:
        num = float(num_match[0])
        return int(num) if num.is_integer() else num

    return answer  # If no numeric content is found, return original string


def extract_date(answer: str) -> dict:
    """
    Parses a date-related answer and categorizes it into structured formats.

    This function processes the sample, cleans unnecessary punctuation and words, and classifies it into
    predefined categories such as full dates, year spans, decades, centuries, and more. It also ensures
    correct formatting of day-month-year sequences.

    Args:
        answer (str): A list of dictionaries where each dictionary contains an 'Answer' key
                      representing a date-related string.

    Returns:
        dict: A structured dictionary with:
            - 'category' (str): The identified category of the date.
            - 'parsed_value' (str or int): A cleaned and structured representation of the date.
    """
    # Strip unnecessary punctuation and whitespace
    answer = re.sub(r'[.,]', '', answer).strip()
    words = answer.split()

    # Remove unnecessary words
    words = [word for word in words if word.lower() not in ('in', 'early', 'mid', 'late')]
    answer_cleaned = ' '.join(words)  # Reconstruct string after filtering
    category = 'unknown'

    # Case: Full date (day, month, year in any order)
    if any(month in words for month in MONTHS) and any(word.isdigit() and len(word) == 4 for word in words) and any(
            word.isdigit() and len(word) <= 2 for word in words):
        category = 'full_date'
        day = next((word for word in words if word.isdigit() and len(word) <= 2), '')
        month = next((word for word in words if word in MONTHS), '')
        year = next((word for word in words if word.isdigit() and len(word) == 4), '')
        answer_cleaned = f"{day} {month} {year}".strip()

    # Case: Year span (handles "YYYY - YYYY", "YYYY -- YYYY", "YYYY — YYYY", "YYYY to YYYY", AND "YYYY -- YY")
    elif YEAR_SPAN_PATTERN.search(answer_cleaned):
        match = YEAR_SPAN_PATTERN.search(answer_cleaned)
        year1 = int(match.group(1) or match.group(3))  # First year
        year2 = match.group(2) or match.group(4)  # Second year (could be short form)

        # If year2 is a two-digit year, expand it correctly
        if year2 and len(year2) == 2:
            base_year = year1 // 100 * 100  # Get the century part (e.g., 1900 for 1933)
            year2 = base_year + int(year2)  # Convert "35" → "1935"

        answer_cleaned = f"{year1} - {year2}"
        category = 'year_span'

    # Case: Month + Year in any order (e.g., "March 2020" or "2020 March")
    elif len(words) == 2 and any(word in MONTHS for word in words) and any(
            word.isdigit() and len(word) == 4 for word in words):
        category = 'month_year'
        month = next((word for word in words if word in MONTHS), '')
        year = next((word for word in words if word.isdigit() and len(word) == 4), '')
        answer_cleaned = f"{month} {year}".strip()

    # Case: Day + Month in any order (e.g., "March 15" or "15 March")
    elif len(words) == 2 and any(word in MONTHS for word in words) and any(
            word.isdigit() and len(word) <= 2 for word in words):
        category = 'day_month'
        day = next((word for word in words if word.isdigit() and len(word) <= 2), '')
        month = next((word for word in words if word in MONTHS), '')
        answer_cleaned = f"{day} {month}".strip()

    # Case: Single year (e.g., "1999")
    elif len(words) == 1 and words[0].isdigit() and int(words[0]) > 1000:
        category = 'year'

    # Case: Decades (e.g., "1920s")
    elif len(words) == 1 and words[0][-1] == 's' and words[0][:-1].isdigit():
        category = 'decade'
        answer_cleaned = words[0][:-1]  # Remove 's' to keep only the number
        if len(answer_cleaned) < 4:
            answer_cleaned = '19' + answer_cleaned

    # Case: Centuries (e.g., "18th century", "20th century")
    elif 'century' in answer_cleaned:
        category = 'century'
        if re.search(r'\d+', answer_cleaned):
            answer_cleaned = re.search(r'\d+', answer_cleaned).group()
        else:
            category = 'unknown'

    # Case: Generic Month (e.g., "July")
    elif len(words) == 1 and words[0] in MONTHS:
        category = 'generic_month'

    # Case: Generic Day (e.g., "Monday")
    elif len(words) == 1 and words[0] in DAYS:
        category = 'generic_day'

    return {'category': category, 'parsed_value': answer_cleaned}
