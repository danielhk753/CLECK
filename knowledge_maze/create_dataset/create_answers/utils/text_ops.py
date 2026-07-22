import re
import difflib
import dateparser
import pandas as pd
from collections import Counter


def normalize_string(s):
    """Lowercase, strip punctuation, and remove extra whitespace."""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    # Collapse multiple spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def light_match(val1, val2):
    norm1 = normalize_string(val1)
    norm2 = normalize_string(val2)

    # Avoid identical raw values
    if val1 == val2 or not norm1 or not norm2:
        return False

    # Check soft containment match
    return norm1 in norm2 or norm2 in norm1


def fuzzy_match(val1, val2, threshold=0.85):
    """
    Returns True if strings are logically similar above the threshold.
    Uses difflib's SequenceMatcher after basic normalization.
    """
    norm1 = normalize_string(val1)
    norm2 = normalize_string(val2)

    # Exact match after normalization
    if norm1 == norm2:
        return True

    # Quick token-set check for word-order swaps (e.g., "Apple Inc." vs "Inc Apple")
    set1 = set(norm1.split())
    set2 = set(norm2.split())
    if set1 == set2 and len(set1) > 0:
        return True

    # Fallback to gestalt pattern matching for typos
    ratio = difflib.SequenceMatcher(None, norm1, norm2).ratio()
    return ratio >= threshold


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


class TemporalTableParser:
    def __init__(self):
        # 1. The Waterfall Order
        self.specialists = [
            self.extract_full_date,
            self.extract_month_year,
            self.extract_year,
            self.extract_decade,
            self.extract_century,
            self.extract_day_month,
            self.extract_placeholder  # Catches TBD, None, etc. -> returns 'FMT_NONE'
        ]

        # 2. Regex:
        self.RE_YEAR = r'\b\d{3,4}\b'
        self.RE_MONTH = r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b'
        self.RE_PRESENT = r'\b(present|now|current|ongoing|today)\b'
        self.RE_DECADE = r'\b(\d{2,3}0)s\b'
        self.RE_ORDINAL = r'\b(\d{1,2})(?:st|nd|rd|th)\b'
        self.RE_CENT_KW = r'\b(?:centur(?:y|ies)|c\.|cent\.)'
        self.RE_PLACEHOLDERS = r'\b(tbd|tba|unknown|unannounced|n/a|none|pending|tbc|present|now)\b|[—\-_]$'

    # ---------------------------------------------------------
    # Specialist functions for date type classification:
    # ---------------------------------------------------------
    def extract_full_date(self, text):
        if not text or pd.isna(text):
            return None, None

        def is_strict_full_date(s):
            s_low = str(s).lower()
            # Checks for Year, Month Name, and 1-2 digit Day
            has_year = bool(re.search(self.RE_YEAR, s_low))
            has_month = bool(re.search(self.RE_MONTH, s_low))
            no_year = re.sub(self.RE_YEAR, ' ', s_low)
            days = re.findall(r'\b\d{1,2}\b', no_year)
            has_day = any(1 <= int(d) <= 31 for d in days)
            return has_year and has_month and has_day

        # --- 1. Cleanup ---
        t = str(text).strip()
        t = re.sub(r'\[.*?\]|†|\*', '', t)
        t = re.sub(r'\s*—\s*|\s+to\s+|\s+-\s+', '–', t)

        # --- 2. Resolve Brackets (Sifter) ---
        if '(' in t or ')' in t:
            parts = [p.strip() for p in re.split(r'\(|\)', t) if p.strip()]
            for p in parts:
                if is_strict_full_date(p):
                    t = p
                    break

        # --- 3. Logic: Universal Spans ---
        if '–' in t:
            p_start, p_end = [p.strip() for p in t.split('–', 1)]

            # A. Shared Year (e.g., "July 6 – August 27, 2017")
            if not re.search(self.RE_YEAR, p_start) and re.search(self.RE_YEAR, p_end):
                year_match = re.search(self.RE_YEAR, p_end)
                p_start = f"{p_start} {year_match.group(0)}"

            # B. Shared Month & Year (e.g., "September 7–8, 2013")
            if not re.search(self.RE_MONTH, p_end) and re.search(self.RE_MONTH, p_start):
                month_match = re.search(self.RE_MONTH, p_start.lower())
                year_match = re.search(self.RE_YEAR, p_start)  # or p_end
                if month_match and year_match:
                    p_end = f"{month_match.group(0)} {p_end} {year_match.group(0)}"

            # C. Shared Month (e.g., "3–6 August 1950")
            if not re.search(self.RE_MONTH, p_start) and re.search(self.RE_MONTH, p_end):
                month_match = re.search(self.RE_MONTH, p_end.lower())
                year_match = re.search(self.RE_YEAR, p_end)
                if month_match and year_match:
                    p_start = f"{month_match.group(0)} {p_start} {year_match.group(0)}"

            if is_strict_full_date(p_start) and is_strict_full_date(p_end):
                dt_s = dateparser.parse(p_start)
                dt_e = dateparser.parse(p_end)
                if dt_s and dt_e:
                    return f"{dt_s.strftime('%d/%m/%Y')}-{dt_e.strftime('%d/%m/%Y')}", "FMT_FULL"

        # --- 4. Logic: Single Date ---
        if is_strict_full_date(t):
            dt = dateparser.parse(t)
            if dt:
                return dt.strftime('%d/%m/%Y'), "FMT_FULL"

        return None, None

    def extract_month_year(self, text):
        """
        Standalone specialist for Month-Year (e.g., "March 1990" or "Jan 2000 – Feb 2005").
        Outputs: "mm/yyyy" or "mm/yyyy-mm/yyyy"
        """
        if not text or pd.isna(text):
            return None, None

        def is_month_year_only(s):
            """Checks for Month + Year, but strictly NO Day digits."""
            s_low = str(s).lower()
            has_year = bool(re.search(self.RE_YEAR, s_low))
            has_month = bool(re.search(self.RE_MONTH, s_low))

            # Remove the year digits so we don't confuse them with a Day
            no_year = re.sub(self.RE_YEAR, ' ', s_low)
            # Look for any 1-2 digit numbers that could be a Day
            days = re.findall(r'\b\d{1,2}\b', no_year)
            has_day = any(1 <= int(d) <= 31 for d in days)

            # We only return True if it has Month/Year AND no Day
            return has_year and has_month and not has_day

        # --- 1. Cleanup & Standardize ---
        t = str(text).strip()
        t = re.sub(r'\[.*?\]|†|\*', '', t)
        t = re.sub(r'\s*—\s*|\s+to\s+|\s+-\s+', '–', t)

        # --- 2. Resolve Redundant Brackets ---
        if '(' in t:
            parts = re.split(r'\(|\)', t)
            prefix = parts[0].strip()
            bracket = parts[1].strip() if len(parts) > 1 else ""

            if is_month_year_only(prefix):
                t = prefix
            elif is_month_year_only(bracket):
                t = bracket

        # --- 3. Parsing Logic ---
        # Case A: Span (mm/yyyy-mm/yyyy)
        if '–' in t:
            p_start, p_end = [p.strip() for p in t.split('–', 1)]

            if is_month_year_only(p_start) and is_month_year_only(p_end):
                dt_s = dateparser.parse(p_start)
                dt_e = dateparser.parse(p_end)
                if dt_s and dt_e:
                    span_str = f"{dt_s.strftime('%m/%Y')}-{dt_e.strftime('%m/%Y')}"
                    return span_str, "FMT_MONTH_YEAR"

        # Case B: Single Date (mm/yyyy)
        if is_month_year_only(t):
            dt = dateparser.parse(t)
            if dt:
                return dt.strftime('%m/%Y'), "FMT_MONTH_YEAR"

        return None, None

    def extract_year(self, text):
        """
        Extracts the global temporal boundaries from a string of multiple spans.
        Example: "1969–1975 1980–1981" -> "1969-1981"
        Example: "2001–2002 2018–present" -> "2001-present"
        """
        if not text or pd.isna(text):
            return None, None

        # --- 1. Cleanup ---
        t = str(text).strip()
        t = re.sub(r'\[.*?\]|†|\*', '', t)
        # Standardize all potential separators to a simple space to avoid split issues
        t_clean = re.sub(r'\s*—\s*|\s+to\s+|\s+-\s+|–', ' ', t)

        # --- 2. Yield to Month-Year or Full Date ---
        if re.search(self.RE_MONTH, t.lower()):
            return None, None

        # --- 3. Extract All Year Candidates ---
        # Find all 3-4 digit numbers
        years = re.findall(r'\b\d{3,4}\b', t_clean)

        # Check for "present/now/current"
        has_present = bool(re.search(self.RE_PRESENT, t, re.IGNORECASE))

        if not years:
            return None, None

        # Convert to integers to find the true min/max
        years_int = [int(y) for y in years]
        start_year = str(min(years_int))

        # --- 4. Logic: Determine the End Point ---
        if has_present:
            # If "present" exists anywhere in the string, it is the ultimate end point
            return f"{start_year}-present", "FMT_YEAR"

        elif len(years_int) > 1 and any(op in t for op in ['–', '-', '—', ' to ', '/']):
            # If no "present", but multiple years, connect the min and the max
            end_year = str(max(years_int))
            # Handle the "1994-95" shorthand case if the last year was extracted as "95"
            # Note: Our findall \b\d{3,4}\b ignores 2-digit years.
            # If you need to handle 1994-95, we use a different regex:
            all_nums = re.findall(r'\b\d{1,4}\b', t_clean)
            if all_nums:
                start_yr = all_nums[0]
                end_yr_raw = all_nums[-1]

                # Use your existing reconstructor logic here
                def reconstruct(s, e):
                    if len(e) >= len(s): return e
                    return s[:-len(e)] + e

                final_start = min([int(reconstruct(start_yr, x)) for x in all_nums if len(x) >= 3])
                final_end = max([int(reconstruct(start_yr, x)) for x in all_nums])
                return f"{final_start}-{final_end}", "FMT_YEAR"

        else:
            # Single year case
            return str(years_int[0]), "FMT_YEAR"

        return None, None

    def extract_decade(self, text):
        """
        Standalone specialist for Decades.
        Handles: "1990s", "1970s–80s", "1970s–1980s", "mid-1990s".
        Outputs: "yyyy-yyyy"
        """
        if not text or pd.isna(text):
            return None, None

        def get_decade_span(decade_start_year):
            """Helper to turn 1990 into 1990-1999"""
            start = int(decade_start_year)
            return f"{start}-{start + 9}"

        # --- 1. Cleanup ---
        t = str(text).strip()
        t = re.sub(r'\[.*?\]|†|\*', '', t)
        t = re.sub(r'\s*—\s*|\s+to\s+|\s+-\s+', '–', t)

        # --- 2. Logic: Decade Spans (e.g., 1970s–1980s) ---
        if '–' in t:
            parts = [p.strip() for p in t.split('–', 1)]
            matches = [re.search(self.RE_DECADE, p) for p in parts]

            if all(matches):
                y1 = int(matches[0].group(1))
                y2 = int(matches[1].group(1))
                # Logic: 1970s (start 1970) to 1980s (end 1989)
                return f"{y1}-{y2 + 9}", "FMT_DECADE"

            # Handle shorthand spans like 1970s–80s
            elif matches[0] and re.search(r'\b(\d{1,2})s\b', parts[1]):
                y1 = int(matches[0].group(1))
                y2_short = re.search(r'\b(\d{1,2})s\b', parts[1]).group(1)
                # Reconstruct 1970 + 80 -> 1980
                y2_full = int(str(y1)[:4 - len(y2_short + '0')] + y2_short + '0')
                return f"{y1}-{y2_full + 9}", "FMT_DECADE"

        # --- 3. Logic: Single Decade (e.g., 1990s) ---
        match = re.search(self.RE_DECADE, t)
        if match:
            return get_decade_span(match.group(1)), "FMT_DECADE"

        return None, None

    def extract_century(self, text):
        """
        Standalone specialist for Centuries and Century Spans.
        Handles: "13th century", "9th century – 15th century", "9th–15th centuries".
        Outputs: "yyyy-yyyy"
        """
        if not text or pd.isna(text):
            return None, None

        def get_century_span(ordinal_num):
            """Converts '13' to ('1201', '1300')"""
            num = int(ordinal_num)
            end = num * 100
            start = end - 99
            return f"{start:04d}", f"{end:04d}"

        # --- 1. Cleanup & Standardize ---
        t = str(text).strip().lower()
        t = re.sub(r'\[.*?\]|†|\*', '', t)
        t = re.sub(r'\s*—\s*|\s+to\s+|\s+-\s+', '–', t)

        # --- 2. Logic: Century Spans ---
        # Case: "9th–15th centuries" or "9th century – 15th century"
        if '–' in t:
            parts = [p.strip() for p in t.split('–', 1)]
            m_start = re.search(self.RE_ORDINAL, parts[0])
            m_end = re.search(self.RE_ORDINAL, parts[1])

            if m_start and m_end:
                start_y1, _ = get_century_span(m_start.group(1))
                _, end_y2 = get_century_span(m_end.group(1))

                # BC Logic (if 'bc' appears anywhere in the string)
                if 'bc' in t:
                    # BC goes backwards: 5th-1st BC -> -0500 to -0001
                    return f"-{end_y2}--{start_y1}", "FMT_CENTURY"

                return f"{start_y1}-{end_y2}", "FMT_CENTURY"

        # --- 3. Logic: Single Century ---
        match = re.search(self.RE_ORDINAL, t)
        if match and re.search(self.RE_CENT_KW, t):
            s, e = get_century_span(match.group(1))
            if 'bc' in t:
                return f"-{e}--{s}", "FMT_CENTURY"
            return f"{s}-{e}", "FMT_CENTURY"

        return None, None

    def extract_day_month(self, text):
        if not text or pd.isna(text):
            return None, None

        def is_day_month_only(s):
            s_low = str(s).lower()
            if re.search(self.RE_YEAR, s_low): return False
            has_month = bool(re.search(self.RE_MONTH, s_low))
            days = re.findall(r'\b\d{1,2}(?:st|nd|rd|th)?\b', s_low)
            has_day = any(1 <= int(re.sub(r'\D', '', d)) <= 31 for d in days)
            is_numeric_dm = bool(re.match(r'^\d{1,2}[/\-]\d{1,2}$', s_low.strip()))
            return (has_month and has_day) or is_numeric_dm

        # --- 1. Cleanup ---
        t = str(text).strip()
        t = re.sub(r'\[.*?\]|†|\*', '', t)
        t = re.sub(r'\s*—\s*|\s+to\s+|\s+-\s+', '–', t)

        # --- 2. Logic: Advanced Spans ---
        if '–' in t:
            p_start, p_end = [p.strip() for p in t.split('–', 1)]

            # Check for Month Day1–Day2 (e.g., March 9–10)
            # If p_end is just a digit, and p_start has a month
            if p_end.isdigit() and is_day_month_only(p_start):
                month_match = re.search(self.RE_MONTH, p_start.lower())
                if month_match:
                    month_name = month_match.group(0)
                    p_end = f"{month_name} {p_end}"  # Reconstruct as "March 10"

            if is_day_month_only(p_start) and is_day_month_only(p_end):
                dt_s = dateparser.parse(p_start)
                dt_e = dateparser.parse(p_end)
                if dt_s and dt_e:
                    return f"{dt_s.strftime('%d/%m')}-{dt_e.strftime('%d/%m')}", "FMT_DAY_MONTH"

        # --- 3. Logic: Single ---
        if is_day_month_only(t):
            dt = dateparser.parse(t)
            if dt:
                return dt.strftime('%d/%m'), "FMT_DAY_MONTH"

        return None, None

    def extract_placeholder(self, text):
        """
        Detects non-data placeholders like TBD, TBA, Unknown.
        Returns: (None, "FMT_NONE") if a match is found.
        """
        if not text or pd.isna(text):
            return None, "FMT_NONE"

        t = str(text).strip().lower()

        if re.search(self.RE_PLACEHOLDERS, t):
            return None, "FMT_NONE"

        return None, None

    # ---------------------------------------------------------
    # Pipeline:
    # ---------------------------------------------------------
    def _run_waterfall(self, text):
        """Runs text through specialists and returns (value, dtype)."""
        for specialist in self.specialists:
            val, dtype = specialist(text)
            if dtype is not None:
                return val, dtype
        return text, 'Other'  # If it fails all, it is 'Other'

    def _convert_format(self, parsed_val, current_dtype, target_dtype):
        """
        Converts a standardized parsed value between temporal formats.
        Specific -> General: Truncates (e.g., 15/05/1990 -> 1990)
        General -> Specific: Pads with '01' (e.g., 1990 -> 01/01/1990)
        """
        if not parsed_val or pd.isna(parsed_val) or current_dtype == 'Other':
            # If we couldn't parse it originally, we cannot mathematically convert it.
            return None

        # 1. Consolidate Decade/Century to Year for conversion mapping
        c_type = 'FMT_YEAR' if current_dtype in ['FMT_DECADE', 'FMT_CENTURY'] else current_dtype
        t_type = 'FMT_YEAR' if target_dtype in ['FMT_DECADE', 'FMT_CENTURY'] else target_dtype

        if c_type == t_type:
            return parsed_val

        # 2. Helper to cleanly convert a single standardized date string
        def convert_single(v, c, t):
            if c == 'FMT_FULL':  # Shape: dd/mm/yyyy
                parts = v.split('/')
                if len(parts) != 3: return None
                if t == 'FMT_MONTH_YEAR': return f"{parts[1]}/{parts[2]}"
                if t == 'FMT_YEAR':       return parts[2]
                if t == 'FMT_DAY_MONTH':  return f"{parts[0]}/{parts[1]}"

            elif c == 'FMT_MONTH_YEAR':  # Shape: mm/yyyy
                parts = v.split('/')
                if len(parts) != 2: return None
                if t == 'FMT_FULL':       return f"01/{parts[0]}/{parts[1]}"
                if t == 'FMT_YEAR':       return parts[1]
                if t == 'FMT_DAY_MONTH':  return f"01/{parts[0]}"

            elif c == 'FMT_YEAR':  # Shape: yyyy (or -yyyy for BC)
                if t == 'FMT_FULL':       return f"01/01/{v}"
                if t == 'FMT_MONTH_YEAR': return f"01/{v}"
                if t == 'FMT_DAY_MONTH':  return "01/01"

            elif c == 'FMT_DAY_MONTH':  # Shape: dd/mm
                # We cannot safely invent a year out of thin air to up-sample.
                return None

            return None

        # 3. Handle Spans cleanly
        # Matches a dash preceded by a digit, preventing breaks on BC minus signs (e.g., -0500--0401)
        match = re.search(r'(?<=\d)-(?=\d|-)', parsed_val)

        try:
            if match:
                idx = match.start()
                start_val = parsed_val[:idx]
                end_val = parsed_val[idx + 1:]

                new_start = convert_single(start_val, c_type, t_type)
                new_end = convert_single(end_val, c_type, t_type)

                if new_start and new_end:
                    return f"{new_start}-{new_end}"
                return None
            else:
                return convert_single(parsed_val, c_type, t_type)
        except Exception:
            return None

    def process_and_normalize_tables(self, tables, verbose=True):
        """
        Parses tables, discards noisy ones (>30% Other), and normalizes valid tables
        to their most common temporal format.
        """
        valid_tables = []
        discarded_count = 0
        total_rows_processed = 0

        for table in tables:
            rows = table.get('Relevant table data', [])
            if not rows:
                continue

            # Extract the name of the target date column (assuming it's the last column)
            target_col_name = list(rows[0].keys())[-1]
            parsed_col_name = f"{target_col_name} (parsed)"

            table_results = []
            dtypes_found = []

            # --- STEP 1: Initial Parse ---
            for row in rows:
                total_rows_processed += 1
                raw_val = row.get(target_col_name, "")

                parsed_val, dtype = self._run_waterfall(raw_val)
                table_results.append({
                    'row_ref': row,
                    'raw_val': raw_val,
                    'parsed_val': parsed_val,
                    'dtype': dtype
                })
                dtypes_found.append(dtype)

            # --- STEP 2: Discard >30% 'Other' ---
            total_table_rows = len(rows)
            other_count = dtypes_found.count('Other')

            if total_table_rows > 0 and (other_count / total_table_rows) > 0.3:
                discarded_count += 1
                continue  # Skip to the next table entirely

            # --- STEP 3: Find Majority Format ---
            # Exclude 'FMT_NONE' and 'Other' from the vote
            valid_votes = [d for d in dtypes_found if d not in ['FMT_NONE', 'Other']]

            if valid_votes:
                # Get the most common type (e.g., 'FMT_YEAR')
                # Map Decade and Century votes to Year before counting
                mapped_votes = ['FMT_YEAR' if v in ['FMT_DECADE', 'FMT_CENTURY'] else v for v in valid_votes]
                target_format = Counter(mapped_votes).most_common(1)[0][0]
            else:
                target_format = 'FMT_NONE'

            # --- STEP 3.5: Span Truncation Check ---
            # If the majority format is years, check if single years dominate spans
            truncate_spans = False
            if target_format == 'FMT_YEAR':
                # Count parsed values that are pure years vs spans
                single_years = sum(1 for item in table_results if
                                   item['dtype'] == 'FMT_YEAR' and '-' not in str(item['parsed_val']))
                spans = sum(
                    1 for item in table_results if item['dtype'] == 'FMT_YEAR' and '-' in str(item['parsed_val']))

                # If pure years are the majority over spans, trigger the flag
                if single_years > spans > 0:
                    truncate_spans = True

            # --- STEP 4: Convert and Mutate ---
            for item in table_results:
                row = item['row_ref']
                curr_dtype = item['dtype']

                # Rule A, B, and C as you have them:
                if curr_dtype == 'FMT_NONE':
                    final_val = None
                elif curr_dtype == target_format:
                    final_val = item['parsed_val']
                else:
                    final_val = self._convert_format(
                        parsed_val=item['parsed_val'],
                        current_dtype=curr_dtype,
                        target_dtype=target_format
                    )

                # --- NEW TRUNCATION EXECUTION ---
                # If our condition was met and this specific final value is a span
                if truncate_spans and final_val and '-' in str(final_val):
                    # Use your existing regex to safely split around the span dash
                    # (This prevents accidentally splitting negative/BC years!)
                    match = re.search(r'(?<=\d)-(?=\d|-)', str(final_val))
                    if match:
                        # Extract only the start year
                        final_val = final_val[:match.start()]

                # Add the new column to the dictionary
                row[parsed_col_name] = final_val

            valid_tables.append(table)

        # --- STEP 5: Summary ---
        if verbose:
            print("\n" + "=" * 50)
            print(f"{'NORMALIZATION PIPELINE SUMMARY':^50}")
            print("=" * 50)
            print(f"Total Tables Processed: {len(tables)}")
            print(f"Tables Discarded:       {discarded_count} (>30% 'Other')")
            print(f"Tables Retained:        {len(valid_tables)}")
            print(f"Total Rows Evaluated:   {total_rows_processed}")
            print("=" * 50)

        return valid_tables