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


import pandas as pd
import re
import dateparser
from datetime import datetime
from collections import Counter
from IPython.display import display


class TemporalTableParser:
    def __init__(self):
        self.stats = Counter({'total': 0, 'discarded_parse_failure': 0, 'discarded_too_few': 0, 'valid': 0})
        self.format_dist = Counter()

        # Standard format names
        self.FMT_YEAR = "year"
        self.FMT_MONTH_YEAR = "month_year"
        self.FMT_FULL = "full_date"

        # Shared Regex patterns
        self.RE_YEAR = r'\b\d{3,4}\b'
        self.RE_MONTH = r'jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec'
        self.RE_CENTURY = r'(\d{1,2})(?:st|nd|rd|th)\s*c'
        self.RE_DECADE = r'\d{3}0s'

    # --- FORMAT DETECTORS (MODULAR) ---

    def _is_full_date(self, text):
        """Checks for Day + Month + Year."""
        t = text.lower()
        has_year = bool(re.search(self.RE_YEAR, t))
        has_month = bool(re.search(self.RE_MONTH, t))
        # Remove the year digits first so we don't mistake 2024 for a "Day"
        no_year_text = re.sub(self.RE_YEAR, '', t)
        has_day = bool(re.search(r'\b\d{1,2}(?![0-9])', no_year_text))

        return has_year and has_month and has_day

    def _is_month_year(self, text):
        """Checks for Month + Year (and ensures NO Day)."""
        t = text.lower()
        has_year = bool(re.search(self.RE_YEAR, t))
        has_month = bool(re.search(self.RE_MONTH, t))

        # If it has a day, it's actually a Full Date, not just Month-Year
        if self._is_full_date(text):
            return False

        return has_year and has_month

    def _is_year(self, text):
        """Checks for Year, Decade, or Century (and ensures NO Month/Day)."""
        t = text.lower()

        # Special cases that always count as 'year' granularity
        if re.search(self.RE_CENTURY, t) or re.search(self.RE_DECADE, t):
            return True

        has_year = bool(re.search(self.RE_YEAR, t))
        has_month = bool(re.search(self.RE_MONTH, t))

        # If it has a month, it's a more specific format
        if has_month:
            return False

        return has_year

    def _get_century_span(self, text):
        """
        Returns integers (start_year, end_year).
        9th century BC -> (-900, -801)
        9th-13th century -> (801, 1300)
        """
        t = text.lower()
        matches = list(re.finditer(self.RE_CENTURY, t))
        if not matches: return None, None

        found_years = []
        # If "BC" appears anywhere in the string, we need to check if it applies
        # to specific centuries or the whole span.
        for m in matches:
            century_num = int(m.group(1))
            # Look for 'bc' within 15 chars after the century mention or at the end
            is_bc = 'bc' in t[m.end():m.end() + 15] or 'bc' in t[-5:]

            if is_bc:
                # 9th c BC: -900 to -801
                start = -(century_num * 100)
                end = -((century_num - 1) * 100 + 1)
            else:
                # 9th c AD: 801 to 900
                start = (century_num - 1) * 100 + 1
                end = century_num * 100
            found_years.extend([start, end])

        return min(found_years), max(found_years)

    def _detect_format(self, text):
        """The 'Traffic Controller' - assigns the correct format enum."""
        if not text: return None

        # Check from most specific to least specific
        if self._is_full_date(text):
            return self.FMT_FULL
        if self._is_month_year(text):
            return self.FMT_MONTH_YEAR
        if self._is_year(text):
            return self.FMT_YEAR

        return None

    def _extract_year_integer(self, text):
        """
        Safely extracts a year as an integer, handling BC.
        Example: "9th century BC" -> -900 (via _get_century_span)
        Example: "1994" -> 1994
        """
        if text is None: return None
        t = str(text).lower()

        # Pull the first standalone number (1 to 4 digits)
        match = re.search(r'\b\d{1,4}\b', t)
        if not match:
            return None

        year_val = int(match.group(0))

        # Apply negative sign for BC
        if 'bc' in t:
            year_val = -year_val

        return year_val

    # --- UTILITIES & NORMALIZERS ---

    def clean_wiki_text(self, text):
        if pd.isna(text): return ""
        text = str(text).strip()

        # 1. REMOVE citations [1][2] (These never contain useful dates)
        text = re.sub(r'\[.*?\]', '', text)

        # 2. UNWRAP parentheses instead of deleting them
        # This turns "Charlemagne (742–814)" into "Charlemagne 742–814"
        text = text.replace('(', ' ').replace(')', ' ')

        # 3. Strip specific non-date symbols
        text = re.sub(r'†|\*', '', text)

        # 4. Standardize spans (to, —, -) into the En-Dash (–)
        text = re.sub(r'\s*—\s*|\s+to\s+|\s+-\s+', '–', text)
        text = re.sub(r'(?<=\d)-(?=\d)', '–', text)

        # 5. Filter for high-signal words only
        keys = rf'{self.RE_MONTH}|present|current|now|ongoing|c|bc|ad'
        words = text.split()
        cleaned = [w for w in words if re.search(r'\d', w) or re.search(keys, w.lower()) or '–' in w]

        return ' '.join(cleaned).strip(',. ')

    def _normalize_value(self, val, target_format):
        if val is None: return "None"

        # If it's an integer (Year/Century)
        if isinstance(val, (int, float)):
            abs_yr = abs(int(val))
            suffix = " BC" if val < 0 else ""
            return f"{abs_yr}{suffix}"

        # If it's a datetime object (Month/Full Date)
        if isinstance(val, datetime):
            if target_format == self.FMT_FULL:
                return val.strftime('%B %d, %Y')
            elif target_format == self.FMT_MONTH_YEAR:
                return val.strftime('%B %Y')
            return val.strftime('%Y')

        return str(val)

    def _extract_span_endpoints(self, cleaned_text):
        if '–' not in cleaned_text:
            return cleaned_text, cleaned_text

        parts = cleaned_text.split('–')
        s_str, e_str = parts[0].strip(), parts[-1].strip()

        if re.fullmatch(r'\d{4}', s_str) and re.fullmatch(r'\d{2}', e_str):
            e_str = s_str[:2] + e_str
        if re.match(r'^\d{1,2}$', e_str):
            month_match = re.search(self.RE_MONTH, s_str.lower())
            if month_match:
                e_str = f"{month_match.group(0).capitalize()} {e_str}"
        return s_str, e_str

    def _format_final_span(self, start_norm, end_norm):
        """
        Post-processor: Collapses identical spans into a single value.
        Example: ('1996', '1996') -> '1996'
        Example: ('1996', '2000') -> '1996–2000'
        """
        if start_norm == "None":
            return "None"

        if start_norm == end_norm:
            return start_norm

        return f"{start_norm}–{end_norm}"

    # --- PIPELINE ---

    def parse_table_temporal_data(self, sample):
        df = pd.DataFrame(sample.get('Relevant table data', []))
        if df.empty: return []

        self.stats['total'] += 1
        target_col = list(df.columns)[1] if len(df.columns) > 1 else df.columns[0]
        ent_col = list(df.columns)[0]

        interim_rows = []
        for _, row in df.iterrows():
            raw_val = row[target_col]
            cleaned = self.clean_wiki_text(raw_val)

            # 1. Handle Empty/None
            if not cleaned or cleaned.lower() in ['tba', 'tbd', 'unknown', 'none', '-', '_']:
                interim_rows.append({'ent': row[ent_col], 'raw': raw_val, 'type': 'none', 's_val': None, 'e_val': None})
                continue

            # 2. Extract Integers for Centuries (Returns raw years like -900, 801)
            s_val, e_val = self._get_century_span(cleaned)
            if s_val is not None:
                interim_rows.append(
                    {'ent': row[ent_col], 'raw': raw_val, 'type': self.FMT_YEAR, 's_val': s_val, 'e_val': e_val})
                continue

            # 3. Detect Format for non-centuries
            s_str, e_str = self._extract_span_endpoints(cleaned)
            detected_fmt = self._detect_format(s_str)

            if detected_fmt is None:
                interim_rows.append({'ent': row[ent_col], 'raw': raw_val, 'type': 'none', 's_val': None, 'e_val': None})
                continue

            # 4. Handle Year format as pure Integers
            if detected_fmt == self.FMT_YEAR:
                s_val = self._extract_year_integer(s_str)
                e_val = self._extract_year_integer(e_str)

            # 5. Handle Month/Full as Datetimes (but we extract year for the sort key)
            else:
                now = datetime.now()
                s_dt = now if s_str.lower() in ['present', 'current', 'now'] else dateparser.parse(s_str, settings={
                    'PREFER_DAY_OF_MONTH': 'first'})
                e_dt = now if e_str.lower() in ['present', 'current', 'now'] else dateparser.parse(e_str, settings={
                    'PREFER_DAY_OF_MONTH': 'first'})

                if s_dt is None or e_dt is None:
                    self.stats['discarded_parse_failure'] += 1
                    return None

                # Store the full DT object in s_val so we don't lose the month/day data
                s_val, e_val = s_dt, e_dt

            interim_rows.append(
                {'ent': row[ent_col], 'raw': raw_val, 'type': detected_fmt, 's_val': s_val, 'e_val': e_val})

        # --- Table Validation ---
        valid_types = [r['type'] for r in interim_rows if r['type'] != 'none']
        if len(valid_types) <= 2:
            self.stats['discarded_too_few'] += 1
            return None

        dominant_fmt = Counter(valid_types).most_common(1)[0][0]
        self.format_dist[dominant_fmt] += 1
        self.stats['valid'] += 1

        # --- Finalization & Integer Sorting ---
        final_rows = []
        for r in interim_rows:
            # Normalize the display strings using the dominant format
            s_norm = self._normalize_value(r['s_val'], dominant_fmt)
            e_norm = self._normalize_value(r['e_val'], dominant_fmt)

            # Determine the Sort Key (The Year Integer)
            if isinstance(r['s_val'], datetime):
                sort_key = r['s_val'].year
            elif isinstance(r['s_val'], (int, float)):
                sort_key = r['s_val']
            else:
                sort_key = float('inf')  # Push "None" to the end

            final_rows.append({
                ent_col: r['ent'],
                target_col: r['raw'],
                f"{target_col} (parsed)": self._format_final_span(s_norm, e_norm),
                "_sort_key": sort_key
            })

        final_rows.sort(key=lambda x: x['_sort_key'])
        for f in final_rows: del f['_sort_key']
        return final_rows

    def get_summary_report(self):
        """Returns a string summary of the parsing run using the refactored stats."""
        total = self.stats['total']
        if total == 0:
            return "No tables processed."

        discarded_parse = self.stats['discarded_parse_failure']
        discarded_count = self.stats['discarded_too_few']
        total_discarded = discarded_parse + discarded_count
        valid_tables = self.stats['valid']

        report = []
        report.append("=" * 45)
        report.append("       TEMPORAL PARSER SUMMARY REPORT")
        report.append("=" * 45)
        report.append(f"Total Tables Analyzed:       {total}")
        report.append(f"Valid Tables Normalized:     {valid_tables}")
        report.append(f"Total Tables Discarded:      {total_discarded}")

        if total_discarded > 0:
            report.append("-" * 45)
            report.append("Discard Reasons:")
            report.append(f"  - Logic/Parse Failures:    {discarded_parse}")
            report.append(f"  - Insufficient Date Rows:  {discarded_count}")

        if valid_tables > 0:
            report.append("-" * 45)
            report.append("Dominant Format Distribution (Valid Tables):")
            for fmt, count in self.format_dist.most_common():
                pct = (count / valid_tables) * 100
                report.append(f"  - {fmt:15}: {count:3} ({pct:5.1f}%)")

        report.append("=" * 45)

        return "\n".join(report)