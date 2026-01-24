import time
import requests
from tqdm import tqdm


class WikiHarvester:
    BASE_URL = "https://en.wikipedia.org/w/api.php"
    DEFAULT_KEYWORDS = [
        "list of", "list", "comparison of", "rankings of",
        "discography of", "filmography of", "demographics of",
        "list of countries by", "list of states by", "list of players",
        "list of cities by", "list of companies by", "list of largest", "list of most",
        "members", "teams", "countries", "cities", "species", "films",
        "awards", "companies", "airlines", "universities", "planets", "missions"
    ]

    def __init__(self, user_agent="Research/0.1", delay=0.1):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.delay = delay
        self.results_metadata = {}

    def fetch_titles(self, keywords=None):
        """Phase 1: Harvest candidate titles."""
        target_keywords = keywords or self.DEFAULT_KEYWORDS
        all_titles = set()

        print(f"Starting Phase 1: Harvesting titles for {len(target_keywords)} keywords...")

        for kw in tqdm(target_keywords, desc="Harvesting", unit="keyword"):
            kw_titles_count = 0
            continue_token = None
            while True:
                params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": f'intitle:"{kw}"',
                    "srnamespace": 0,
                    "srlimit": 500,
                    "format": "json"
                }
                if continue_token:
                    params["sroffset"] = continue_token

                try:
                    response = self.session.get(self.BASE_URL, params=params)
                    response.raise_for_status()
                    data = response.json()
                except:
                    break

                batch = [x["title"] for x in data.get("query", {}).get("search", [])]
                all_titles.update(batch)
                kw_titles_count += len(batch)

                if "continue" in data:
                    continue_token = data["continue"]["sroffset"]
                    time.sleep(self.delay)
                else:
                    break

            self.results_metadata[kw] = kw_titles_count

        self._print_summary(len(all_titles))
        return list(all_titles)

    def filter_pages_with_tables(self, titles):
        """
        Phase 2: Filter out pages that do not contain a 'wikitable'.
        Uses batching (50 titles per request) for efficiency.
        """
        print(f"Starting Phase 2: Filtering {len(titles)} pages for tables...")
        valid_titles = []

        # Split titles into chunks of 50
        chunks = [titles[i:i + 50] for i in range(0, len(titles), 50)]

        for chunk in tqdm(chunks, desc="Filtering", unit="batch"):
            params = {
                "action": "query",
                "prop": "revisions",
                "rvprop": "content",
                "titles": "|".join(chunk),
                "format": "json",
                "formatversion": 2  # Makes response easier to parse
            }

            try:
                response = self.session.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()

                pages = data.get("query", {}).get("pages", [])
                for page in pages:
                    if "revisions" in page:
                        content = page["revisions"][0].get("content", "")
                        # Look for the standard wikitext table class
                        if 'class="wikitable' in content or 'class=\'wikitable' in content or '{|' in content:
                            valid_titles.append(page["title"])

            except Exception as e:
                print(f"\n[Error] Batch failed: {e}")
                continue

            time.sleep(self.delay)

        print(f"\nPhase 2 Complete: Kept {len(valid_titles)} pages out of {len(titles)}.")
        return valid_titles

    def _print_summary(self, total_unique):
        print("\n" + "=" * 34)
        print(f"{'Keyword':<20} | {'Pages Found':<10}")
        print("-" * 34)
        for kw, count in self.results_metadata.items():
            print(f"{kw:<20} | {count:<10}")
        print("-" * 34)
        print(f"{'TOTAL UNIQUE':<20} | {total_unique:<10}")
        print("=" * 34 + "\n")
