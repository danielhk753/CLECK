"""
WikiHarvester: A Resilient Wikipedia Data Extraction Module

This module provides the `WikiHarvester` class, designed for large-scale discovery
and filtering of Wikipedia pages containing data tables. It is optimized for
harvesting >100,000 titles while staying within API rate limits.

Key Features:
    * Dual-Phase Harvesting: Combines 'Prefix' searches (high volume) with
      'Inline Search' (high precision) to maximize coverage.
    * Resilient Networking: Implements exponential backoff for 429 (Rate Limit)
      errors and retry logic for network blips.
    * Asynchronous Filtering: Uses `httpx` and `asyncio` with semaphores to
      check page content 10x faster than standard requests.
    * Resource Management: Employs 'Super-Batching' to reset TCP connections
      and clear memory every 5,000 pages, preventing silent 85% crashes.
    * Responsive UI: Includes a CSS Grid-based HTML summary for viewing
      thousands of results efficiently in Jupyter Notebooks.

Usage:
    harvester = WikiHarvester(user_agent="MyBot/1.0 (contact@email.com)")
    harvester.run_complete_harvest()
    await harvester.filter_pages_async()

Requirements:
    requests, httpx, asyncio, pandas, numpy, tqdm
"""

import time
import httpx
import asyncio
import requests
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display, HTML

DEFAULT_KEYWORDS = [
    "list", "list of", "members", "teams", "countries", "cities", "species", "films",
    "awards", "companies", "airlines", "universities", "planets", "missions", "players", "athletes"
]

DEFAULT_PREFIXES = [
    # --- GEOGRAPHY & PLACES (The highest volume) ---
    "List of cities", "List of countries", "List of towns", "List of villages", "List of districts",
    "List of municipalities", "List of communes", "List of counties", "List of provinces",
    "List of territories", "List of islands", "List of mountains", "List of rivers", "List of lakes",
    "List of waterfalls", "List of volcanoes", "List of caves", "List of deserts",
    "List of beaches", "List of forests", "List of parks", "List of reservoirs",

    # --- INFRASTRUCTURE & BUILDINGS ---
    "List of airports", "List of stations", "List of ports", "List of bridges",
    "List of tunnels", "List of highways", "List of roads", "List of stadiums",
    "List of skyscrapers", "List of museums", "List of libraries", "List of hospitals",
    "List of schools", "List of universities", "List of colleges", "List of banks",
    "List of churches", "List of cathedrals", "List of mosques", "List of temples",
    "List of synagogues", "List of castles", "List of palaces", "List of dams",
    "List of power", "List of lighthouses", "List of monuments", "List of malls",

    # --- PEOPLE & ROLES (High Link Density) ---
    "List of presidents", "List of mayors", "List of governors", "List of monarchs",
    "List of kings", "List of queens", "List of senators", "List of members",
    "List of ambassadors", "List of ministers", "List of bishops", "List of popes",
    "List of actors", "List of directors", "List of writers", "List of poets",
    "List of painters", "List of musicians", "List of composers", "List of singers",
    "List of athletes", "List of players", "List of olympic", "List of billionaires",
    "List of philosophers", "List of scientists", "List of inventors", "List of alumni",

    # --- ORGANIZATIONS & COMPANIES ---
    "List of companies", "List of airlines", "List of manufacturers", "List of brands",
    "List of schools", "List of newspapers", "List of magazines", "List of journals",
    "List of organizations", "List of agencies", "List of nonprofits", "List of retailers",

    # --- SCIENCE & TAXONOMY ---
    "List of species", "List of mammals", "List of birds", "List of reptiles",
    "List of fish", "List of insects", "List of plants", "List of trees",
    "List of fungi", "List of minerals", "List of elements", "List of isotopes",
    "List of stars", "List of galaxies", "List of asteroids", "List of comets",

    # --- MEDIA & CULTURE ---
    "List of films", "List of series", "List of episodes", "List of characters",
    "List of books", "List of albums", "List of singles", "List of games",
    "List of festivals", "List of awards", "List of paintings", "List of buildings",

    # --- EVENTS & HISTORY ---
    "List of battles", "List of wars", "List of treaties", "List of riots",
    "List of protests", "List of massacres", "List of accidents", "List of earthquakes",
    "List of shipwrecks",

    # Members & Groups
    "Members of ", "Fellows of ", "Alumni of ", "Players of ", "Roster of ",
    "Winners of ", "Recipients of ", "Inductees of ", "Signatories of ",

    # Geography & Places
    "Cities in ", "Towns in ", "Villages in ", "Communes in ", "Districts of ",
    "Provinces of ", "Counties in ", "Islands of ", "Mountains in ", "Rivers of ",
    "Lakes of ", "Bridges in ", "Airports in ", "Stations in ", "Museums in ",
    "Hospitals in ", "Schools in ", "Universities in ", "Parks in ", "Dams in ",

    # Countries & Comparisons
    "Countries by ", "States by ", "Ranking of ", "Comparison of ", "Index of ",
    "Statistics of ", "Demographics of ", "Economy of ", "Politics of ",

    # People & Occupations
    "Presidents of ", "Mayors of ", "Governors of ", "Monarchs of ", "Kings of ",
    "Ambassadors of ", "Ministers of ", "People from ", "Artists from ",
    "Actors in ", "Directors of ", "Writers of ", "Scientists of ",

    # Companies & Organizations
    "Companies of ", "Airlines of ", "Banks in ", "Newspapers in ",
    "Organizations based in ", "Manufacturers of ", "Retailers of ",

    # Science & Species
    "Species of ", "Genera of ", "Birds of ", "Mammals of ", "Fish of ",
    "Plants of ", "Insects of ", "Minerals of ", "Elements in ", "Stars in ",

    # Media & Arts
    "Films of ", "Series of ", "Episodes of ", "Characters in ", "Books by ",
    "Albums by ", "Songs by ", "Singles by ", "Games in ", "Paintings by ",
    "Awards for ", "Festivals in ", "Buildings in ", "Historic sites in ",

    # History & Events
    "Battles of ", "Wars in ", "Treaties of ", "Events in ",
    "Shipwrecks of ", "Disasters in ", "Elections in ", "Protests in "
]

class WikiHarvester:
    BASE_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(self, user_agent="Research/0.1", delay=0.05):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

        self.delay = delay

        self.all_titles = set()
        self.valid_titles = []
        self.metadata = []

    def _safe_request(self, params, verbose=None):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.session.get(self.BASE_URL, params=params, timeout=15)

                # 1. Handle Rate Limiting specifically
                if response.status_code == 429:
                    wait_time = (attempt + 1) * 10  # Increasing wait: 10s, 20s, 30s...
                    if verbose:
                        print(f"⚠️ Rate limited (429). Sleeping {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                # 2. Raise for other errors (404, 500, etc.)
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    if verbose:
                        print(f"\n❌ Request failed after {max_retries} attempts: {e}")
                    return None

                wait = (attempt + 1) * 2
                if verbose:
                    print(f"\n🔄 Connection error. Retrying in {wait}s...")
                time.sleep(wait)

        return None

    def run_complete_harvest(self, prefix_stems=None, search_keywords=None, output=None):
        prefix_stems = prefix_stems or DEFAULT_PREFIXES
        search_keywords = search_keywords or DEFAULT_KEYWORDS
        total_tasks = len(prefix_stems) + len(search_keywords)

        # Use auto to prevent newline issues in Notebooks
        from tqdm.auto import tqdm
        pbar = tqdm(total=total_tasks, desc="Harvesting Pages By Keywords", unit="keyword")

        # 1. Harvest by Prefix
        for kw in prefix_stems:
            pbar.set_postfix(current_kw=kw[:15], total_pages=len(self.all_titles))
            count = 0
            continue_token = None

            # Use .strip() to ensure no accidental double spaces break the prefix
            clean_kw = kw.strip() + " "

            while True:
                params = {
                    "action": "query",
                    "list": "allpages",
                    "apnamespace": 0,
                    "apprefix": clean_kw,
                    "aplimit": 500,
                    "format": "json"
                }
                if continue_token: params["apcontinue"] = continue_token

                data = self._safe_request(params)
                if not data or "query" not in data: break

                batch = [p["title"] for p in data["query"].get("allpages", [])]
                self.all_titles.update(batch)
                count += len(batch)

                if "continue" in data:
                    continue_token = data["continue"]["apcontinue"]
                    time.sleep(self.delay)
                else:
                    break

            self.metadata.append({"Method": "Prefix", "Keyword": kw, "Count": count})
            pbar.update(1)

        # 2. Harvest by Inline Search
        for kw in search_keywords:
            pbar.set_postfix(current_kw=kw[:15], total_pages=len(self.all_titles))
            count = 0
            continue_token = None

            while True:
                # Removed 'intitle' quotes to allow for more flexible matching
                # searching for intitle:members finds "Members of..." and "List of members..."
                params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": f'intitle:{kw}',
                    "srnamespace": 0,
                    "srlimit": 500,
                    "format": "json"
                }
                if continue_token: params["sroffset"] = continue_token

                data = self._safe_request(params)
                if not data or "query" not in data: break

                batch = [x["title"] for x in data["query"].get("search", [])]
                self.all_titles.update(batch)
                count += len(batch)

                if "continue" in data:
                    continue_token = data["continue"]["sroffset"]
                    time.sleep(self.delay)
                else:
                    break

            self.metadata.append({"Method": "Inline", "Keyword": kw, "Count": count})
            pbar.update(1)

        pbar.close()
        self.display_full_summary()

        if output:
            return list(self.all_titles)

    def display_full_summary(self, num_parts=5):
        df = pd.DataFrame(self.metadata)

        # Calculate how many rows per sub-table to split into num_parts parts
        chunk_size = int(np.ceil(len(df) / num_parts))

        # Generate HTML for each chunk
        tables_html = ""
        for i in range(num_parts):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            chunk = df.iloc[start:end]

            # We wrap each table in a div to give it some padding
            tables_html += f"""
            <div style="flex: 1; padding: 5px;">
                {chunk.to_html(index=False)}
            </div>
            """

        # Wrap the chunks in a Flexbox container
        scroll_html = f"""
        <div style="height: 400px; width: 100%; overflow: auto; border: 1px solid #ccc;">
            <div style="display: flex; flex-direction: row; align-items: flex-start;">
                {tables_html}
            </div>
        </div>
        """

        print(f"\nTOTAL UNIQUE TITLES COLLECTED: {len(self.all_titles)}\n")
        display(HTML(scroll_html))

    async def filter_pages_async(self, output=None):

        titles = list(self.all_titles)

        self.valid_titles = []
        semaphore = asyncio.Semaphore(15)
        chunks = [titles[i:i + 50] for i in range(0, len(titles), 50)]
        stats = {"good": 0}

        pbar = tqdm(total=len(titles), desc="Filtering Pages with Tables", unit="page")
        pbar.set_postfix(good_pages=0)

        async def check_batch(client_, chunk):
            async with semaphore:
                params = {
                    "action": "query", "prop": "revisions", "rvprop": "content",
                    "titles": "|".join(chunk), "format": "json", "formatversion": 2
                }

                # RETRY LOOP for 429 errors
                for attempt in range(3):
                    try:
                        resp = await client_.get(self.BASE_URL, params=params)

                        # NEGATE 429: Handle rate limit
                        if resp.status_code == 429:
                            wait = (attempt + 1) * 5
                            await asyncio.sleep(wait)
                            continue  # Try the same batch again

                        if resp.status_code != 200:
                            pbar.update(len(chunk))
                            return []

                        data = resp.json()
                        found_in_batch = []
                        pages = data.get("query", {}).get("pages", [])

                        for page in pages:
                            revisions = page.get("revisions")
                            if not revisions: continue

                            content = revisions[0].get("content", "")
                            if '{|' in content or 'wikitable' in content.lower():
                                found_in_batch.append(page["title"])

                        stats["good"] += len(found_in_batch)
                        pbar.set_postfix(good_pages=stats["good"])
                        pbar.update(len(chunk))
                        return found_in_batch

                    except Exception as f:
                        if attempt == 2:  # Last try failed
                            pbar.write(f"❌ Batch failed: {str(f)[:50]}...")
                            pbar.update(len(chunk))
                        await asyncio.sleep(1)

                return []

        # 1. Define strict connection limits
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)

        # 2. Break the massive task into smaller "Super-Batches" to prevent memory leaks
        all_results = []

        # Calculate how many super-batches we need
        for i in range(0, len(chunks), 100):  # 100 chunks of 50 = 5000 pages
            sub_chunks = chunks[i: i + 100]

            try:
                async with httpx.AsyncClient(
                        headers=self.session.headers,
                        timeout=60,
                        limits=limits,
                        verify=True  # Ensures SSL doesn't hang
                ) as client:

                    tasks = [check_batch(client, chunk) for chunk in sub_chunks]
                    # return_exceptions=True prevents one crash from killing the 85%
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                    for r in batch_results:
                        if isinstance(r, list):
                            all_results.extend(r)
                        elif isinstance(r, Exception):
                            pbar.write(f"⚠️ Worker died: {r}")

                # Small "Cool Down" for the OS between super-batches
                await asyncio.sleep(2)

            except Exception as e:
                pbar.write(f"❌ Super-Batch Error at {i * 50} pages: {e}")
                continue

        self.valid_titles = all_results
        pbar.close()

        print(f"\nTOTAL PAGES WITH TABLES COLLECTED: {len(self.valid_titles)}\n")

        if output:
            return self.valid_titles
