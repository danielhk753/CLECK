# # DEFAULT_KEYWORDS = [
# #     "list", "members", "teams", "countries", "cities", "species", "films",
# #     "awards", "companies", "airlines", "universities", "planets", "missions"
# # ]
#
#
# # def fetch_titles(self, keywords=None):
# #     """Phase 1: Harvest candidate titles."""
# #     target_keywords = keywords or self.DEFAULT_KEYWORDS
# #     all_titles = set()
# #
# #     print(f"Starting Phase 1: Harvesting titles for {len(target_keywords)} keywords...")
# #
# #     for kw in tqdm(target_keywords, desc="Harvesting", unit="keyword"):
# #         kw_titles_count = 0
# #         continue_token = None
# #         while True:
# #             params = {
# #                 "action": "query",
# #                 "list": "search",
# #                 "srsearch": f'intitle:"{kw}"',
# #                 "srnamespace": 0,
# #                 "srlimit": 500,
# #                 "format": "json"
# #             }
# #             if continue_token:
# #                 params["sroffset"] = continue_token
# #
# #             try:
# #                 response = self.session.get(self.BASE_URL, params=params)
# #                 response.raise_for_status()
# #                 data = response.json()
# #             except:
# #                 break
# #
# #             batch = [x["title"] for x in data.get("query", {}).get("search", [])]
# #             all_titles.update(batch)
# #             kw_titles_count += len(batch)
# #
# #             if "continue" in data:
# #                 continue_token = data["continue"]["sroffset"]
# #                 time.sleep(self.delay)
# #             else:
# #                 break
# #
# #         self.results_metadata[kw] = kw_titles_count
# #
# #     self._print_summary(len(all_titles))
# #     return list(all_titles)
#
# import time
# import requests
# import re
# import pandas as pd
# import numpy as np
# import networkx as nx
# from tqdm import tqdm
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.neighbors import NearestNeighbors
# import scipy.sparse as sp
#
# DEFAULT_KEYWORDS = [
#     "list", "members", "teams", "countries", "cities", "species", "films",
#     "awards", "companies", "airlines", "universities", "planets", "missions"
# ]
#
# DEFAULT_PREFIXES = [
#     # --- GEOGRAPHY & PLACES (The highest volume) ---
#     "List of cities", "List of towns", "List of villages", "List of districts",
#     "List of municipalities", "List of communes", "List of counties", "List of provinces",
#     "List of islands", "List of mountains", "List of rivers", "List of lakes",
#     "List of waterfalls", "List of volcanoes", "List of caves", "List of deserts",
#     "List of beaches", "List of forests", "List of parks", "List of reservoirs",
#
#     # --- INFRASTRUCTURE & BUILDINGS ---
#     "List of airports", "List of stations", "List of ports", "List of bridges",
#     "List of tunnels", "List of highways", "List of roads", "List of stadiums",
#     "List of skyscrapers", "List of museums", "List of libraries", "List of hospitals",
#     "List of schools", "List of universities", "List of colleges", "List of banks",
#     "List of churches", "List of cathedrals", "List of mosques", "List of temples",
#     "List of synagogues", "List of castles", "List of palaces", "List of dams",
#     "List of power", "List of lighthouses", "List of monuments", "List of malls",
#
#     # --- PEOPLE & ROLES (High Link Density) ---
#     "List of presidents", "List of mayors", "List of governors", "List of monarchs",
#     "List of kings", "List of queens", "List of senators", "List of members",
#     "List of ambassadors", "List of ministers", "List of bishops", "List of popes",
#     "List of actors", "List of directors", "List of writers", "List of poets",
#     "List of painters", "List of musicians", "List of composers", "List of singers",
#     "List of athletes", "List of players", "List of olympic", "List of billionaires",
#     "List of philosophers", "List of scientists", "List of inventors", "List of alumni",
#
#     # --- ORGANIZATIONS & COMPANIES ---
#     "List of companies", "List of airlines", "List of manufacturers", "List of brands",
#     "List of schools", "List of newspapers", "List of magazines", "List of journals",
#     "List of organizations", "List of agencies", "List of nonprofits", "List of retailers",
#
#     # --- SCIENCE & TAXONOMY ---
#     "List of species", "List of mammals", "List of birds", "List of reptiles",
#     "List of fish", "List of insects", "List of plants", "List of trees",
#     "List of fungi", "List of minerals", "List of elements", "List of isotopes",
#     "List of stars", "List of galaxies", "List of asteroids", "List of comets",
#
#     # --- MEDIA & CULTURE ---
#     "List of films", "List of series", "List of episodes", "List of characters",
#     "List of books", "List of albums", "List of singles", "List of games",
#     "List of festivals", "List of awards", "List of paintings", "List of buildings",
#
#     # --- EVENTS & HISTORY ---
#     "List of battles", "List of wars", "List of treaties", "List of riots",
#     "List of protests", "List of massacres", "List of accidents", "List of earthquakes",
#     "List of shipwrecks",
#
#     # Members & Groups
#     "Members of ", "Fellows of ", "Alumni of ", "Players of ", "Roster of ",
#     "Winners of ", "Recipients of ", "Inductees of ", "Signatories of ",
#
#     # Geography & Places
#     "Cities in ", "Towns in ", "Villages in ", "Communes in ", "Districts of ",
#     "Provinces of ", "Counties in ", "Islands of ", "Mountains in ", "Rivers of ",
#     "Lakes of ", "Bridges in ", "Airports in ", "Stations in ", "Museums in ",
#     "Hospitals in ", "Schools in ", "Universities in ", "Parks in ", "Dams in ",
#
#     # Countries & Comparisons
#     "Countries by ", "States by ", "Ranking of ", "Comparison of ", "Index of ",
#     "Statistics of ", "Demographics of ", "Economy of ", "Politics of ",
#
#     # People & Occupations
#     "Presidents of ", "Mayors of ", "Governors of ", "Monarchs of ", "Kings of ",
#     "Ambassadors of ", "Ministers of ", "People from ", "Artists from ",
#     "Actors in ", "Directors of ", "Writers of ", "Scientists of ",
#
#     # Companies & Organizations
#     "Companies of ", "Airlines of ", "Banks in ", "Newspapers in ",
#     "Organizations based in ", "Manufacturers of ", "Retailers of ",
#
#     # Science & Species
#     "Species of ", "Genera of ", "Birds of ", "Mammals of ", "Fish of ",
#     "Plants of ", "Insects of ", "Minerals of ", "Elements in ", "Stars in ",
#
#     # Media & Arts
#     "Films of ", "Series of ", "Episodes of ", "Characters in ", "Books by ",
#     "Albums by ", "Songs by ", "Singles by ", "Games in ", "Paintings by ",
#     "Awards for ", "Festivals in ", "Buildings in ", "Historic sites in ",
#
#     # History & Events
#     "Battles of ", "Wars in ", "Treaties of ", "Events in ",
#     "Shipwrecks of ", "Disasters in ", "Elections in ", "Protests in "
# ]
#
# class WikiHarvester:
#     BASE_URL = "https://en.wikipedia.org/w/api.php"
#
#     # DEFAULT_KEYWORDS = [
#     #     "list of", "comparison of", "discography of", "filmography of",
#     #     "demographics of", "members", "countries", "cities", "species", "films",
#     #     "awards", "companies", "universities"
#     # ]
#
#     def __init__(self, user_agent="Research/0.1 (your@email.com)", delay=0.05):
#         self.session = requests.Session()
#         self.session.headers.update({"User-Agent": user_agent})
#         self.delay = delay
#         self.total_found = 0
#         self.results_metadata = {}
#         self.all_titles = set()
#         self.metadata = []
#
#     def _safe_request(self, params):
#         try:
#             response = self.session.get(self.BASE_URL, params=params, timeout=15)
#             response.raise_for_status()
#             return response.json()
#         except Exception:
#             return None
#
#     def harvest_by_prefix(self):
#         """Method A: Traditional Prefix Indexing (Deep & Sequential)"""
#         print(f"Phase 1A: Harvesting by Prefix ({len(DEFAULT_PREFIXES)} stems)...")
#         for kw in tqdm(DEFAULT_PREFIXES, desc="Prefix Search", unit="stem"):
#             count = 0
#             continue_token = None
#             while True:
#                 params = {
#                     "action": "query", "list": "allpages", "apnamespace": 0,
#                     "apprefix": kw, "aplimit": 500, "format": "json"
#                 }
#                 if continue_token: params["apcontinue"] = continue_token
#
#                 data = self._safe_request(params)
#                 if not data: break
#
#                 pages = [p["title"] for p in data.get("query", {}).get("allpages", [])]
#                 self.all_titles.update(pages)
#                 count += len(pages)
#
#                 if "continue" in data:
#                     continue_token = data["continue"]["apcontinue"]
#                     time.sleep(self.delay)
#                 else:
#                     break
#             self.metadata.append({"Method": "Prefix", "Keyword": kw, "Found": count})
#
#     def harvest_by_search(self):
#         """Method B: Full-Text 'intitle' Search (Broad & Keyword-focused)"""
#         print(f"Phase 1B: Harvesting by Search ({len(DEFAULT_KEYWORDS)} keywords)...")
#         for kw in tqdm(DEFAULT_KEYWORDS, desc="Intitle Search", unit="kw"):
#             count = 0
#             continue_token = None
#             while True:
#                 params = {
#                     "action": "query", "list": "search", "srnamespace": 0,
#                     "srsearch": f'intitle:"{kw}"', "srlimit": 500, "format": "json"
#                 }
#                 if continue_token: params["sroffset"] = continue_token
#
#                 data = self._safe_request(params)
#                 if not data: break
#
#                 results = [x["title"] for x in data.get("query", {}).get("search", [])]
#                 self.all_titles.update(results)
#                 count += len(results)
#
#                 if "continue" in data:
#                     continue_token = data["continue"]["sroffset"]
#                     time.sleep(self.delay)
#                 else:
#                     break
#             self.metadata.append({"Method": "Search", "Keyword": kw, "Found": count})
#
#     def show_summary(self):
#         df = pd.DataFrame(self.metadata)
#         print("\n" + "=" * 40)
#         print(f"HARVESTING COMPLETE: {len(self.all_titles)} Unique Titles")
#         print("=" * 40)
#         # Grouping to show summary by method
#         summary = df.groupby(['Method', 'Keyword']).sum()
#         print(summary)
#         print("=" * 40)
#         return df
#
#     # def _safe_request(self, params, pbar, max_retries=5):
#     #     """Handles requests with backoff and updates tqdm status instead of printing."""
#     #     retries = 0
#     #     backoff = 2
#     #
#     #     while retries < max_retries:
#     #         try:
#     #             response = self.session.get(self.BASE_URL, params=params, timeout=15)
#     #
#     #             if response.status_code == 429:
#     #                 wait_time = int(response.headers.get("Retry-After", backoff))
#     #                 # Update status to Halted and keep total count visible
#     #                 pbar.set_description(f"HALTED (429)")
#     #                 pbar.set_postfix(total_pages=self.total_found, wait=f"{wait_time}s")
#     #
#     #                 time.sleep(wait_time)
#     #                 retries += 1
#     #                 backoff *= 2
#     #                 continue
#     #
#     #             # Reset status to Running if request succeeds
#     #             pbar.set_description(f"RUNNING")
#     #             response.raise_for_status()
#     #             return response.json()
#     #
#     #         except Exception as e:
#     #             pbar.set_description(f"ERROR (Retry {retries})")
#     #             time.sleep(backoff)
#     #             retries += 1
#     #             backoff *= 2
#     #
#     #     return None
#     #
#     # def fetch_titles(self, keywords=None):
#     #     target_keywords = keywords or self.DEFAULT_KEYWORDS
#     #     all_titles = set()
#     #
#     #     print(f"Starting Phase 1: Harvesting titles for {len(target_keywords)} keywords...")
#     #
#     #     # Wrap keywords in tqdm
#     #     with tqdm(target_keywords, unit="keyword", dynamic_ncols=True) as pbar:
#     #         for kw in pbar:
#     #             kw_titles_count = 0
#     #             continue_token = None
#     #
#     #             # Update prefix to show current keyword being processed
#     #             pbar.set_postfix(kw=kw, total_pages=self.total_found)
#     #
#     #             while True:
#     #                 params = {
#     #                     "action": "query",
#     #                     "list": "allpages",
#     #                     "apprefix": kw,
#     #                     "apnamespace": 0,
#     #                     "aplimit": 500,
#     #                     "format": "json"
#     #                 }
#     #                 if continue_token:
#     #                     params["apcontinue"] = continue_token
#     #
#     #                 data = self._safe_request(params, pbar)
#     #                 if not data:
#     #                     break
#     #
#     #                 pages = data.get("query", {}).get("allpages", [])
#     #                 batch = [p["title"] for p in pages]
#     #
#     #                 # Logic to track unique total count
#     #                 before_count = len(all_titles)
#     #                 all_titles.update(batch)
#     #                 self.total_found = len(all_titles)
#     #
#     #                 kw_titles_count += len(batch)
#     #
#     #                 # Live update of the found count on the bar
#     #                 pbar.set_postfix(total_pages=self.total_found, current_kw=kw[:10])
#     #
#     #                 if "continue" in data:
#     #                     continue_token = data["continue"]["apcontinue"]
#     #                     time.sleep(self.delay)
#     #                 else:
#     #                     break
#     #
#     #             self.results_metadata[kw] = kw_titles_count
#     #
#     #     self._print_summary(len(all_titles))
#     #     return list(all_titles)
#     #
#     # def cluster_and_deduplicate(self, titles, threshold=0.9):
#     #     """Phase 2: Memory-Efficient Clustering using Sparse Nearest Neighbors."""
#     #     if not titles:
#     #         return []
#     #
#     #     # 1. Normalization
#     #     def clean(t):
#     #         t = t.lower()
#     #         t = re.sub(r'\b(list of|table of|comparison of)\b', '', t)
#     #         return re.sub(r'[^\w\s]', '', t).strip()
#     #
#     #     cleaned_titles = [clean(t) for t in tqdm(titles, desc="Normalizing")]
#     #
#     #     # 2. Sparse Vectorization
#     #     print("Phase 2.2: Vectorizing...")
#     #     # Use max_features to keep the vocabulary manageable
#     #     vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char_wb', max_features=100000)
#     #     tfidf_matrix = vectorizer.fit_transform(cleaned_titles)
#     #
#     #     # 3. Use BallTree / NearestNeighbors (This is the 'Smart' bit)
#     #     # Instead of N*N, we only look for the neighbors near our threshold
#     #     print("Phase 2.3: Finding similar titles (Sparse Search)...")
#     #     # metric='cosine' with a radius search
#     #     nbrs = NearestNeighbors(radius=1 - threshold, metric='cosine', algorithm='brute', n_jobs=-1)
#     #     nbrs.fit(tfidf_matrix)
#     #
#     #     # This returns only the pairs that are similar enough
#     #     # 'indices' is a list of arrays containing matching title IDs
#     #     adj_matrix = nbrs.radius_neighbors_graph(tfidf_matrix)
#     #
#     #     # 4. Graph Construction from Sparse Matrix
#     #     print("Phase 2.4: Building Graph...")
#     #     G = nx.from_scipy_sparse_array(adj_matrix)
#     #
#     #     # 5. Extract Clusters
#     #     print("Phase 2.5: Extracting Clusters...")
#     #     clusters = list(nx.connected_components(G))
#     #
#     #     unique_titles = []
#     #     for cluster in tqdm(clusters, desc="Selecting Reps"):
#     #         unique_titles.append(titles[list(cluster)[0]])
#     #
#     #     print(f"Deduplication Complete: {len(titles)} -> {len(unique_titles)}")
#     #     return unique_titles
#
#     def filter_pages_with_tables(self, titles):
#         """
#         Phase 3: Filter out pages that do not contain a 'wikitable'.
#         Uses batching (50 titles per request) for efficiency.
#         """
#         print(f"Starting Phase 3: Filtering {len(titles)} pages for tables...")
#         valid_titles = []
#
#         # Split titles into chunks of 50
#         chunks = [titles[i:i + 50] for i in range(0, len(titles), 50)]
#
#         for chunk in tqdm(chunks, desc="Filtering", unit="batch"):
#             params = {
#                 "action": "query",
#                 "prop": "revisions",
#                 "rvprop": "content",
#                 "titles": "|".join(chunk),
#                 "format": "json",
#                 "formatversion": 2  # Makes response easier to parse
#             }
#
#             try:
#                 response = self.session.get(self.BASE_URL, params=params)
#                 response.raise_for_status()
#                 data = response.json()
#
#                 pages = data.get("query", {}).get("pages", [])
#                 for page in pages:
#                     if "revisions" in page:
#                         content = page["revisions"][0].get("content", "")
#                         # Look for the standard wikitext table class
#                         if 'class="wikitable' in content or 'class=\'wikitable' in content or '{|' in content:
#                             valid_titles.append(page["title"])
#
#             except Exception as e:
#                 print(f"\n[Error] Batch failed: {e}")
#                 continue
#
#             time.sleep(self.delay)
#
#         print(f"\nPhase 3 Complete: Kept {len(valid_titles)} pages out of {len(titles)}.")
#         return valid_titles
#
#     def _print_summary(self, total_unique):
#         print("\n" + "=" * 44)
#         print(f"{'Keyword':<30} | {'Pages Found':<10}")
#         print("-" * 44)
#         for kw, count in self.results_metadata.items():
#             print(f"{kw:<30} | {count:<10}")
#         print("-" * 44)
#         print(f"{'TOTAL UNIQUE':<30} | {total_unique:<10}")
#         print("=" * 44 + "\n")

import asyncio
import httpx
import pandas as pd
import time
import requests
from tqdm.auto import tqdm
from IPython.display import display, HTML

DEFAULT_KEYWORDS = [
    "list", "members", "teams", "countries", "cities", "species", "films",
    "awards", "companies", "airlines", "universities", "planets", "missions"
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
        self.valid_titles = []
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.delay = delay
        self.all_titles = set()
        self.metadata = []


    def _safe_request(self, params):
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except:
            return None

    def run_complete_harvest(self, prefix_stems=None, search_keywords=None, output=None):
        prefix_stems = prefix_stems or DEFAULT_PREFIXES
        search_keywords = search_keywords or DEFAULT_KEYWORDS
        total_tasks = len(prefix_stems) + len(search_keywords)

        # Use auto to prevent newline issues in Notebooks
        from tqdm.auto import tqdm
        pbar = tqdm(total=total_tasks, desc="RUNNING", unit="keyword")

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

    def display_full_summary(self):
        df = pd.DataFrame(self.metadata)

        # Generate the HTML for the table
        html_table = df.to_html()

        # Wrap it in a scrollable div with a fixed height
        scroll_html = f"""
        <div style="height: 400px; width: 100%; overflow: auto; border: 1px solid #ccc;">
            {html_table}
        </div>
        """

        print(f"\nTOTAL UNIQUE TITLES COLLECTED: {len(self.all_titles)}\n")
        display(HTML(scroll_html))

    def filter_pages_with_tables(self, output=None):
        """
        Phase 2: Filter out pages that do not contain a 'wikitable'.
        Updates self.valid_titles and returns them.
        """

        titles = list(self.all_titles)

        # Using tqdm.auto for the notebook-friendly slide bar
        pbar = tqdm(total=len(titles), desc="FILTERING", unit="page")
        pbar.set_postfix(good_pages=0)

        # Split titles into chunks of 50 (Wikipedia API limit for titles)
        chunks = [titles[i:i + 50] for i in range(0, len(titles), 50)]
        newly_validated = []

        for chunk in chunks:
            params = {
                "action": "query",
                "prop": "revisions",
                "rvprop": "content",
                "titles": "|".join(chunk),
                "format": "json",
                "formatversion": 2
            }

            try:
                response = self.session.get(self.BASE_URL, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()

                pages = data.get("query", {}).get("pages", [])
                for page in pages:
                    if "revisions" in page:
                        content = page["revisions"][0].get("content", "")
                        # Detection logic: looks for class or the standard {| table start
                        if 'wikitable' in content.lower() or '{|' in content:
                            newly_validated.append(page["title"])

                    # Update bar for every page in the chunk
                    pbar.update(1)

                # Show how many valid tables found so far in the postfix
                pbar.set_postfix(good_pages=len(newly_validated))

            except Exception as e:
                # Using pbar.write to prevent breaking the slide bar with errors
                pbar.write(f"Batch failed: {e}")
                pbar.update(len(chunk))
                continue

            time.sleep(self.delay)

        pbar.close()

        # Store in the class instance
        self.valid_titles = newly_validated

        print(f"\nPhase 2 Complete: Kept {len(self.valid_titles)} pages out of {len(titles)}.")

        if output:
            return self.valid_titles

    async def filter_pages_async(self, output=None):

        titles = list(self.all_titles)

        self.valid_titles = []
        semaphore = asyncio.Semaphore(15)
        chunks = [titles[i:i + 50] for i in range(0, len(titles), 50)]
        stats = {"good": 0}

        pbar = tqdm(total=len(titles), desc="ASYNC FILTER", unit="page")
        pbar.set_postfix(good_pages=0)

        async def check_batch(client, chunk):
            async with semaphore:
                params = {
                    "action": "query", "prop": "revisions", "rvprop": "content",
                    "titles": "|".join(chunk), "format": "json", "formatversion": 2
                }

                # RETRY LOOP for 429 errors
                for attempt in range(3):
                    try:
                        resp = await client.get(self.BASE_URL, params=params)

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

                    except Exception as e:
                        if attempt == 2:  # Last try failed
                            pbar.write(f"❌ Batch failed: {str(e)[:50]}...")
                            pbar.update(len(chunk))
                        await asyncio.sleep(1)

                return []

        async with httpx.AsyncClient(headers=self.session.headers, timeout=30) as client:
            tasks = [check_batch(client, chunk) for chunk in chunks]
            results = await asyncio.gather(*tasks)
            self.valid_titles = [t for sublist in results for t in sublist]

        pbar.close()

        if output:
            return self.valid_titles

    # def filter_pages_with_tables(self, titles):
    #     """
    #     Filter out pages that do not contain a 'wikitable'.
    #     Uses batching (50 titles per request) for efficiency.
    #     """
    #     print(f"Starting Phase 2: Filtering {len(titles)} pages for tables...")
    #     valid_titles = []
    # 
    #     # Split titles into chunks of 50
    #     chunks = [titles[i:i + 50] for i in range(0, len(titles), 50)]
    # 
    #     for chunk in tqdm(chunks, desc="Filtering", unit="batch"):
    #         params = {
    #             "action": "query",
    #             "prop": "revisions",
    #             "rvprop": "content",
    #             "titles": "|".join(chunk),
    #             "format": "json",
    #             "formatversion": 2  # Makes response easier to parse
    #         }
    # 
    #         try:
    #             response = self.session.get(self.BASE_URL, params=params)
    #             response.raise_for_status()
    #             data = response.json()
    # 
    #             pages = data.get("query", {}).get("pages", [])
    #             for page in pages:
    #                 if "revisions" in page:
    #                     content = page["revisions"][0].get("content", "")
    #                     # Look for the standard wikitext table class
    #                     if 'class="wikitable' in content or 'class=\'wikitable' in content or '{|' in content:
    #                         valid_titles.append(page["title"])
    # 
    #         except Exception as e:
    #             print(f"\n[Error] Batch failed: {e}")
    #             continue
    # 
    #         time.sleep(self.delay)
    # 
    #     print(f"\nPhase 2 Complete: Kept {len(valid_titles)} pages out of {len(titles)}.")
    #     return valid_titles