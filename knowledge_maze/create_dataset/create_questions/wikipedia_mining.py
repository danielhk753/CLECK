"""
WikiMiner: Targeted Wikipedia Discovery and Structural Validation Engine

This module facilitates the automated discovery of Wikipedia "List" pages
across diverse thematic domains. It employs a multi-stage pipeline to find,
verify, and de-duplicate high-value data sources.

WORKFLOW:
1.  DISCOVERY: Executes prefix-based (e.g., "Films of...") and keyword-based
    (e.g., "List of...") API queries against ~200+ entity categories.
2.  VALIDATION: Performs high-concurrency asynchronous HTTP requests to
    inspect the raw wikitext of discovered pages, filtering for those
    containing actual data structures ({| or 'wikitable').
3.  CLUSTERING: Utilizes TF-IDF Vectorization and Cosine Similarity to identify
    thematic overlaps. A graph-based (NetworkX) approach groups similar titles,
    selecting a representative to eliminate redundant data.
4.  REPORTING: Generates a low-latency, dark-mode HTML dashboard for
    real-time performance monitoring and data distribution analysis.

DEPENDENCIES:
    - httpx/asyncio: For non-blocking structural validation.
    - sklearn: For NLP-based title vectorization.
    - networkx: For cluster discovery via connected components.
    - tqdm: For granular progress tracking.
"""

import time
import httpx
import random
import asyncio
import requests
import numpy as np
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm
from IPython.display import display, HTML
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


ENTITY_GROUPS = {
    "Films & TV": [
        "films", "movies", "television series", "series", "animated films", "animated series",
        "documentaries", "short films", "feature films", "miniseries", "web series", "episodes"
    ],
    "Books & Literature": [
        "books", "novels", "short stories", "poems", "plays", "manga", "comics", "graphic novels", "volumes", "authors"
    ],
    "Music": [
        "albums", "singles", "songs", "tracks", "musicians", "bands", "singers", "composers",
         "operas", "symphonies", "concertos", "music videos", "recordings", "releases", "concerts", "tours"
    ],
    "Countries & Geography": [
        "countries", "counties", "cities", "towns", "villages", "states", "provinces", "regions", "territories",
        "islands", "rivers", "mountains", "lakes", "volcanoes", "deserts", "forests", "peninsulas", "bays",
        "gulfs", "valleys", "plateaus", "districts"
    ],
    "Infrastructure": [
        "airports", "stations", "railways", "tramways", "ports", "harbors", "roads", "highways", "motorways", "streets",
        "bridges", "tunnels", "buildings", "stadiums", "arenas", "skyscrapers", "terminals", "dams", "reservoirs",
        "canals", "power plants", "power stations", "pipelines", "hospitals", "schools", "universities", "libraries",
        "museums", "factories", "warehouses","campuses"
    ],
    "Companies & Organizations": [
        "companies", "corporations", "subsidiaries", "airlines", "banks", "investment banks", "credit unions",
        "central banks", "insurance companies", "newspapers", "news agencies", "magazines", "colleges",
        "research institutes", "schools", "high schools", "elementary schools", "hospitals",
        "charities", "foundations", "ngos", "shops", "retail chains"
    ],
    "Sports": [
        "athletes", "players", "coaches", "managers", "referees", "teams", "clubs", "national teams", "stadiums",
        "arenas", "circuits", "tournaments", "leagues", "cups", "championships", "finals", "medalists", "world records",
        "season records", "sports rivalries","footballers", "cricketers", "basketballers",
    ],
    "Science & Nature": [
        "species", "subspecies", "genera", "mammals", "birds", "reptiles", "amphibians", "fish", "insects",
        "plants", "trees", "flowers", "crops", "minerals", "rocks", "elements", "chemical compounds",
        "planets", "moons", "asteroids", "comets", "stars", "nebulae", "galaxies", "habitats"
    ],
    "People & Roles": [
        "presidents", "kings", "queens", "emperors", "prime ministers", "chancellors", "governors", "mayors",
        "ambassadors", "diplomats", "philosophers", "theologians", "scientists", "researchers", "mathematicians",
        "inventors", "engineers", "poets", "novelists", "playwrights", "painters", "sculptors", "architects",
        "actors", "directors", "musicians", "members", "artists", "writers", "politicians", "leaders", "captains"
    ],
    "History & Events": [
        "battles", "sieges", "campaigns", "wars", "treaties", "alliances", "agreements", "revolutions", "uprisings",
        "coups", "massacres", "genocides", "earthquakes", "volcanic eruptions", "tsunamis", "famines", "shipwrecks",
        "aviation accidents", "elections", "referendums", "protests", "riots", "strikes", "shipwrecks"
    ],
    "Awards & Rankings": [
        "awards", "medals", "honors", "titles", "prizes", "winners", "nominees", "laureates",
        "rankings", "billionaires", "millionaires", "milestones", "firsts",
    ],
    "Technology & Computing": [
        "programming languages", "software", "operating systems", "data structures", "databases", "file systems",
        "video games", "game engines", "applications", "web browsers", "cryptocurrencies", "blockchains",
    ],
    "Politics & Government": [
        "political parties", "movements", "parliaments", "legislatures", "ministries", "government agencies",
        "constitutions", "laws", "acts", "election results",
    ],
    "Military & Security": [
        "armies", "navies", "air forces", "military units", "divisions", "brigades", "weapons", "missiles", "tanks",
        "warships", "fighter aircraft", "military operations", "campaigns",
    ],
    "Religion & Belief": [
        "religions", "denominations", "sects", "religious movements", "deities", "mythological figures",
        "scriptures", "churches", "temples", "mosques", "bishops"
    ],
    "Culture & Society": [
        "festivals", "holidays", "traditions", "folklore", "historic sites"
    ],
    "Languages & Linguistics": [
        "languages", "dialects", "writing systems", "alphabets", "scripts", "language families",
    ],
    "Economy & Finance": [
        "currencies", "coins", "banknotes", "stock exchanges", "companies by revenue", "recessions"
    ],
    "Law & Crime": [
        "laws", "court cases", "trials", "judges", "lawyers", "crimes", "scandals",
    ],
    "Health & Medicine": [
        "diseases", "syndromes", "drugs", "medications", "treatments"
    ],
    "Food & Agriculture": [
        "foods", "dishes", "ingredients", "beverages", "crops", "livestock",
    ],
    "Art & Architecture": [
        "art movements", "paintings", "sculptures", "architectural styles", "buildings", "skyscrapers",
        "monuments", "memorials",
    ],
}

ALL_ENTITIES = [entity for group in ENTITY_GROUPS.values() for entity in group]


def build_search_queries(entity: str) -> list[dict]:
    queries = []
    cap = entity.capitalize()
    for prep in ("in", "of"):
        queries.append({"method": "prefix", "label": f'"{cap} {prep} ..."', "prefix": f"{cap} {prep} ",
                        "pattern_type": f"X {prep} ..."})
    queries.append({"method": "list_of_search", "label": f'"List of ... {entity} ..."', "entity": entity,
                    "pattern_type": "List of ... X ..."})
    return queries


class WikiMiner:
    BASE_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(self, user_agent: str = "Research/0.1", delay: float = 0.05, entities: list[str] | None = None):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.delay = delay
        self.entities = entities or ALL_ENTITIES
        self.all_titles: set[str] = set()
        self.valid_titles: list[str] = []
        self.title_info: dict[str, dict] = {}
        self.similar_pages = []

        # Build subject map
        self.entity_to_subject = {}
        for subject, ent_list in ENTITY_GROUPS.items():
            for ent in ent_list:
                self.entity_to_subject[ent] = subject

    def _safe_request(self, params: dict) -> dict | None:
        for attempt in range(5):
            try:
                resp = self.session.get(self.BASE_URL, params=params, timeout=15)
                if resp.status_code == 429:
                    time.sleep((attempt + 1) * 5)
                    continue
                resp.raise_for_status()
                return resp.json()
            except:
                time.sleep((attempt + 1) * 2)
        return None

    def _harvest_by_prefix(self, prefix: str) -> list[str]:
        titles, continue_token = [], None
        while True:
            params = {"action": "query", "list": "allpages", "apnamespace": 0, "apprefix": prefix, "aplimit": 500,
                      "format": "json"}
            if continue_token: params["apcontinue"] = continue_token
            data = self._safe_request(params)
            if not data or "query" not in data: break
            titles.extend([p["title"] for p in data["query"].get("allpages", [])])
            if "continue" in data:
                continue_token = data["continue"]["apcontinue"]
                time.sleep(self.delay)
            else:
                break
        return titles

    def _harvest_list_of_entity(self, entity: str) -> list[str]:
        titles, continue_token = [], None
        while True:
            params = {"action": "query", "list": "search", "srsearch": f'intitle:"List of" intitle:{entity}',
                      "srnamespace": 0, "srlimit": 500, "format": "json"}
            if continue_token: params["sroffset"] = continue_token
            data = self._safe_request(params)
            if not data or "query" not in data: break
            titles.extend([x["title"] for x in data["query"].get("search", []) if
                           x["title"].lower().startswith("list of") and entity.lower() in x["title"].lower()])
            if "continue" in data:
                continue_token = data["continue"]["sroffset"]
                time.sleep(self.delay)
            else:
                break
        return titles

    def run_complete_search(self) -> None:
        total_tasks = len(self.entities)
        pbar = tqdm(total=total_tasks, desc="Mining Pages", unit="entity")
        for entity in self.entities:
            subject = self.entity_to_subject.get(entity, "Misc")
            for q in build_search_queries(entity):
                pbar.set_postfix(entity=entity[:12], total=len(self.all_titles))
                batch = self._harvest_by_prefix(q["prefix"]) if q["method"] == "prefix" else self._harvest_list_of_entity(q["entity"])
                new_titles = [t for t in batch if t not in self.all_titles]
                self.all_titles.update(new_titles)
                for t in new_titles:
                    self.title_info[t] = {"entity": entity, "pattern_type": q["pattern_type"], "subject": subject}
            pbar.update(1)
        pbar.close()

    async def filter_pages_async(self):

        titles = list(self.all_titles)

        self.valid_titles = []
        semaphore = asyncio.Semaphore(15)
        chunks = [titles[i:i + 50] for i in range(0, len(titles), 50)]
        stats = {"good": 0}

        pbar = tqdm(total=len(titles), desc="Filtering Pages with Tables", unit="page")
        pbar.set_postfix(valid=0)

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
                        pbar.set_postfix(valid=stats["good"])
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

    def cluster_titles(self, similarity_threshold=0.75, batch_size=1000):
        if not self.valid_titles:
            return []

        # 1. Vectorize (Fast)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.valid_titles)

        # 2. Compute Similarity in Batches (The Long Part)
        g = nx.Graph()
        g.add_nodes_from(range(len(self.valid_titles)))

        num_titles = tfidf_matrix.shape[0]
        pbar = tqdm(total=num_titles, desc=f"Computing Similarities for {len(self.valid_titles)} Titles", unit="titles")

        for start_idx in range(0, num_titles, batch_size):
            end_idx = min(start_idx + batch_size, num_titles)

            # Compare current batch against ALL titles
            batch_sim = cosine_similarity(tfidf_matrix[start_idx:end_idx], tfidf_matrix)

            # Find matches above threshold within this batch
            rows, cols = np.where(batch_sim > similarity_threshold)

            # Shift rows back to global index and filter self-loops/duplicates
            edges = [
                (r + start_idx, c)
                for r, c in zip(rows, cols)
                if (r + start_idx) < c
            ]
            g.add_edges_from(edges)

            pbar.update(end_idx - start_idx)
        pbar.close()

        # 3. Extract Clusters and Pick Representative
        clusters = list(nx.connected_components(g))
        final_titles = []

        for cluster_nodes in tqdm(clusters, desc=f"Picking Representatives from {len(clusters)} Clusters", unit="cluster"):
            cluster_list = list(cluster_nodes)

            if len(cluster_list) == 1:
                final_titles.append(self.valid_titles[cluster_list[0]])
                continue

            representative_node = random.choice(cluster_list)
            final_titles.append(self.valid_titles[representative_node])

            # Capture the relationship between the representative and the next most relevant title
            peer_node = random.choice(cluster_list)
            while peer_node == representative_node:
                peer_node = random.choice(cluster_list)
            self.similar_pages.append((
                self.valid_titles[representative_node],
                self.valid_titles[peer_node]
            ))

        self.valid_titles = final_titles

    def display_summary(self) -> None:
        total, valid = len(self.all_titles), len(self.valid_titles)
        rate = (valid / total * 100) if total > 0 else 0
        df_v = pd.DataFrame([self.title_info[t] for t in self.valid_titles if t in self.title_info])

        # Aggregations
        patterns = df_v['pattern_type'].value_counts() if not df_v.empty else {}
        subjects = df_v['subject'].value_counts().nlargest(10) if not df_v.empty else {}
        entities = df_v['entity'].value_counts().nlargest(10) if not df_v.empty else {}

        def make_table(series, title, color, is_similarity=False):
            rows = ""
            if is_similarity:
                # Show exactly 3 rows to help balance the column height
                examples = list(getattr(self, 'similar_pages', []))[:3]
                for survivor, discarded in examples:
                    # Neutral styling, centered text, with a subtle separator
                    rows += f"""<tr style='border-bottom: 1px solid #334155; height: 45px;'>
                        <td style='padding: 5px; color: #f8fafc; background-color: #1e293b; font-size: 11px; text-align: center;'>
                            {survivor}<br/>
                            <span style='color: #64748b; font-size: 10px;'>≈</span><br/>
                            {discarded}
                        </td>
                    </tr>"""
                # Pad with empty rows if there aren't enough examples to keep the rectangle shape
                for _ in range(4 - len(examples)):
                    rows += "<tr style='height: 45px;'><td style='background-color: #1e293b;'>&nbsp;</td></tr>"
            else:
                # Standard data rows
                items = list(series.items())
                for k, v in items:
                    rows += f"""<tr style='border-bottom: 1px solid #334155; height: 40px;'>
                        <td style='padding: 10px; color: #f8fafc; background-color: #1e293b; text-align: center;'>{k}</td>
                        <td style='padding: 10px; text-align: center; color: {color}; background-color: #1e293b; font-weight: bold;'>{v:,}</td>
                    </tr>"""
                # Ensure tables like "Patterns" stay consistent if fewer than 10 rows
                # (Though we are stacking Patterns + Similarity to fill one 10-row column)

            return f"""
            <div style='flex: 1; min-width: 240px; background-color: #1e293b; border: 1px solid #ffffff; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column;'>
                <h4 style='color: {color}; background-color: #0f172a; margin: 0; padding: 10px; text-align: center; border-bottom: 1px solid #ffffff; font-size: 14px;'>
                    {title}
                </h4>
                <table style='width: 100%; border-collapse: collapse; font-size: 13px; table-layout: fixed;'>
                    {rows}
                </table>
            </div>
            """

        # Build the components
        pattern_table = make_table(patterns, "Pattern Type", "#38bdf8")
        similarity_table = make_table(None, "Clustering Examples", "#94a3b8", is_similarity=True)
        subject_table = make_table(subjects, "Top Subjects", "#fbbf24")
        phrase_table = make_table(entities, "Top Phrases", "#f472b6")

        html = f"""
        <div style="background: #0f172a; padding: 25px; border-radius: 15px; border: 1px solid #ffffff; color: #f8fafc; font-family: sans-serif; max-width: 1100px;">
            <h1 style="text-align: center; color: #38bdf8; margin-bottom: 30px;">🛰 Wikipedia Page Extraction Summary</h1>

            <div style="display: flex; gap: 15px; margin-bottom: 20px;">
                <div style="flex: 1; background: #1e293b; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #ffffff;">
                    <small style="color: #94a3b8;">DISCOVERED</small><br/><span style="font-size: 22px; font-weight: bold;">{total:,}</span>
                </div>
                <div style="flex: 1; background: #1e293b; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #ffffff;">
                    <small style="color: #94a3b8;">VALIDATED</small><br/><span style="font-size: 22px; font-weight: bold; color: #34d399;">{valid:,}</span>
                </div>
                <div style="flex: 1; background: #1e293b; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #ffffff;">
                    <small style="color: #94a3b8;">EFFICIENCY</small><br/><span style="font-size: 22px; font-weight: bold; color: #c084fc;">{rate:.1f}%</span>
                </div>
            </div>

            <div style="display: flex; gap: 15px; align-items: stretch;">
                <div style="flex: 1; display: flex; flex-direction: column; gap: 15px;">
                    {pattern_table}
                    {similarity_table}
                </div>

                <div style="flex: 1; display: flex;">
                    {subject_table}
                </div>

                <div style="flex: 1; display: flex;">
                    {phrase_table}
                </div>
            </div>
        </div>
        """
        display(HTML(html))

    async def extract(self) -> None:
        self.run_complete_search()
        await self.filter_pages_async()
        self.cluster_titles()
        self.display_summary()
