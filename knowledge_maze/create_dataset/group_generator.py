# import os
# import json
# import random
# import time
# from collections import Counter
# from typing import Dict, List, Any
#
# import requests
#
#
# WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
# HEADERS = {
#     "User-Agent": "KnowledgeMaze/1.0 (research; contact: your@email)"
# }
#
# REPO_PATH = os.getenv('REPO_PATH')
# if REPO_PATH is None:
#     raise RuntimeError("REPO_PATH is not set in the .env file")
#
# MIN_GROUP_SIZE = 5
# MAX_GROUP_SIZE = 1000
#
# # controls sparsity
# MAX_VALUES_PER_FILTER = 25     # cap frequent values
# MAX_GROUPS_PER_FILTER = 10     # how many groups we sample per filter
# P_TWO_FILTERS = 0.15           # probability of adding a second filter
#
#
# # ---------------------------
# # Low-level SPARQL utilities
# # ---------------------------
#
# def run_sparql(query: str) -> List[Dict[str, Any]]:
#     r = requests.get(
#         WIKIDATA_SPARQL,
#         params={"query": query, "format": "json"},
#         headers=HEADERS,
#         timeout=60
#     )
#     r.raise_for_status()
#     return r.json()["results"]["bindings"]
#
#
# # ---------------------------
# # Seed loading
# # ---------------------------
#
# def load_seed(seed_idx: int, path=os.path.join(REPO_PATH, "data_files/group_seed.json")) -> Dict:
#     with open(path, "r", encoding="utf-8") as f:
#         seeds = json.load(f)
#     return seeds[seed_idx]
#
#
# # ---------------------------
# # Step 1: universe definition
# # ---------------------------
#
# def build_universe_where(seed: Dict) -> str:
#     clauses = []
#     for m in seed["mandatory_properties"]:
#         if "expected_value_qid" in m:
#             clauses.append(
#                 f"?entity wdt:{m['property']} wd:{m['expected_value_qid']} ."
#             )
#         else:
#             clauses.append(
#                 f"?entity wdt:{m['property']} ?_v_{m['property']} ."
#             )
#     return "\n".join(clauses)
#
#
# # ---------------------------
# # Step 2: discover common values
# # ---------------------------
#
# def discover_common_values(
#     seed: Dict,
#     filter_prop: Dict,
#     universe_where: str,
#     sample_limit=3000
# ) -> List[str]:
#     """
#     Returns QIDs of common categorical values (teams, countries, etc.)
#     """
#     query = f"""
#     SELECT ?val (COUNT(?entity) AS ?cnt) WHERE {{
#       {universe_where}
#       ?entity wdt:{filter_prop['property']} ?val .
#     }}
#     GROUP BY ?val
#     ORDER BY DESC(?cnt)
#     LIMIT {sample_limit}
#     """
#     rows = run_sparql(query)
#     return [r["val"]["value"].split("/")[-1] for r in rows]
#
#
# # ---------------------------
# # Step 3: build one group
# # ---------------------------
#
# def fetch_group_entities(
#     universe_where: str,
#     filters: List[Dict]
# ) -> List[str]:
#     where = [universe_where]
#     for f in filters:
#         where.append(
#             f"?entity wdt:{f['property']} wd:{f['value_qid']} ."
#         )
#
#     query = f"""
#     SELECT ?entity WHERE {{
#       {' '.join(where)}
#     }}
#     """
#     rows = run_sparql(query)
#     return [r["entity"]["value"].split("/")[-1] for r in rows]
#
#
# # ---------------------------
# # Step 4: sparse group generation
# # ---------------------------
#
# def generate_groups_from_seed(seed: Dict) -> List[Dict]:
#     """
#     Returns a list of group definitions:
#     {
#       group_id,
#       seed_domain,
#       seed_subdomain,
#       filters,
#       entity_qids
#     }
#     """
#     universe_where = build_universe_where(seed)
#     groups = []
#
#     categorical_filters = [
#         f for f in seed["optional_filters"]
#         if f["type"] == "categorical"
#     ]
#
#     for filt in categorical_filters:
#         values = discover_common_values(seed, filt, universe_where)
#
#         if not values:
#             continue
#
#         # sparsify aggressively
#         values = values[:MAX_VALUES_PER_FILTER]
#         sampled_vals = random.sample(
#             values,
#             min(len(values), MAX_GROUPS_PER_FILTER)
#         )
#
#         for val_qid in sampled_vals:
#             filters = [{
#                 "property": filt["property"],
#                 "name": filt["name"],
#                 "value_qid": val_qid
#             }]
#
#             # occasional second filter (low overlap)
#             if random.random() < P_TWO_FILTERS:
#                 other = random.choice(
#                     [f for f in categorical_filters if f != filt]
#                 )
#                 other_vals = discover_common_values(seed, other, universe_where)
#                 if other_vals:
#                     filters.append({
#                         "property": other["property"],
#                         "name": other["name"],
#                         "value_qid": random.choice(other_vals[:10])
#                     })
#
#             entities = fetch_group_entities(universe_where, filters)
#
#             if MIN_GROUP_SIZE <= len(entities) <= MAX_GROUP_SIZE:
#                 groups.append({
#                     "seed_domain": seed["domain"],
#                     "seed_subdomain": seed["subdomain"],
#                     "filters": filters,
#                     "entity_qids": entities
#                 })
#
#             time.sleep(0.2)  # polite throttling
#
#     return groups
#
#
#
# import requests
# import time
# from functools import lru_cache
#
# WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
#
# HEADERS = {
#     "Accept": "application/sparql-results+json",
#     "User-Agent": "GroupGenerator/1.0 (research use)"
# }
#
# # ------------------------------------------------------------------
# # Core SPARQL runner (rate-safe)
# # ------------------------------------------------------------------
#
# def run_sparql(query, sleep=0.2):
#     r = requests.get(
#         WIKIDATA_SPARQL,
#         params={"query": query, "format": "json"},
#         headers=HEADERS,
#         timeout=60
#     )
#     r.raise_for_status()
#     time.sleep(sleep)
#     return r.json()["results"]["bindings"]
#
#
# # ------------------------------------------------------------------
# # Label resolution with caching
# # ------------------------------------------------------------------
#
# @lru_cache(maxsize=200_000)
# def get_label(qid_or_pid):
#     """
#     Resolves a Wikidata QID or PID to its English label.
#     Cached aggressively.
#     """
#     if not qid_or_pid:
#         return None
#
#     # Property IDs (Pxxx) are entities too — same query works
#     query = f"""
#     SELECT ?label WHERE {{
#       wd:{qid_or_pid} rdfs:label ?label .
#       FILTER(LANG(?label) = "en")
#     }}
#     LIMIT 1
#     """
#
#     res = run_sparql(query)
#     if res:
#         return res[0]["label"]["value"]
#     return qid_or_pid  # fallback
#
#
# def translate_qids(qids):
#     """
#     Batch helper (still cached internally).
#     """
#     return {qid: get_label(qid) for qid in qids}
#
#
# # ------------------------------------------------------------------
# # Pretty printer for candidate groups
# # ------------------------------------------------------------------
#
# def pretty_print_group(group, max_entities=15):
#     """
#     Prints a candidate group in a human-readable way.
#
#     Args:
#         group (dict): output from group_generator
#         max_entities (int): truncate entity list for display
#     """
#
#     print("=" * 80)
#     print(f"Domain      : {group['seed_domain']}")
#     print(f"Subdomain   : {group['seed_subdomain']}")
#     print()
#
#     print("Filters:")
#     for f in group["filters"]:
#         prop_label = get_label(f["property"])
#         val_label = get_label(f["value_qid"])
#         print(f"  - {prop_label}: {val_label}")
#
#     print()
#     print(f"Entities ({len(group['entity_qids'])} total):")
#
#     for qid in group["entity_qids"][:max_entities]:
#         print(f"  - {get_label(qid)} ({qid})")
#
#     if len(group["entity_qids"]) > max_entities:
#         print(f"  ... +{len(group['entity_qids']) - max_entities} more")
#
#     print("=" * 80)
#     print()

import os
import json
import time
import requests
from collections import defaultdict
from functools import lru_cache

# =========================
# CONFIG
# =========================

REPO_PATH = os.getenv('REPO_PATH')
if REPO_PATH is None:
    raise RuntimeError("REPO_PATH is not set in the .env file")

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "DatasetGroupGenerator/1.0 (research use)",
    "Accept": "application/sparql-results+json"
}

MIN_GROUP_SIZE = 5
MAX_GROUP_SIZE = 1000
OPTIONAL_FILTER_SAMPLE = 3      # max optional refinements per base group
REQUEST_SLEEP = 0.2             # avoid 429s


# =========================
# SPARQL UTILITIES
# =========================

def run_sparql(query):
    r = requests.get(
        WIKIDATA_SPARQL,
        params={"query": query, "format": "json"},
        headers=HEADERS
    )
    r.raise_for_status()
    return r.json()["results"]["bindings"]


@lru_cache(maxsize=50_000)
def get_label(qid):
    query = f"""
    SELECT ?label WHERE {{
      wd:{qid} rdfs:label ?label .
      FILTER(LANG(?label) = "en")
    }} LIMIT 1
    """
    res = run_sparql(query)
    time.sleep(REQUEST_SLEEP)
    return res[0]["label"]["value"] if res else qid


# =========================
# CORE DISCOVERY FUNCTIONS
# =========================

def discover_partition_values(seed, partition):
    """
    Discover distinct values for a partition property (e.g. teams, leagues)
    with group size constraints enforced at query time.
    """
    mandatory = seed["mandatory_properties"]
    prop = partition["property"]

    mandatory_patterns = []
    for m in mandatory:
        mandatory_patterns.append(
            f"?entity wdt:{m['property']} wd:{m['value_qid']} ."
        )

    query = f"""
    SELECT ?value (COUNT(?entity) AS ?cnt) WHERE {{
      {' '.join(mandatory_patterns)}
      ?entity wdt:{prop} ?value .
    }}
    GROUP BY ?value
    HAVING (?cnt >= {partition.get("min_group_size", MIN_GROUP_SIZE)} &&
            ?cnt <= {partition.get("max_group_size", MAX_GROUP_SIZE)})
    """

    results = run_sparql(query)
    time.sleep(REQUEST_SLEEP)

    return [
        {
            "value_qid": r["value"]["value"].split("/")[-1],
            "count": int(r["cnt"]["value"])
        }
        for r in results
    ]


def fetch_entities(seed, filters):
    """
    Fetch entity QIDs for a fully specified group definition.
    """
    patterns = []

    for m in seed["mandatory_properties"]:
        patterns.append(
            f"?entity wdt:{m['property']} wd:{m['value_qid']} ."
        )

    for f in filters:
        patterns.append(
            f"?entity wdt:{f['property']} wd:{f['value_qid']} ."
        )

    query = f"""
    SELECT ?entity WHERE {{
      {' '.join(patterns)}
    }}
    """

    res = run_sparql(query)
    time.sleep(REQUEST_SLEEP)

    entities = list({
        r["entity"]["value"].split("/")[-1]
        for r in res
    })

    return entities


# =========================
# GROUP ENUMERATION
# =========================

def enumerate_groups(seed):
    """
    Generator that yields one group at a time.
    Each yielded item is a dict with:
      - seed_domain
      - seed_subdomain
      - filters
      - entity_qids
    """

    group_idx = 0

    for partition in seed["partition_options"]:
        partition_prop = partition["property"]

        partition_values = discover_partition_values(seed, partition)

        for pv in partition_values:
            base_filters = [{
                "property": partition_prop,
                "name": partition["name"],
                "value_qid": pv["value_qid"]
            }]

            # ---- progress log (cheap, before entity fetch)
            desc = describe_group(seed, base_filters)
            print(f"[GROUP {group_idx:04d}] {desc}")

            entities = fetch_entities(seed, base_filters)

            if not (MIN_GROUP_SIZE <= len(entities) <= MAX_GROUP_SIZE):
                continue

            yield {
                "seed_domain": seed["domain"],
                "seed_subdomain": seed["subdomain"],
                "filters": base_filters,
                "entity_qids": entities
            }

            group_idx += 1

            # ---- optional refinements (sparse)
            for opt in seed.get("optional_filters", [])[:OPTIONAL_FILTER_SAMPLE]:
                opt_values = discover_optional_values(
                    seed, base_filters, opt
                )

                for ov in opt_values:
                    refined_filters = base_filters + [{
                        "property": opt["property"],
                        "name": opt["name"],
                        "value_qid": ov
                    }]

                    desc = describe_group(seed, refined_filters)
                    print(f"[GROUP {group_idx:04d}] {desc}")

                    entities = fetch_entities(seed, refined_filters)

                    if not (MIN_GROUP_SIZE <= len(entities) <= MAX_GROUP_SIZE):
                        continue

                    yield {
                        "seed_domain": seed["domain"],
                        "seed_subdomain": seed["subdomain"],
                        "filters": refined_filters,
                        "entity_qids": entities
                    }

                    group_idx += 1


def discover_optional_values(seed, base_filters, optional):
    """
    Discover common categorical refinements within a base group.
    """
    patterns = []

    for m in seed["mandatory_properties"]:
        patterns.append(
            f"?entity wdt:{m['property']} wd:{m['value_qid']} ."
        )

    for f in base_filters:
        patterns.append(
            f"?entity wdt:{f['property']} wd:{f['value_qid']} ."
        )

    prop = optional["property"]

    query = f"""
    SELECT ?value (COUNT(?entity) AS ?cnt) WHERE {{
      {' '.join(patterns)}
      ?entity wdt:{prop} ?value .
    }}
    GROUP BY ?value
    HAVING (?cnt >= {MIN_GROUP_SIZE})
    ORDER BY DESC(?cnt)
    LIMIT 10
    """

    res = run_sparql(query)
    time.sleep(REQUEST_SLEEP)

    return [
        r["value"]["value"].split("/")[-1]
        for r in res
    ]


# =========================
# PRETTY PRINTING
# =========================

def describe_group(seed, filters):
    parts = [seed["subdomain"].replace("_", " ")]

    for f in filters:
        parts.append(f"{f['name']}={get_label(f['value_qid'])}")

    return " | ".join(parts)


def pretty_print_group(group, max_entities=15):
    print("=" * 80)
    print(f"Domain      : {group['seed_domain']}")
    print(f"Subdomain   : {group['seed_subdomain']}")
    print("\nFilters:")

    for f in group["filters"]:
        print(
            f"  - {f['name']}: {get_label(f['value_qid'])}"
        )

    ents = group["entity_qids"]
    print(f"\nEntities ({len(ents)} total):")

    for qid in ents[:max_entities]:
        print(f"  - {get_label(qid)} ({qid})")

    if len(ents) > max_entities:
        print(f"  ... +{len(ents) - max_entities} more")

    print("=" * 80)
    print()


# =========================
# ENTRYPOINT FOR NOTEBOOK
# =========================

def load_seeds(path=os.path.join(REPO_PATH, "data_files/group_seed.json")):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
