import os
import json
from tqdm import tqdm
from dotenv.main import load_dotenv

import requests


# --------------------------------------------------
# Environment loading
# --------------------------------------------------

load_dotenv()

WIKIDATA_PATH = os.getenv("WIKIDATA_PATH")
if WIKIDATA_PATH is None:
    raise RuntimeError("WIKIDATA_PATH is not set in the .env file")
REPO_PATH = os.getenv('REPO_PATH')
if REPO_PATH is None:
    raise RuntimeError("REPO_PATH is not set in the .env file")

ENTITY_FILE = os.path.join(WIKIDATA_PATH, "wikidata5m_entity.txt")
RELATION_FILE = os.path.join(WIKIDATA_PATH, "wikidata5m_relation.txt")
TRIPLET_FILE = os.path.join(WIKIDATA_PATH, "wikidata5m_all_triplet.txt")


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def save_json(file_name, file):
    print(f"Saving {file_name}... ", end="")
    with open(os.path.join(REPO_PATH, "data_files", f"{file_name}.json"), "w", encoding="utf-8") as out_file:
        json.dump(file, out_file, indent=4)
    print("Done.")


# --------------------------------------------------
# Alias loader (entities / properties)
# --------------------------------------------------

def load_alias_file(path, normalize=True, lowercase=True):
    """
    Loads a Wikidata5M alias file.

    Format per line:
        <ID> \\t <alias1> \\t <alias2> ...

    Returns:
        dict[str, list[str]] : ID -> aliases
    """
    id_to_aliases = {}

    with open(path, "r", encoding="utf-8") as f:
        total = count_lines(path)
        for line in tqdm(f, total=total, desc=f"Loading aliases from {os.path.basename(path)}"):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            item_id = parts[0]
            aliases = parts[1:]

            clean_aliases = []
            for a in aliases:
                if lowercase:
                    a = a.lower()

                if normalize:
                    a = (
                        a.strip().replace("–", "-").replace("—", "-")
                    )

                if a:
                    clean_aliases.append(a)

            # Deduplicate while preserving order
            clean_aliases = list(dict.fromkeys(clean_aliases))
            id_to_aliases[item_id] = clean_aliases

    return id_to_aliases


# --------------------------------------------------
# Triplet loader
# --------------------------------------------------

def load_triplets(path):
    """
    Loads Wikidata5M triplets.

    Format per line:
        <head_id> \\t <relation_id> \\t <tail_id>

    Returns:
        list[tuple[str, str, str]]
    """
    triplets = []

    with open(path, "r", encoding="utf-8") as f:
        total = count_lines(path)
        for line in tqdm(f, total=total, desc="Loading triplets"):
            line = line.strip()
            if not line:
                continue

            h, r, t = line.split("\t")
            triplets.append((h, r, t))

    return triplets


# --------------------------------------------------
# Unified loader
# --------------------------------------------------

def load_wikidata5m():
    """
    Loads:
    - entity aliases
    - relation aliases
    - triplets

    Returns:
        entity_aliases: dict[str, list[str]]
        relation_aliases: dict[str, list[str]]
        triplets: list[tuple[str, str, str]]
    """
    print("Counting dataset sizes...")
    print("Entity aliases:", count_lines(ENTITY_FILE))
    print("Relation aliases:", count_lines(RELATION_FILE))
    print("Triplets:", count_lines(TRIPLET_FILE))
    print()

    entity_aliases = load_alias_file(ENTITY_FILE)
    relation_aliases = load_alias_file(RELATION_FILE)
    triplets = load_triplets(TRIPLET_FILE)

    save_json("entity_aliases", entity_aliases)
    save_json("relation_aliases", relation_aliases)
    save_json("triplets", triplets)

    return entity_aliases, relation_aliases, triplets


# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def build_alias_index(id_to_aliases):
    """
    Builds alias -> ID lookup.
    If multiple IDs share an alias, keeps the first encountered.
    """
    alias_to_id = {}
    for _id, aliases in id_to_aliases.items():
        for a in aliases:
            if a not in alias_to_id:
                alias_to_id[a] = _id
    return alias_to_id


def get_entity_code(name, entity_alias_index, lowercase=True):
    if lowercase:
        name = name.lower()
    return entity_alias_index.get(name)


def get_property_code(name, relation_alias_index, lowercase=True):
    if lowercase:
        name = name.lower()
    return relation_alias_index.get(name)


def get_entity_properties(entity_code, triplets, relation_aliases):
    """
    Returns a sorted list of property names (canonical alias)
    for which the entity appears as the head.
    """
    property_codes = set()

    for h, r, _ in triplets:
        if h == entity_code and r not in ["P31", "P279"]:
            property_codes.add(r)

    property_names = []
    for r in property_codes:
        aliases = relation_aliases.get(r)
        if aliases:
            property_names.append(aliases[0])  # canonical name

    return sorted(property_names)


def get_entities_of_type(type_qid, triplets, include_subclasses=False):
    """
    Returns a set of entity QIDs that are instances of the given type.

    Args:
        type_qid (str): e.g. "Q39201" (Dog breed)
        triplets (list): (head, relation, tail)
        include_subclasses (bool): whether to include subclass-of chains

    Returns:
        set[str]
    """
    instance_of = "P31"
    subclass_of = "P279"

    entities = set()
    subclasses = {type_qid}

    if include_subclasses:
        # First pass: collect subclasses
        changed = True
        while changed:
            changed = False
            for h, r, t in triplets:
                if r == subclass_of and t in subclasses and h not in subclasses:
                    subclasses.add(h)
                    changed = True

    for h, r, t in triplets:
        if r == instance_of and t in subclasses:
            entities.add(h)

    return entities


def get_entities_with_value(value_qid, triplets):
    """
    Returns all entities that have *any* property whose value is value_qid.

    Args:
        value_qid (str): e.g. "Q39201" (Dog breed)
        triplets (list): (head, relation, tail)

    Returns:
        set[str]
    """
    entities = set()

    for h, _, t in triplets:
        if t == value_qid:
            entities.add(h)

    return entities


WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"


def top_incoming_properties_with_percent(
    entity_qid: str,
    sample_limit: int = 100,
    top_k: int = 3,
    timeout: int = 60
):
    """
    Finds the top-K most common direct Wikidata properties P such that:
        Ei --wdt:P--> entity_qid

    Returns each property with its percentage of the sampled incoming edges.

    Args:
        entity_qid (str): Target entity QID
        sample_limit (int): Number of incoming edges to sample
        top_k (int): Number of top properties to return
        timeout (int): HTTP timeout

    Returns:
        list of dicts:
        [
          {
            "property_pid": "P106",
            "property_label": "occupation",
            "count": 742,
            "percent": 24.7
          },
          ...
        ]
    """

    # Step 1: discover top-K properties + counts
    discovery_query = f"""
    SELECT ?property (COUNT(*) AS ?count) WHERE {{
      {{
        SELECT ?property WHERE {{
          ?entity ?property wd:{entity_qid} .
          FILTER(STRSTARTS(STR(?property), "http://www.wikidata.org/prop/direct/"))
        }}
        LIMIT {sample_limit}
      }}
    }}
    GROUP BY ?property
    ORDER BY DESC(?count)
    LIMIT {top_k}
    """

    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "wikidata-analysis/1.0"
    }

    r = requests.get(
        WIKIDATA_SPARQL,
        params={"query": discovery_query},
        headers=headers,
        timeout=timeout
    )
    r.raise_for_status()
    bindings = r.json()["results"]["bindings"]

    if not bindings:
        return []

    total_sampled = sum(int(b["count"]["value"]) for b in bindings)

    pids = []
    counts = {}
    for b in bindings:
        pid = b["property"]["value"].split("/")[-1]
        count = int(b["count"]["value"])
        pids.append(pid)
        counts[pid] = count

    # Step 2: resolve property labels (batched)
    values = " ".join(f"wd:{pid}" for pid in pids)

    label_query = f"""
    SELECT ?property ?propertyLabel WHERE {{
      VALUES ?property {{ {values} }}
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en" .
      }}
    }}
    """

    r = requests.get(
        WIKIDATA_SPARQL,
        params={"query": label_query},
        headers=headers,
        timeout=timeout
    )
    r.raise_for_status()
    label_bindings = r.json()["results"]["bindings"]

    labels = {
        b["property"]["value"].split("/")[-1]: b["propertyLabel"]["value"]
        for b in label_bindings
    }

    # Step 3: assemble result
    results = []
    for pid in pids:
        count = counts[pid]
        percent = round((count / total_sampled) * 100, 2)

        results.append({
            "property_pid": pid,
            "property_label": labels.get(pid, pid),
            "count": count,
            "percent": percent
        })

    return results
