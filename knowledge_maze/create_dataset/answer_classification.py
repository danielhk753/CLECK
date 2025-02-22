import re
import spacy
import torch
import asyncio
import warnings
import wikipedia
from tqdm.asyncio import tqdm
from dateutil.parser import parse

# Suppress all warnings
warnings.simplefilter("ignore")

# Precompiled regex
NUMBER_REGEX = re.compile(r"^\d+(\.\d+)?$")

# Force GPU usage if available
if torch.cuda.is_available():
    spacy.require_gpu()  # Tells spaCy to use GPU
    print("Using GPU for spaCy!")
else:
    print("GPU not available, running on CPU.")

# Load spaCy's Named Entity Recognizer
nlp = spacy.load("en_core_web_trf")


async def get_wikipedia_entity(answer: str) -> str | None:
    """Optimized Wikipedia entity check using search before fetching full page."""
    try:
        search_results = await asyncio.to_thread(wikipedia.search, answer, results=1)
        if search_results:
            return search_results[0]  # Return first matched title
    except:
        try:
            search_results = await asyncio.to_thread(wikipedia.search, answer, results=1, auto_suggest=False)
            if search_results:
                return search_results[0]  # Return first matched title
        except:
            return None
    return None


async def is_wikipedia_entity(answer: str) -> str | None:
    """Determine if an answer is a Wikipedia entity using NER and Wikipedia API.

    Args:
        answer (str): The answer string to check.

    Returns:
        str | None: The Wikipedia entity title if valid, otherwise None.
    """
    doc = await asyncio.to_thread(nlp, answer)  # Run spaCy parsing asynchronously
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "NORP", "EVENT"}:
            return await get_wikipedia_entity(answer)
    return None


def is_number(answer: str) -> bool:
    """Check if the answer is a number (integer, decimal, fraction).

    Args:
        answer (str): The answer string to check.

    Returns:
        bool: True if the answer is a number, otherwise False.
    """
    return bool(NUMBER_REGEX.match(answer))


def is_numeric_entity(answer: str) -> bool:
    """Check for numbers using spaCy NER.

    Args:
        answer (str): The answer string to check.

    Returns:
        bool: True if the answer is a recognized numeric entity, otherwise False.
    """
    doc = nlp(answer)
    return any(ent.label_ in {"CARDINAL", "QUANTITY", "PERCENT", "MONEY"} for ent in doc.ents)


def is_date(answer: str) -> bool:
    """Check if the answer is a date using regex and parsing.

    Args:
        answer (str): The answer string to check.

    Returns:
        bool: True if the answer is a date, otherwise False.
    """
    try:
        parse(answer, fuzzy=False)
        return True
    except:
        return False


def is_date_entity(answer: str) -> bool:
    """Check for dates using spaCy NER.

    Args:
        answer (str): The answer string to check.

    Returns:
        bool: True if the answer is a recognized date entity, otherwise False.
    """
    doc = nlp(answer)
    return any(ent.label_ == "DATE" for ent in doc.ents)


async def check_wikipedia_entities(entities: list[str], num_workers: int) -> list[str | None]:
    """Asynchronously check if extracted entities exist on Wikipedia using a semaphore."""

    semaphore = asyncio.Semaphore(num_workers)  # Limit concurrent Wikipedia requests

    async def get_wiki(entity):
        async with semaphore:
            return await is_wikipedia_entity(entity)

    tasks = [get_wiki(entity) for entity in entities]

    # tqdm progress bar for Wikipedia entity checking
    return list(await tqdm.gather(*tasks, desc="Checking Wikipedia Entities", total=len(entities)))


async def classify_answers(answers: list[str], num_workers: int = 50) -> list[tuple[str, str]]:
    """Classify a list of answers efficiently with batch entity extraction."""

    # Step 1: Extract Named Entities in Batches
    docs = list(tqdm(nlp.pipe(answers, batch_size=64), total=len(answers), desc="Extracting Entities"))

    extracted_entities = [doc.ents[0].text if doc.ents else None for doc in docs]  # First entity per answer

    # Step 2: Check Wikipedia Entities Asynchronously with a semaphore
    entity_list = [e for e in extracted_entities if e]
    wiki_results = await check_wikipedia_entities(entity_list, num_workers)

    # Step 3: Classify Each Answer
    results = []
    wiki_idx = 0  # Track Wikipedia checks

    for answer, entity in tqdm(zip(answers, extracted_entities), total=len(answers)):
        if entity:
            wiki_entity = wiki_results[wiki_idx] if wiki_idx < len(wiki_results) else None
            wiki_idx += 1
            if wiki_entity:
                results.append(("Wikipedia Entity", wiki_entity))
                continue  # Skip further checks if it's a Wikipedia entity

        if is_number(answer) or is_numeric_entity(answer):
            results.append(("Number", answer))
        elif is_date(answer) or is_date_entity(answer):
            results.append(("Date", answer))
        else:
            results.append(("Free-form", answer))

    return results

