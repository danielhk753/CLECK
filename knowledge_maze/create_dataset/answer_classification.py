"""
Module for classifying answers based on Named Entity Recognition (NER) and Wikipedia entity matching.

This module utilizes spaCy's transformer-based NER model for efficient entity extraction
and performs asynchronous Wikipedia searches for entity validation.

Classification categories:
- Wikipedia Entity (validated via Wikipedia API)
- Date (identified by spaCy NER)
- Number (identified by NER labels or regex)
- Free-form (default category)
"""
import re
import spacy
import torch
import asyncio
import aiohttp
import warnings
from tqdm.asyncio import tqdm

# Suppress all warnings:
warnings.simplefilter("ignore")

# Precompiled regex:
NUMBER_REGEX = re.compile(r"^\d+(\.\d+)?$")

# Force GPU usage if available:
if torch.cuda.is_available():
    spacy.require_gpu()
    print("Using GPU for spaCy.")
else:
    print("GPU not available, running on CPU.")

# Load spaCy's Named Entity Recognizer:
nlp = spacy.load("en_core_web_trf")


def extract_number(answer: str):
    """Extract numeric values from a string and return as int if possible, else float.

    Args:
        answer (str): The input string containing numbers.

    Returns:
        int | float | str: Extracted number as int if possible, else float, or original string if no numbers exist.
    """
    num_str = "".join(re.findall(r"\d+\.?\d*", answer))
    if not num_str:
        return answer  # No numeric content, return original string

    return int(num_str) if num_str.isdigit() else float(num_str)


async def wiki_search(entity: str) -> str | None:
    """
    Checks if an entity exists on Wikipedia using the OpenSearch API.

    Args:
        entity (str): The entity name to search.

    Returns:
        str | None: The Wikipedia title if the entity has a Wikipedia page, None otherwise.
    """
    url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={entity}&limit=1&namespace=0&format=json"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()
            return data[1][0] if data[1] else None  # Check if a page exists


async def classify_answer(answer, entities, semaphore):
    """
    Classifies an answer based on entity recognition and Wikipedia validation.

    Args:
        answer (str): The input answer to classify.
        entities (list[tuple[str, str]]): List of extracted entities (text, label).
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent Wikipedia queries.

    Returns:
        tuple[str, str]: The classification category and processed answer.
    """
    if len(answer.split(' ')) > 10:
        return "Free-form", answer

    for ent_text, ent_label in entities:
        if ent_label in {"PERSON", "ORG", "GPE", "NORP", "EVENT", "FAC", "PRODUCT", "WORK_OF_ART"}:
            async with semaphore:
                wiki_title = await wiki_search(ent_text)
                if wiki_title:
                    return "Wikipedia Entity", wiki_title
        if ent_label == "DATE" and len(answer.split(' ')) <= 3:
            clean_number = extract_number(answer)
            return "Date", clean_number
        if ent_label in {"CARDINAL", "QUANTITY", "PERCENT", "MONEY"} and len(answer.split(' ')) <= 3:
            clean_number = extract_number(answer)
            return "Number", clean_number

    # Check for number via regex:
    if NUMBER_REGEX.match(answer):
        clean_number = extract_number(answer)
        return "Number", clean_number

    async with semaphore:
        wiki_title = await wiki_search(answer)
        if wiki_title:
            return "Wikipedia Entity", wiki_title

    return "Free-form", answer  # Default case


async def classify_answers(answers: list[str], num_workers: int = 100) -> list[tuple[str, str]]:
    """
    Classifies a list of answers using NER and async Wikipedia queries.

    This function processes answers efficiently by first extracting entities in batch
    and then classifying answers asynchronously.

    Args:
        answers (list[str]): A list of answers to classify.
        num_workers (int): Maximum number of concurrent Wikipedia searches.

    Returns:
        list[tuple[str, str]]: A list of tuples containing category and answer.
    """
    semaphore = asyncio.Semaphore(num_workers)

    # Step 1: Extract entities in batch (GPU-accelerated):
    extracted_entities = []
    for doc in tqdm(nlp.pipe(answers, batch_size=128), total=len(answers), desc="Extracting Entities"):
        extracted_entities.append([(ent.text, ent.label_) for ent in doc.ents])

    # Step 2: Process answers in parallel:
    tasks = [classify_answer(ans, ents, semaphore) for ans, ents in zip(answers, extracted_entities)]
    results = await tqdm.gather(*tasks, desc="Classifying Answers")

    return list(results)
