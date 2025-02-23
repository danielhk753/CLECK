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
import json
import spacy
import torch
import asyncio
import aiohttp
import warnings
import numpy as np
from tqdm.asyncio import tqdm
import matplotlib.pyplot as plt

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
    for doc in tqdm(nlp.pipe(answers, batch_size=64), total=len(answers), desc="Extracting Entities"):
        extracted_entities.append([(ent.text, ent.label_) for ent in doc.ents])

    # Step 2: Process answers in parallel:
    tasks = [classify_answer(ans, ents, semaphore) for ans, ents in zip(answers, extracted_entities)]
    results = await tqdm.gather(*tasks, desc="Classifying Answers")

    return list(results)


async def classify_dataset_answers(dataset: list[dict], output_path: str | None = None) -> list[dict]:
    """classify dataset - return the dataset with types (and normalized answer)"""
    dataset_questions = [sample['Question'] for sample in dataset]
    dataset_answers = [sample['Answer'] for sample in dataset]
    try:
        dataset_types = [sample['Type'] for sample in dataset]
    except:
        dataset_types = [None for sample in dataset]

    # Classify answers:
    dataset_answer_classes = await classify_answers(dataset_answers, num_workers=100)

    # Insert new information to the dataset:
    output_dataset = []
    for question, answer, og_type, (pred_type, extracted_answer) in zip(dataset_questions,
                                                                        dataset_answers,
                                                                        dataset_types,
                                                                        dataset_answer_classes):
        if og_type == 'WikipediaEntity' and pred_type == 'Free-form':
            answer_type = og_type
            norm_answer = answer
        else:
            answer_type = pred_type
            norm_answer = extracted_answer

        output_dataset.append({
            'Question': question,
            'Answer': answer,
            'Answer type': answer_type,
            'Normalized answer': norm_answer
        })

    # Save processed data if output path is given:
    if output_path:
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(output_dataset, out_file)

    return output_dataset


def get_stats(datasets: dict[list[dict]]) -> None:
    """Visualize results"""
    # Prepare data
    total_samples = 0
    relevant_samples = 0
    for dataset_name, data in datasets.items():
        pass


def create_layered_pie_chart(data) -> None:
    # Define data for the inner and outer rings
    inner_labels = ["Wikipedia Entities", "Numerical Values", "Time & Date"]
    inner_sizes = [50, 30, 20]

    # Define colors
    colors_inner = ["#66b3ff", "#ff9999", "#99ff99"]
    colors_outer = ["#c2d6ff", "#99c2ff", "#c2d6ff", "#ffb3b3", "#ff6666", "#ffb3b3", "#b3ffb3", "#66ff66", "#b3ffb3"]

    outer_labels = ["SQuAD", "Natural Questions", "TriviaQA", "SQuAD", "Natural Questions", "TriviaQA", "SQuAD",
                    "Natural Questions", "TriviaQA"]
    outer_sizes = [30, 15, 5, 15, 5, 10, 8, 8, 4]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot inner ring (larger groups)
    wedges_inner, _ = ax.pie(inner_sizes, radius=0.7, labels=None, colors=colors_inner,
                             wedgeprops=dict(width=0.5, edgecolor='white'))

    # Plot outer ring (smaller subcategories)
    wedges_outer, _ = ax.pie(outer_sizes, radius=1.2, labels=None, colors=colors_outer,
                             wedgeprops=dict(width=0.5, edgecolor='white'))

    # Function to add labels and values inside the wedges
    def add_labels(wedges, labels, sizes, radius_factor=0.85):
        for wedge, label, size in zip(wedges, labels, sizes):
            angle = (wedge.theta2 + wedge.theta1) / 2  # Find center angle
            x = radius_factor * np.cos(np.radians(angle))  # X position
            y = radius_factor * np.sin(np.radians(angle))  # Y position
            ax.text(x, y, f"{label}: {size}", ha='center', va='center', fontsize=6, color='black', fontweight="bold")

    # Add labels & values inside the rings
    add_labels(wedges_inner, inner_labels, inner_sizes, radius_factor=0.5)
    add_labels(wedges_outer, outer_labels, outer_sizes, radius_factor=1.4)

    # Ensure the chart is circular
    ax.set(aspect="equal")

    plt.show()