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
import matplotlib.colors as mcolors

# Suppress all warnings:
warnings.simplefilter("ignore")

# Precompiled regex:
NUMBER_REGEX = re.compile(r"^\d+(\.\d+)?$")

# Force GPU usage if available:
if torch.cuda.is_available():
    spacy.require_gpu()
else:
    print("Notice: GPU not available, running on CPU.")

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

    try:
        return int(num_str) if num_str.isdigit() else float(num_str)
    except:
        return answer


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
            try:
                data = await response.json()
                return data[1][0] if data[1] else None  # Check if a page exists
            except:
                return None


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


async def classify_answers(answers: list[str], num_workers: int = 50) -> list[tuple[str, str]]:
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
    """
    Classifies answers in a dataset and returns the dataset with answer types and normalized answers.

    Args:
        dataset (list[dict]): A list of dictionaries containing questions and answers.
        output_path (str | None, optional): File path to save the classified dataset. Defaults to None.

    Returns:
        list[dict]: The dataset with classified answer types and normalized answers.
    """
    dataset_questions = [sample['Question'] for sample in dataset]
    dataset_answers = [sample['Answer'] for sample in dataset]
    try:
        dataset_types = [sample['Type'] for sample in dataset]
    except:
        dataset_types = [None for sample in dataset]

    # Classify answers:
    dataset_answer_classes = await classify_answers(dataset_answers, num_workers=50)

    # Insert new information to the dataset:
    output_dataset = []
    for question, answer, og_type, (pred_type, extracted_answer) in zip(dataset_questions,
                                                                        dataset_answers,
                                                                        dataset_types,
                                                                        dataset_answer_classes):
        if og_type == 'WikipediaEntity' and pred_type == 'Free-form':
            answer_type = 'Wikipedia Entity'
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


def merge_datasets(datasets: dict[list[dict]], output_path: str | None = None, verbose: bool = False) -> list[dict]:
    """
    Merges multiple datasets, filters out 'Free-form' answers, and visualizes results.

    Args:
        datasets (dict[list[dict]]): A dictionary mapping dataset names to lists of Q-A samples.
        output_path (str | None, optional): File path to save the merged dataset. Defaults to None.
        verbose (bool, optional): Whether to visualize the merged dataset with a pie chart. Defaults to False.

    Returns:
        list[dict]: The merged dataset containing non-'Free-form' answers.
    """
    # Merge dataset:
    merged_dataset = []
    total_samples = 0
    relevant_samples = 0
    for dataset_name, data in datasets.items():
        for sample in data:
            if sample['Answer type'] != 'Free-form':
                merged_dataset.append(sample)
                relevant_samples += 1
            total_samples += 1

    print(f'Merged dataset contains {relevant_samples} Q-A pairs out of original '
          f'{total_samples}({round(100 * relevant_samples / total_samples)}%).')

    # Visualize answer classification results:
    if verbose:
        create_layered_pie_chart(datasets)

    # Save processed data if output path is given:
    if output_path:
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(merged_dataset, out_file)

    return merged_dataset


def create_layered_pie_chart(data: dict[list[dict]]) -> None:
    """
    Creates a layered pie chart showing the distribution of answer types.

    The inner ring represents high-level answer categories, while the outer ring
    shows subcategories corresponding to datasets.

    Args:
        data (dict[list[dict]]): A dictionary where keys are dataset names and values
                                 are lists of Q-A samples with classified answer types.
    """
    # Define categories:
    inner_labels = ["Wikipedia Entity", "Number", "Date", "Free-form"]
    inner_sizes = [0] * len(inner_labels)
    outer_labels = []
    outer_sizes = []

    # Collect outer labels and distribute sizes:
    for dataset_name, content in data.items():
        outer_labels.append(dataset_name)
        dataset_inner_sizes = [0] * len(inner_labels)
        for sample in content:
            answer_type = sample['Answer type']
            dataset_inner_sizes[inner_labels.index(answer_type)] += 1
            inner_sizes[inner_labels.index(answer_type)] += 1
        outer_sizes.append(dataset_inner_sizes)

    outer_sizes = [x for col in zip(*outer_sizes) for x in col]
    outer_labels *= len(inner_labels)  # Repeat dataset names for each category

    # Define colors to keep inner and outer rings aligned:
    base_colors = {
        "Wikipedia Entity": "#66b3ff",
        "Number": "#ff9999",
        "Date": "#99ff99",
        "Free-form": "#ffcc99",
    }

    # Generate slightly different shades for outer slices
    color_variations = lambda base_color, num_shades: [
        mcolors.to_hex(mcolors.hsv_to_rgb((mcolors.rgb_to_hsv(mcolors.to_rgb(base_color)) + np.array([0, 0.2 * i, 0]))))
        for i in range(num_shades)
    ]

    outer_color_map = []
    for label in inner_labels:
        outer_color_map.extend(color_variations(base_colors[label], len(data)))

    inner_colors = [base_colors[label] for label in inner_labels]
    outer_colors = outer_color_map

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot inner ring (larger groups)
    wedges_inner, _ = ax.pie(inner_sizes, radius=0.7, labels=None, colors=inner_colors,
                             wedgeprops=dict(width=0.5, edgecolor='white'))

    # Plot outer ring (smaller subcategories)
    wedges_outer, _ = ax.pie(outer_sizes, radius=1.2, labels=None, colors=outer_colors,
                             wedgeprops=dict(width=0.5, edgecolor='white'))

    # Function to add labels and values inside the wedges:
    def add_labels(wedges, labels, sizes, radius_factor, fontsize=7):
        """
        Adds labels and values inside the pie chart wedges.

        Args:
            wedges (list): List of wedge objects from matplotlib.
            labels (list): Corresponding labels for the wedges.
            sizes (list): Corresponding sizes (values) for the wedges.
            radius_factor (float): Distance factor from the center for label placement.
            fontsize (int, optional): Font size for the labels. Defaults to 7.
        """
        for wedge, label_, size in zip(wedges, labels, sizes):
            angle = (wedge.theta2 + wedge.theta1) / 2  # Find the center angle of the slice
            x = radius_factor * np.cos(np.radians(angle))  # X position
            y = radius_factor * np.sin(np.radians(angle))  # Y position

            # Adjust rotation angle for better readability
            rotation = angle if (angle < 90 or angle > 270) else angle - 180  # Keep text upright
            ha = "left" if (angle < 90 or angle > 270) else "right"  # Align to left or right
            ax.text(x, y, f"{label_}: {size}", ha=ha, va="center", fontsize=fontsize,
                    rotation=rotation, rotation_mode="anchor", color='black', fontweight="bold")

    # Add labels aligned to slices
    add_labels(wedges_inner, inner_labels, inner_sizes, radius_factor=0.2)
    add_labels(wedges_outer, outer_labels, outer_sizes, radius_factor=0.8)

    # Add title:
    ax.set_title('Answer type distribution in the merged dataset')

    # Ensure the chart is circular:
    ax.set(aspect="equal")

    plt.show()
