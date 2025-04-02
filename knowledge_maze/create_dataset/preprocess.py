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
import warnings

import spacy
import torch
from typing import Any
from tqdm.asyncio import tqdm
from collections import Counter
import matplotlib.pyplot as plt

from knowledge_maze.tools.parse_answers import extract_number, extract_date

# Suppress all warnings:
warnings.simplefilter("ignore")

# Define relevant questions words:
WH_WORDS = ["What", "Who", "When", "Where", "Which", "How much", "How many", "How old"]

# Force GPU usage if available:
if torch.cuda.is_available():
    spacy.require_gpu()
else:
    print("Notice: GPU not available, running on CPU.")

# Load spaCy's Named Entity Recognizer:
nlp = spacy.load("en_core_web_trf")


def get_summary(input_data: list, output_data: list) -> None:
    """Get summary of filtering process."""
    tested = len(input_data)
    passed = len(output_data)
    print(f"{passed} out of {tested} samples have passed ({100 * passed / tested:.1f}%).")


def filter_wh_questions(dataset: list[dict]) -> list[dict]:
    """
    Filters a dataset of questions to include only those containing exactly one WH question word.

    Args:
        dataset (list[dict]): A list of dictionaries, each containing "Question" and "Answer" keys.

    Returns:
        list[dict]: A filtered list of dictionaries with "Question" and "Answer" keys,
        containing only questions with exactly one WH question word.
    """
    filtered_dataset = []

    for sample in tqdm(dataset, desc="Filtering WH questions"):
        # Count how many WH words are present in the question
        match_count = sum(1 for word in WH_WORDS if word.lower() in sample['Question'].lower())

        # Add to the filtered dataset if exactly one WH word is present
        if match_count == 1:
            filtered_dataset.append({"Question": sample['Question'], "Answer": sample['Answer']})

    get_summary(dataset, filtered_dataset)
    return filtered_dataset


def filter_ans_in_questions(dataset: list[dict]) -> list[dict]:
    """
    Filters out from a dataset samples with questions include the answer inside of them.

    Args:
        dataset (list[dict]): A list of dictionaries, each containing "Question" and "Answer" keys.

    Returns:
        list[dict]: A filtered list of dictionaries with "Question" and "Answer" keys,
        containing only questions that doesn't contain the answer within them.
    """
    filtered_dataset = []

    for sample in tqdm(dataset, desc="Filtering self containing questions"):
        # Check if the answer appears in the question
        question = sample['Question']
        answer = sample['Answer']
        norm = lambda s: re.sub(r'\s+', ' ', re.sub(r'[^\w\s-]', '', s.lower()).replace('-', ' ')).strip()
        is_subset =  set(w for w in norm(answer).split() if w) <= set(w for w in norm(question).split() if w)

        # Add to the filtered dataset if condition is met
        if not is_subset:
            filtered_dataset.append({"Question": sample['Question'], "Answer": sample['Answer']})

    get_summary(dataset, filtered_dataset)
    return filtered_dataset


def filter_long_answers(dataset: list[dict], max_length: int = 7) -> list[dict]:
    """
    Filters out from a dataset samples with questions longer than a specified number of words.

    Args:
        dataset (list[dict]): A list of dictionaries, each containing "Question" and "Answer" keys.
        max_length (int): Maximum length allowed to pass the filter.

    Returns:
        list[dict]: A filtered list of dictionaries with "Question" and "Answer" keys,
        containing only questions with length less or equal to the specified length.
    """
    filtered_dataset = []

    for sample in tqdm(dataset, desc="Filtering out long questions"):
        answer_length = len(sample['Answer'].split(' '))
        if answer_length <= max_length:
            filtered_dataset.append({"Question": sample['Question'], "Answer": sample['Answer']})

    get_summary(dataset, filtered_dataset)
    return filtered_dataset


def get_answer_types(dataset: list[dict]) -> list[dict]:
    types_dataset = []

    # Structure QA pairs as sentences for entity extraction
    sentences = [f"Question: {sample['Question']} | Answer: {sample['Answer']}" for sample in dataset]

    # Extract entities in batch (GPU-accelerated):
    extracted_entities = []
    for doc in tqdm(nlp.pipe(sentences, batch_size=1024), total=len(sentences), desc="Extracting Entities"):
        if len(doc.ents) > 0:
            extracted_entities.append((doc.ents[-1].text, doc.ents[-1].label_))
        else:
            extracted_entities.append((None, None))
    torch.cuda.empty_cache()

    for sample, entity in zip(dataset, extracted_entities):
        if sample['Answer'] == entity[0]:
            answer_type = entity[1]
        else:
            answer_type = None

        types_dataset.append(
            {
                'Question': sample['Question'],
                'Answer': sample['Answer'],
                'Type': answer_type
            }
        )

    return types_dataset


def filter_numerical(dataset: list[dict]) -> list[dict]:
    """    """
    numerical_dataset = []
    numerical_words = ['how many', 'how much', 'how old']  # Keywords related to numerical questions

    for sample in tqdm(dataset.copy(), desc='Extracting numerical samples'):

        answer = sample['Answer']
        question = sample['Question']

        # Check if any numerical keyword is in the question (case-insensitive)
        if any(word in question.lower() for word in numerical_words) and sample['Type'] in {"CARDINAL", "PERCENT", "MONEY"}:
            norm_answer = extract_number(answer)
            if norm_answer:
                if re.search(r'[$€£¥]', answer):
                    sample['Type'] = 'MONEY'
                sample['Answer'] = {'Original': answer, 'Normalized': norm_answer}
                numerical_dataset.append(sample)

    print(f"{len(numerical_dataset)} numerical samples found ({100 * len(numerical_dataset) / len(dataset):.1f}%).")

    return numerical_dataset


def filter_dates(dataset: list[dict]) -> list[dict]:
    """ """
    date_dataset = []
    date_words = ['when', 'what year', 'what month', 'what day', 'the date']  # Keywords related to date & time questions

    for sample in tqdm(dataset.copy(), desc='Extracting date & time samples'):

        answer = sample['Answer']
        question = sample['Question']

        if any([word in question.lower() for word in date_words]) and sample['Type'] == "DATE":
            norm_answer = extract_date(answer)
            if norm_answer['category'] != 'unknown':
                sample['Answer'] = {'Original': answer, 'Normalized': norm_answer['parsed_value']}
                sample['Type'] = (sample['Type'], norm_answer['category'])
                date_dataset.append(sample)

    print(f"{len(date_dataset)} date & time samples found ({100 * len(date_dataset) / len(dataset):.1f}%).")

    return date_dataset


def filter_answers(dataset: list[dict]) -> list[dict]:
    """ """
    output_dataset = []
    relevant_ent_types = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT",
                          "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"]
    for sample in tqdm(dataset.copy(), desc='Extracting rest of the relevant samples'):
        if sample['Type'] in relevant_ent_types:
            output_dataset.append(sample)

    print(f"{len(output_dataset)} relevant samples found ({100 * len(output_dataset) / len(dataset):.1f}%).")

    return output_dataset


def show_stats(data: list[dict[str, Any]], type_key: str = 'Type') -> None:
    """
    Analyzes a list of dictionaries, extracts a primary type, and generates a
    pie chart showing the distribution of these types with counts and percentages.
    Legend items include the total count for each type.

    Args:
        data: A list of dictionaries.
        type_key: The key in each dictionary that holds the type information.
                  The value can be a string (e.g., "TIME") or a tuple/list
                  where the first element is the type string
                  (e.g., ("LOCATION", "City")).
    """
    if not isinstance(data, list) or not data:
        print("Input data must be a non-empty list of dictionaries.")
        return # Return None implicitly

    type_counts = Counter()
    total_valid_items = 0

    # 1. Extract primary types and count occurrences
    skipped_items = 0
    for item in data:
        if not isinstance(item, dict):
            skipped_items += 1
            continue # Skip items that are not dictionaries

        raw_type_value = item.get(type_key) # Use .get() for safety
        primary_type = None

        if isinstance(raw_type_value, str) and raw_type_value: # Check if non-empty string
            primary_type = raw_type_value
        elif isinstance(raw_type_value, (tuple, list)) and raw_type_value: # Check if tuple/list and not empty
            if len(raw_type_value) > 0 and isinstance(raw_type_value[0], str) and raw_type_value[0]:
                 # Check if first element is a non-empty string
                 primary_type = raw_type_value[0]

        if primary_type:
            type_counts[primary_type] += 1
            total_valid_items += 1
        else:
            skipped_items += 1 # Count items skipped due to missing/invalid type

    if skipped_items > 0:
         print(f"Skipped {skipped_items} items due to missing or invalid '{type_key}' format.")

    if total_valid_items == 0:
        print(f"No valid '{type_key}' entries found in the provided data. Cannot generate chart.")
        return # Return None implicitly

    # 2. Prepare data for plotting (sort for consistent chart appearance)
    original_labels = list(type_counts.keys())
    counts = list(type_counts.values())

    # Sort by count descending
    sorted_indices = sorted(range(len(counts)), key=lambda k: counts[k], reverse=True)
    sorted_labels = [original_labels[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]

    # Create labels for the legend including counts
    legend_labels = [f"{label} ({count})" for label, count in zip(sorted_labels, sorted_counts)]

    # 3. Define the autopct function to display count and percentage
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            # Only display value if percentage is large enough to avoid clutter
            if pct < 3: # Adjust threshold as needed
                return ''
            # Show percentage only inside slice to avoid clutter with legend
            return f'{pct:.1f}%'
            # Or keep count + percentage: return f'{val}\n({pct:.1f}%)'
        return my_autopct

    # 4. Create the pie chart
    print("Generating pie chart...")
    plt.style.use('seaborn-v0_8-deep') # Use a nice style
    fig, ax = plt.subplots(figsize=(12, 8)) # Adjust figure size as needed

    wedges, texts, autotexts = ax.pie(
        sorted_counts, # Use sorted counts
        autopct=make_autopct(sorted_counts),
        startangle=90,            # Start the first slice at the top
        pctdistance=0.80,         # Position percentage text inside the slice
        textprops={'color':"w"},   # Default text color (used if labels were inside pie)
        # labels=sorted_labels, # Avoid using labels here if using legend
    )

    # Improve visibility of text inside slices
    plt.setp(autotexts, size=9, weight="bold", color="white")

    ax.set_title(f'Distribution of Types (Total Valid Items: {total_valid_items})', fontsize=14, pad=20)

    # Add a legend outside the pie chart
    ax.legend(wedges, legend_labels,
              title="Types (Count)",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1), # Position legend to the right
              fontsize='medium')

    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.show()


def preprocess(dataset: list[dict]) -> list[dict]:
    # Filter data before classification:
    dataset = filter_wh_questions(dataset)
    dataset = filter_ans_in_questions(dataset)
    dataset = filter_long_answers(dataset)

    # Classify by answer type:
    dataset = get_answer_types(dataset)
    numerical_dataset = filter_numerical(dataset)
    date_dataset = filter_dates(dataset)
    dataset = filter_answers(dataset)

    # Merge dataset and show statistics:
    dataset = dataset + date_dataset + numerical_dataset
    show_stats(dataset)

    return dataset

