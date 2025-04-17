"""
Dataset preprocessing module for filtering and classifying question-answer pairs.

This module provides functions to preprocess datasets containing question-answer
pairs. It includes functionality for:
- Filtering questions based on the presence of WH words.
- Filtering samples where the answer is contained within the question.
- Filtering samples based on the length of the answer.
- Identifying the entity type of answers using spaCy NER.
- Filtering and normalizing numerical answers.
- Filtering and normalizing date/time answers.
- Selecting samples with specific relevant entity types.
- Visualizing the distribution of answer types using a pie chart.
- Running a complete preprocessing pipeline on a dataset.

The module leverages spaCy for Named Entity Recognition (NER), automatically
utilizing GPU if available via `torch` and `spacy.require_gpu()`. Warnings are
suppressed for cleaner output during processing.
"""
import re
import json
import warnings

import spacy
import torch
import numpy as np
from typing import Any
import matplotlib.cm as cm
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
    """
    Processes a dataset of question-answer pairs to determine the entity type of each answer.

    This function structures question-answer pairs into sentences for entity extraction using a
    language model. Entities are extracted in batch mode (GPU-accelerated if available), and the
    entity type of each answer is identified. The function outputs a dataset with the original
    questions and answers, along with their corresponding answer types.

    Args:
        dataset (list[dict]): A list of dictionaries containing "Question" and "Answer" keys.

    Returns:
        list[dict]: A list of dictionaries where each dictionary includes:
            - 'Question': The original question string.
            - 'Answer': The original answer string.
            - 'Type': The entity type of the answer, or None if no matching entity type is found.
    """
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
            if answer_type in ['GPE', 'LOC', 'FAC']:
                answer_type = 'PLACE'
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
    """
    Filters a dataset to extract numerical samples based on keywords and entity types.

    This function scans a dataset of question-answer pairs and identifies samples that match
    specific numerical criteria. It checks for numerical keywords in the question and ensures
    the answer corresponds to specific entity types such as "CARDINAL", "PERCENT", or "MONEY".
    If a valid numerical answer is found, it normalizes the answer and categorizes its type.

    Args:
        dataset (list[dict]): A list of dictionaries containing "Question", "Answer", and "Type".

    Returns:
        list[dict]: A filtered list of dictionaries where each entry contains:
            - 'Question': The original question.
            - 'Answer': A dictionary with the original and normalized answer.
            - 'Type': The entity type (e.g., "MONEY", "CARDINAL").
    """
    numerical_dataset = []
    numerical_words = ['how many', 'how much', 'how old']  # Keywords related to numerical questions

    for sample in tqdm(dataset.copy(), desc='Extracting numerical samples'):
        answer = sample['Answer']
        question = sample['Question']

        # Check for numerical keywords and valid entity types
        if any(word in question.lower() for word in numerical_words) and sample['Type'] in {"CARDINAL", "PERCENT", "MONEY"}:
            norm_answer = extract_number(answer)
            sample['Type'] = 'NUMERICAL'
            if isinstance(norm_answer, int):
                sample['Answer'] = {'Original': answer, 'Normalized': norm_answer}
                numerical_dataset.append(sample)

    print(f"{len(numerical_dataset)} numerical samples found ({100 * len(numerical_dataset) / len(dataset):.1f}%).")

    return numerical_dataset


def filter_dates(dataset: list[dict]) -> list[dict]:
    """
    Filters a dataset to extract date-related samples based on keywords and entity types.

    This function processes a dataset of question-answer pairs, identifying samples that match
    specific date-related criteria. It checks for keywords in the question and ensures the answer
    corresponds to the "DATE" entity type. If a valid date is found, the answer is normalized
    and its type is further categorized.

    Args:
        dataset (list[dict]): A list of dictionaries containing "Question", "Answer", and "Type".

    Returns:
        list[dict]: A filtered list of dictionaries where each entry includes:
            - 'Question': The original question.
            - 'Answer': A dictionary with the original and normalized answer.
            - 'Type': A tuple containing the original type and its sub-category (e.g., "DATE", "day", "year").
    """
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
    """
    Filters a dataset to extract samples with relevant entity types.

    This function processes a dataset of question-answer pairs and identifies samples whose
    entity type belongs to a predefined list of relevant types. The filtered samples are
    returned in a new dataset.

    Args:
        dataset (list[dict]): A list of dictionaries containing "Question", "Answer", and "Type".

    Returns:
        list[dict]: A filtered list of dictionaries where each entry has an entity type included
        in the predefined list of relevant types.
    """
    output_dataset = []
    relevant_ent_types = ["PERSON", "PLACE", "ORG", "EVENT", "WORK_OF_ART", "LANGUAGE"]
    for sample in tqdm(dataset.copy(), desc='Extracting rest of the relevant samples'):
        if sample['Type'] in relevant_ent_types:
            if sample['Type'] == "PERSON":
                if len(sample['Answer'].split(' ')) >= 2: # only full names of people, any other variation might be ambiguous.
                    output_dataset.append(sample)
            else:
                output_dataset.append(sample)

    print(f"{len(output_dataset)} relevant samples found ({100 * len(output_dataset) / len(dataset):.1f}%).")

    return output_dataset


def show_stats(data: list[dict[str, Any]],
               type_key: str = 'Type') -> None:
    """
    Analyzes item types, plots distribution as a pie chart with unique colors.

    Counts the occurrences of primary types extracted from a list of dictionaries
    using the specified `type_key`. Handles cases where the type is a direct
    string or the first element of a list/tuple. Generates and displays a
    pie chart showing the distribution, with percentages on slices and counts
    in the legend. Ensures distinct colors for slices using a specified colormap.

    Args:
        data: A list of dictionaries to analyze.
        type_key: The key in each dictionary holding the type information.
                  The value can be a string or a tuple/list where the first
                  element is the primary type string.

    Returns:
        None. Displays a matplotlib plot.
    """
    if not isinstance(data, list) or not data:
        print("Input data must be a non-empty list of dictionaries.")
        return

    cmap_name = 'tab20'

    # --- 1. Extract and Count Types ---
    type_counts = Counter()
    skipped_items = 0
    for item in data:
        primary_type = None # Define type for clarity

        # --- Inline Type Extraction Logic ---
        if isinstance(item, dict):
            raw_type_value = item.get(type_key)
            if isinstance(raw_type_value, str) and raw_type_value:
                primary_type = raw_type_value
            elif isinstance(raw_type_value, (tuple, list)) and raw_type_value:
                if len(raw_type_value) > 0 and isinstance(raw_type_value[0], str) and raw_type_value[0]:
                    primary_type = raw_type_value[0]
        # --- End Inline Type Extraction ---

        if primary_type:
            type_counts[primary_type] += 1
        else:
            # Item was not a dict, or type_key missing, or type format invalid/empty
            skipped_items += 1

    if skipped_items > 0:
         print(f"Skipped {skipped_items} items due to missing item dict structure or invalid/missing '{type_key}'.")

    if not type_counts: # Check if the Counter is empty
        print(f"No valid '{type_key}' entries found. Cannot generate chart.")
        return

    total_valid_items = sum(type_counts.values())

    # --- 2. Prepare Data for Plotting (Sorted by Count) ---
    sorted_items = type_counts.most_common()
    sorted_labels = [item[0] for item in sorted_items]
    sorted_counts = [item[1] for item in sorted_items]
    num_types = len(sorted_labels)

    # Create labels for the legend including counts
    legend_labels = [f"{label} ({count})" for label, count in sorted_items]

    # --- 3. Generate Unique Colors ---
    try:
        colormap = cm.get_cmap(cmap_name)
        colors = colormap(np.linspace(0, 1, num_types))
        if num_types > 20 and cmap_name in ['tab10','tab20','tab20b','tab20c']:
            print(f"Warning: More than 20 categories ({num_types}), color distinction from '{cmap_name}' might decrease.")
    except ValueError:
         print(f"Warning: Colormap '{cmap_name}' not found. Using default 'viridis'.")
         colormap = cm.get_cmap('viridis')
         colors = colormap(np.linspace(0, 1, num_types))

    # --- 4. Define autopct Function ---
    def make_autopct(values):
        """Helper function to display percentages on pie slices."""
        def my_autopct(pct):
            if pct is None or total_valid_items == 0: return ''
            return f'{pct:.1f}%'
        return my_autopct

    # --- 5. Create Pie Chart ---
    print(f"Generating pie chart for {num_types} types...")
    plt.style.use('seaborn-v0_8-deep')
    fig, ax = plt.subplots(figsize=(12, 8))

    wedges, texts, autotexts = ax.pie(
        sorted_counts,
        labels=None,
        autopct=make_autopct(sorted_counts),
        startangle=90,
        pctdistance=0.80,
        colors=colors,
        wedgeprops={'edgecolor': 'white'}
    )

    plt.setp(autotexts, size=9, weight="bold", color="white")

    ax.set_title(f'Distribution of Relevant Types (Total Valid Items: {total_valid_items})', fontsize=14, pad=20)

    ax.legend(wedges, legend_labels,
              title="Types (Count)",
              loc="center left",
              bbox_to_anchor=(1.05, 0.5),
              fontsize='medium')

    plt.tight_layout(rect=(0, 0, 0.8, 1))
    plt.show()


def preprocess(dataset: list[dict], output_path: str | None = None) -> list[dict]:
    """
    Preprocesses a dataset by filtering, classifying, and saving question-answer pairs.

    This function performs multiple preprocessing steps on a dataset of question-answer pairs:
    it applies filters, classifies the data by answer type, merges the classified subsets, and
    generates statistics. Optionally, the processed dataset can be saved to a specified file path.

    Args:
        dataset (list[dict]): A list of dictionaries containing question-answer pairs.
        output_path (str | None): The file path for saving the processed dataset. If None, the data
                                  is not saved.

    Returns:
        list[dict]: The fully processed dataset containing filtered and classified question-answer pairs.
    """
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

    # Save processed data if output path is given:
    if output_path:
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(dataset, out_file)

    return dataset
