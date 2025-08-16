"""
Utilities for sampling and transforming multi-hop question-answering datasets.

This module contains a collection of functions designed to process, filter,
and restructure a specific type of multi-hop QA dataset. The primary goal
is to create various evaluation sets from a source dataset, allowing for
targeted analysis of a model's reasoning capabilities.

The dataset is expected to have samples with two distinct reasoning paths,
distractor paragraphs, and associated metadata like answer types.

Key Functions:
- `create_test_set`: Builds a balanced evaluation set by sampling a fixed
  number of items for each unique pair of reasoning path lengths. Can also
  filter by broad answer types ('ENT' or 'NUM').
- `get_dataset_no_conflict`: Creates two separate evaluation items from a
  single source item. One item contains reasoning path 1 plus distractors,
  and the other contains path 2 plus distractors. This is useful for testing
  a model's ability to follow a single logical chain without interference.
- `get_dataset`: Creates a single evaluation item with all paragraphs
  (both paths and distractors) shuffled together. It provides a 'Paragraph Order'
  map to trace each paragraph in the shuffled context back to its original
  source and order, enabling detailed performance analysis in complex scenarios.

Constants:
- `ANSWER_TYPES`: A dictionary categorizing specific answer types into 'ENT'
  (entity-based) and 'NUM' (numerical).
"""

import random
from typing import List, Dict, Any

ANSWER_TYPES = {
    'ENT': ['PERSON', 'PLACE', 'ORG', 'EVENT', 'LANGUAGE', 'WORK_OF_ART'],
    'NUM': ['DATE', 'NUMERICAL']
}


def sample_by_exact_path_length(data, len1, len2, sample_size):
    """
    Samples `sample_size` random items from `data` where
    Reasoning path 1 length == len1 and Reasoning path 2 length == len2.

    Args:
        data (list[dict]): The dataset.
        len1 (int): Required length of Reasoning path 1.
        len2 (int): Required length of Reasoning path 2.
        sample_size (int): Number of samples to return.

    Returns:
        list[dict]: Sampled items.
    """
    # Filter to matching lengths
    filtered = [
        item for item in data
        if len(item.get("Reasoning path 1", [])) == len1
        and len(item.get("Reasoning path 2", [])) == len2
    ]

    # Sample without replacement
    if len(filtered) < sample_size:
        raise ValueError(f"Not enough samples with lengths ({len1}, {len2}) — only {len(filtered)} found.")

    return random.sample(filtered, sample_size)


def create_test_set(data, sum_per_length, answer_type=None):
    """
    Creates a test set by sampling X items from each possible (len1, len2) reasoning length pair.

    Args:
        data (list[dict]): The dataset.
        sum_per_length (int): Number of samples to take from each length pair.
        answer_type (str, optional): 'ENT' or 'NUM' to filter by answer type.

    Returns:
        list[dict]: The combined sampled test set.
    """
    # Optionally filter by answer type
    if answer_type:
        data = [
            item for item in data
            if item.get("Answer type") in ANSWER_TYPES[answer_type]
        ]

    # Find all unique (len1, len2) pairs
    length_pairs = set(
        (len(item.get("Reasoning path 1", [])), len(item.get("Reasoning path 2", [])))
        for item in data
    )

    # Create final test set
    test_set = []
    for len1, len2 in length_pairs:
        try:
            samples = sample_by_exact_path_length(
                data, len1, len2, sum_per_length
            )
            test_set.extend(samples)
        except ValueError:
            # Skip if not enough samples for this pair
            continue

    return test_set


def get_dataset_no_conflict(qa_data: List[Dict[str, Any]],
                            paragraph_data: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Transforms QA data into a dataset of non-conflicting reasoning contexts.

    For each sample in the input `qa_data`, this function generates two distinct
    output samples to create separate evaluation scenarios. The first output
    sample contains a context consisting only of the paragraphs from "Reasoning
    path 1" combined with all distractor paragraphs. The second output sample
    contains a context of "Reasoning path 2" paragraphs and the distractors.

    The paragraphs within each generated context are randomly shuffled. This is
    useful for creating evaluation sets where a model must answer a question
    using only one of the possible reasoning paths, without interference from
    the other path's supporting paragraphs.

    Args:
        qa_data (List[Dict[str, Any]]): A list of QA sample dictionaries, each
            expected to contain keys like 'ID', 'Question', 'Answer',
            'Reasoning path 1', 'Reasoning path 2', and 'Distractor paragraphs'.
        paragraph_data (Dict[str, Dict[str, str]]): A dictionary mapping paragraph
            IDs to their corresponding content, which should be a dictionary
            containing a 'Text' key.

    Returns:
        List[Dict[str, Any]]: A new list of processed dictionaries. This list
            will contain two entries for each valid sample from the input
            `qa_data`. Each dictionary will have the keys 'ID', 'Question',
            'Answer', and 'Context'.

    """
    processed_dataset = []

    for sample in qa_data:
        try:
            sample_id = sample['ID']
            question = sample['Question']
            answer = sample['Answer']

            # --- 1. Collect paragraph texts for each group ---
            path1_ids = [step['Paragraph'] for step in sample.get('Reasoning path 1', [])]
            path2_ids = [step['Paragraph'] for step in sample.get('Reasoning path 2', [])]
            distractor_ids = sample.get('Distractor paragraphs', [])

            # Fetch texts from paragraph_data, skipping any IDs that might be missing
            path1_texts = [paragraph_data[pid]['Text'] for pid in path1_ids if pid in paragraph_data]
            path2_texts = [paragraph_data[pid]['Text'] for pid in path2_ids if pid in paragraph_data]
            distractor_texts = [paragraph_data[pid]['Text'] for pid in distractor_ids if pid in paragraph_data]

            # --- 2. Create the required data outputs ---
            # Output 1: Context for Reasoning Path 1
            processed_dataset.append({
                'ID': sample_id,
                'Question': question,
                'Answer': {
                    'Memory answer': answer['Memory answer'],
                    'Counter answer': answer['Counter answer 1'],
                    'Distractor answer 1': answer['Counter answer 2'],
                    'Distractor answer 2': answer['Distractor answer']
                },
                'Context': random.sample(path1_texts + distractor_texts, len(path1_texts + distractor_texts)),
            })

            # Output 2: Context for Reasoning Path 2
            processed_dataset.append({
                'ID': sample_id,
                'Question': question,
                'Answer': {
                    'Memory answer': answer['Memory answer'],
                    'Counter answer': answer['Counter answer 2'],
                    'Distractor answer 1': answer['Counter answer 1'],
                    'Distractor answer 2': answer['Distractor answer']
                },
                'Context': random.sample(path2_texts + distractor_texts, len(path2_texts + distractor_texts)),
            })

        except KeyError as e:
            print(f"Skipping sample with ID {sample.get('ID', 'N/A')} due to missing key: {e}")
            continue

    return processed_dataset


def get_dataset(qa_data: List[Dict[str, Any]],
                paragraph_data: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Transforms QA and paragraph data.

    For each sample, this function shuffles the context paragraphs and adds a
    dictionary ('Paragraph Order') that maps each source to a list of its
    final indices. The order of indices in these lists reflects the
    original order of the paragraphs from their source.

    Args:
        qa_data: A list of QA sample dictionaries.
        paragraph_data: A dictionary mapping paragraph IDs to their text.

    Returns:
        A list of dictionaries with the 'Paragraph Order' key structured to
        preserve original ordering.
    """
    processed_dataset = []

    for sample in qa_data:
        try:
            sample_id = sample['ID']
            question = sample['Question']
            answer = sample['Answer']

            # --- 1. Collect original, ordered paragraph texts ---
            path1_ids = [step['Paragraph'] for step in sample.get('Reasoning path 1', [])]
            path2_ids = [step['Paragraph'] for step in sample.get('Reasoning path 2', [])]
            distractor_ids = sample.get('Distractor paragraphs', [])

            path1_texts = [paragraph_data[pid]['Text'] for pid in path1_ids if pid in paragraph_data]
            path2_texts = [paragraph_data[pid]['Text'] for pid in path2_ids if pid in paragraph_data]
            distractor_texts = [paragraph_data[pid]['Text'] for pid in distractor_ids if pid in paragraph_data]

            # --- 2. Create the final shuffled context ---
            all_paragraphs = path1_texts + path2_texts + distractor_texts
            shuffled_context = random.sample(all_paragraphs, len(all_paragraphs))

            # --- 3. Create a lookup map: text -> final_shuffled_index ---
            # This is the key step. It tells us where each paragraph landed.
            # NOTE: This assumes all paragraph texts are unique.
            text_to_final_index_map = {text: i for i, text in enumerate(shuffled_context)}

            # --- 4. Build the index mapping by iterating through ORIGINAL lists ---
            # This preserves the original order of paragraphs within each group.
            paragraph_order = {
                'path 1': [text_to_final_index_map[text] for text in path1_texts],
                'path 2': [text_to_final_index_map[text] for text in path2_texts],
                'distractors': [text_to_final_index_map[text] for text in distractor_texts]
            }

            # --- 5. Create the final data output ---
            processed_dataset.append({
                'ID': sample_id,
                'Question': question,
                'Answer': answer,
                'Paragraph Order': paragraph_order,
                'Context': shuffled_context,
            })

        except KeyError as e:
            print(f"Skipping sample with ID {sample.get('ID', 'N/A')} due to missing key: {e}")
            continue

    return processed_dataset
