"""
Generates structured QA data with reasoning paths and counter answers.

This module takes clustered question-answer data as input and transforms it
into a structured dataset suitable for multi-hop reasoning QA tasks.
It partitions questions from each cluster into a main question and two
distinct reasoning paths (chains of sub-questions).

The module handles various answer types (numerical, date, entity) and
generates multiple answer variations for each sample:
    - 'Memory answer': The original, correct answer.
    - 'Counter answer 1/2': Plausible but incorrect answers.
    - 'Distractor answer': Another incorrect answer.

Counter and distractor answer generation relies on asynchronous helper
functions (`numerical_counter_answers`, `entity_counter_answers`) and
synchronous ones (`date_counter_answer`, `get_memory_answer`) imported from
`knowledge_maze.create_dataset.generation_tools`.

The final output is a list of dictionaries, each representing a complete
QA sample including the main question, different answer types, the two
reasoning paths, and an initial placeholder for distractor paragraphs.
Utility functions are provided for displaying individual samples (`display_qa_sample`)
and visualizing the distribution of reasoning path lengths across the dataset
(`display_reasoning_lengths`).

Key Functions:
    create_data_qa: Orchestrates the entire data generation process for QA samples.
    partition_questions: Splits a list of questions into a main question and two sub-lists.
    create_reasoning_path: Structures a list of questions into a reasoning path format.
    process_counter_answers: Generates counter/distractor answers for different data types.
    display_reasoning_lengths: Visualizes the distribution of path lengths.
"""
import re
import json
import random
import textwrap

import numpy as np
from tqdm import tqdm
from uuid import uuid4
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Any, List, Dict

from knowledge_maze.create_dataset.generation_tools.gen_answers import (
    get_memory_answer, date_counter_answer, numerical_counter_answers, entity_counter_answers
)


async def create_data_qa(db: list[dict],
                         output_path: str = None) -> list[dict]:
    """
    Processes the input database into a structured list of questions, answers, and reasoning paths.

    Args:
        db (list[dict]): List of dictionaries containing answers and corresponding questions.
        output_path (str, optional): The file path where the reasoning chains will be saved as
            a JSON file. Defaults to None.

    Returns:
        list[dict]: A list where each dictionary contains:
            - 'ID': Unique sample identifier.
            - 'Question': The main question.
            - 'Answer': The associated answer (or multiple answer types for some cases).
            - 'Answer type': The type of answer.
            - 'Reasoning path 1': A chain of reasoning sub-questions (partition 1).
            - 'Reasoning path 2': A chain of reasoning sub-questions (partition 2).
    """
    results = []

    for cluster in tqdm(db, desc="Creating dataset QA samples"):
        sample_id = str(uuid4())
        if len(cluster['Questions']) >= 2:  # Make sure no single question clusters are processed.

            # Get main question and questions for reasoning paths:
            main_question, path1_questions, path2_questions = partition_questions(cluster['Questions'])
            reasoning_path_1 = create_reasoning_path(main_question, path1_questions)
            reasoning_path_2 = create_reasoning_path(main_question, path2_questions)

            # Get the original answer:
            answer = process_answer(cluster['Answer'], main_question)

            # Structure the sample for the final dataset:
            result = {
                'ID': sample_id,
                'Question': main_question,
                'Answer': answer,
                'Answer type': cluster['Answer'][1],
                'Reasoning path 1': reasoning_path_1,
                'Reasoning path 2': reasoning_path_2,
                'Distractor paragraphs': []
            }
            results.append(result)

    # Create Counter answers and add them to the dataset:
    final_results = await process_counter_answers(results)

    # Save in folder:
    if output_path:
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(final_results, out_file)

    # Display reasoning path lengths distribution:
    display_reasoning_lengths(final_results)

    return final_results


def partition_questions(questions: list[str]) -> tuple[str, list[str], list[str]]:
    """Splits the questions into a main question and two reasoning paths."""
    questions = questions.copy()

    # Get main question:
    main_question = random.choice(questions)
    questions.remove(main_question)

    # Get questions for each reasoning path:
    random.shuffle(questions)
    a, b = get_unique_count_range_for_split(len(questions))
    split_point = random.randint(a, b)

    return main_question, questions[:split_point], questions[split_point:]


def get_unique_count_range_for_split(n_remaining: int,
                                     max_final_size: int = 4 ) -> Optional[Tuple[int, int]]:
    """
    Calculates the possible range for the number of *unique* elements in the first sub-list.

    Given the number of elements remaining *after* removing the one shared element,
    this function finds the minimum and maximum number of unique items that can be
    allocated to the first group while ensuring the second group also meets the
    max_final_size constraint (when the shared element is added back).

    Args:
        n_remaining: The number of unique elements to distribute (N - 1).
        max_final_size: The maximum size allowed for the final sub-lists
                         (including the shared element). Defaults to 5.

    Returns:
        A tuple (min_a_unique, max_a_unique) representing the minimum and
        maximum number of unique items that can be put in the first list,
        or None if no valid split is possible.
    """
    possibilities = []
    max_unique_items = max_final_size - 1

    # Find all valid pairs (a_unique, b_unique) first
    for a_unique in range(min(n_remaining, max_unique_items) + 1):
        b_unique = n_remaining - a_unique
        if 0 <= b_unique <= max_unique_items:
            possibilities.append((a_unique, b_unique))

    # If no valid possibilities were found, return None
    if not possibilities:
        return None

    # Extract only the counts for the first group (a_unique values)
    a_counts = [p[0] for p in possibilities]

    # Find the minimum and maximum of these counts
    min_a_unique = min(a_counts)
    max_a_unique = max(a_counts)

    return min_a_unique, max_a_unique


def create_reasoning_path(main_question: str,
                          path_questions: list[str]) -> list[dict]:
    """Generates a reasoning path as a sequence of linked questions."""
    path = []
    previous_sub_question = main_question
    for question in path_questions:
        path.append({
            'Previous question': previous_sub_question,
            'Next question': question,
            'Paragraph': str(uuid4())
        })
        previous_sub_question = question

    path.append({
        'Previous question': previous_sub_question,
        'Next question': None,
        'Paragraph': str(uuid4())
    })
    return path


def process_answer(answer: Any,
                   question: str) -> Any:
    """Processes the answer based on its type."""
    if answer[1] in ['CARDINAL', 'MONEY', 'PERCENT', 'AGE']:
        return get_memory_answer(question), answer[0]
    if answer[1] == 'DATE':
        return get_memory_answer(question), answer[0]
    return answer[0]


async def process_counter_answers(results: list[dict]) -> list[dict]:
    """Processes and updates counter-answers for numerical and Wikipedia entity types."""
    # Split data:
    general_data = []
    numerical_data = []
    date_data = []
    for sample in results:
        if sample['Answer type'] in ['CARDINAL', 'MONEY', 'PERCENT', 'AGE']:
            numerical_data.append(sample)
        elif sample['Answer type'] == 'DATE':
            date_data.append(sample)
        else:
            general_data.append(sample)

    # Process numerical data:
    print("Generating counter answers for numerical data:")
    inputs = [{
        'Real answer': data['Answer'][0]['Normalized'],
        'Fake answer range': data['Answer'][1],
        'Question': data['Question']
    } for data in numerical_data]

    num_fake_answers = await numerical_counter_answers([str(i) for i in inputs])

    for data, counter_ans in zip(numerical_data, num_fake_answers):
        data['Answer'] = {
            'Memory answer': data['Answer'][0]['Normalized'],
            'Counter answer 1': extract_number(counter_ans[0]),
            'Counter answer 2': extract_number(counter_ans[1]),
            'Distractor answer': extract_number(counter_ans[2])
        }

    # Process general data:
    print("Generating counter answers for general data:")
    inputs = [{
        'Real answer': data['Answer'],
        'Questions': get_questions(data)
    } for data in general_data]

    general_fake_answers = await entity_counter_answers([str(i) for i in inputs])

    for data, counter_ans in zip(general_data, general_fake_answers):
        data['Answer'] = {
            'Memory answer': data['Answer'],
            'Counter answer 1': counter_ans[0],
            'Counter answer 2': counter_ans[1],
            'Distractor answer': counter_ans[2]
        }

    print("Generating counter answers for date data:")
    for data in tqdm(date_data, desc="Usage cost: 0.0000$ (Estimated: 0.00$)"):
        data['Answer'] = date_counter_answer(data['Answer'])

    return general_data + numerical_data + date_data


def get_questions(sample: dict) -> list[str]:
    """Extracts the edge questions from reasoning paths."""
    reasoning_path_1 = [step['Previous question'] for step in sample['Reasoning path 1']]
    reasoning_path_2 = [step['Previous question'] for step in sample['Reasoning path 2']]
    questions = list(set(reasoning_path_1 + reasoning_path_2))
    return questions


def extract_number(text: str) -> int:
    """Extracts the first integer from a given text string."""
    match = re.search(r'\d+', text)
    return int(match.group()) if match else 0


def wrap_text(text: str, ident: str = '') -> str:
    """Wraps a given text using textwrap."""
    wrapper = textwrap.TextWrapper(width=115)
    return f'\n{ident}'.join(wrapper.wrap(text))


def display_qa_sample(sample: dict) -> None:
    """
    Displays detailed QA sample information with line wrapping using the textwrap module.

    Args:
        sample (dict): A dictionary containing QA sample data.

    Returns:
        None
    """
    original_answer = get_memory_answer(sample['Question'])
    if isinstance(original_answer, dict):
        original_answer = original_answer['Original']
    sub_questions_chain_1 = [q['Previous question'] for q in sample['Reasoning path 1']]
    sub_questions_chain_2 = [q['Previous question'] for q in sample['Reasoning path 2']]

    print('=' * 120)
    print(f"ID: {sample['ID']}")
    print(wrap_text(f"Question: {sample['Question']}", ident=' '*10))
    print(wrap_text(f"Answer (Original): {original_answer}"))
    print(f"Answer type: {sample['Answer type']}")
    print("\nAnswers:")
    for key, value in sample['Answer'].items():
        print(f"- {value} ({key})")
    print("\nSub-questions path 1:")
    for i, question in enumerate(sub_questions_chain_1):
        memory_answer = get_memory_answer(question)
        if isinstance(memory_answer, dict):
            memory_answer = memory_answer['Original']
        print(wrap_text(f" {i+1}. {question} ({memory_answer})", ident='    '))
    print("\nSub-questions path 2:")
    for i, question in enumerate(sub_questions_chain_2):
        memory_answer = get_memory_answer(question)
        if isinstance(memory_answer, dict):
            memory_answer = memory_answer['Original']
        print(wrap_text(f" {i+1}. {question} ({memory_answer})", ident='    '))
    print('=' * 120)


def display_reasoning_lengths(data: List[Dict[str, Any]]) -> None:
    """
    Generates and displays a 2D histogram (heatmap) showing the distribution
    of integer lengths for two reasoning paths ('Reasoning path 1' and
    'Reasoning path 2'). Displays counts directly on the heatmap cells.

    Args:
        data: A list of dictionaries. Each dictionary is expected to have
              at least the keys 'Reasoning path 1' and 'Reasoning path 2'.
              The values associated with these keys should be sequences
              (e.g., strings, lists) for which len() can be calculated.

    Returns:
        None. Displays a matplotlib plot.

    Raises:
        KeyError: If a dictionary in the list is missing one of the required keys.
        TypeError: If the value associated with a reasoning path key is not
                   a sequence type for which len() is defined.
        ValueError: If no valid data points could be extracted.
    """
    if not data:
        print("Input data list is empty. Cannot generate histogram.")
        return

    lengths1 = []
    lengths2 = []
    key1 = 'Reasoning path 1'
    key2 = 'Reasoning path 2'

    # --- Data Extraction ---
    for i, item in enumerate(data):
        try:
            if key1 not in item:
                raise KeyError(f"Dictionary at index {i} is missing key: '{key1}'")
            if key2 not in item:
                 raise KeyError(f"Dictionary at index {i} is missing key: '{key2}'")

            path1 = item[key1]
            path2 = item[key2]

            len1 = len(path1)
            len2 = len(path2)

            # Ensure lengths are integers (histograms work best with numeric types)
            if not isinstance(len1, int) or not isinstance(len2, int):
                 raise TypeError(f"Length calculation did not produce integers at index {i}.")

            lengths1.append(len1)
            lengths2.append(len2)

        except KeyError as e:
            print(f"Warning: Skipping item {i} due to KeyError: {e}")
            continue
        except TypeError as e:
            path1_type = type(item.get(key1, None)).__name__
            path2_type = type(item.get(key2, None)).__name__
            print(f"Warning: Skipping item {i} due to TypeError: {e}. "
                  f"'{key1}' type: {path1_type}, '{key2}' type: {path2_type}. "
                  f"Ensure values are sequences (like str or list).")
            continue

    if not lengths1:
        print("No valid data points found after processing the input list.")
        return

    # --- Data Preparation for Symmetric Histogram ---
    # Determine the overall range across both paths
    all_lengths = lengths1 + lengths2
    if not all_lengths:
        print("No lengths found after processing.")
        return
    min_len = min(all_lengths)
    max_len = max(all_lengths)

    # Define square bins covering the full range for both axes
    bins_xy = np.arange(min_len - 0.5, max_len + 1.5, 1)
    num_bins = len(bins_xy) - 1  # Number of bins along each axis

    # Calculate the standard 2D histogram counts
    counts, x_edges, y_edges = np.histogram2d(lengths1, lengths2, bins=(bins_xy, bins_xy))

    # --- Combine counts symmetrically ---
    symmetric_counts = np.zeros_like(counts)
    for i in range(num_bins):  # Index corresponding to x-axis (length1) bins
        for j in range(num_bins):  # Index corresponding to y-axis (length2) bins
            if i == j:
                symmetric_counts[i, j] = counts[i, j]
            elif i < j:
                combined = counts[i, j] + counts[j, i]
                symmetric_counts[i, j] = combined
                symmetric_counts[j, i] = combined  # Still calculate the full symmetric matrix

    # --- Mask the upper triangle (where length1 > length2) ---
    # Create a copy to modify for display
    counts_for_display = symmetric_counts.copy()
    # Iterate through the *original* matrix indices (i=x, j=y)
    for i in range(num_bins):
        for j in range(num_bins):
            # Set elements to NaN where length1 > length2 (i.e., x-index > y-index)
            # This masks the upper triangle of the conceptual (x,y) plot
            if i > j:
                counts_for_display[i, j] = np.nan

    # Transpose for standard imshow orientation (y-axis first)
    counts_for_display = counts_for_display.T

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(num_bins + 3, num_bins + 3))

    # Get the colormap and set 'bad' color (for NaN) to white or transparent
    cmap = plt.get_cmap('Blues')
    cmap.set_bad(color='white')  # Make NaN values white

    # Use imshow to display the *masked* symmetric counts
    ax.imshow(
        counts_for_display,  # Use the transposed, masked counts
        interpolation='nearest',
        origin='lower',
        cmap=cmap,  # Use the modified colormap
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect='equal'
    )

    # --- Add Text Labels ---
    max_display_count = np.nanmax(counts_for_display)
    threshold = max_display_count * 0.6 if max_display_count > 0 else 0

    for i in range(num_bins):  # Index for y-bins (rows of counts_for_display)
        for j in range(num_bins):  # Index for x-bins (cols of counts_for_display)
            count_val = counts_for_display[i, j]
            # Only add text for non-NaN and non-zero values
            if not np.isnan(count_val) and count_val > 0:
                x_center = (x_edges[j] + x_edges[j + 1]) / 2
                y_center = (y_edges[i] + y_edges[i + 1]) / 2
                text_color = 'white' if count_val > threshold else 'black'
                ax.text(x_center, y_center, f'{int(count_val)}',
                        ha='center', va='center', color=text_color, fontsize=8)

    # --- Axis Configuration ---
    axis_ticks = np.arange(min_len, max_len + 1)
    ax.set_xticks(axis_ticks)
    ax.set_yticks(axis_ticks)

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0]+1, y_edges[-1])

    # Update labels and title
    ax.set_xlabel('Path Length 1')
    ax.set_ylabel('Path Length 2')
    ax.set_title('Distribution of Path Lengths')

    plt.tight_layout()
    plt.show()
