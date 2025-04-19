"""
Generates structured QA data with reasoning paths and counter answers.

This module transforms clustered question-answer data into a format suitable
for multi-hop reasoning tasks. It focuses on creating samples containing a
main question, associated answers (including generated counterfactuals), and
two distinct reasoning paths composed of sub-questions.

Core Workflow (`create_data_qa` function):
1.  **Input:** Takes a list of pre-clustered data, where each cluster contains
    multiple related questions for a single answer.
2.  **Path Partitioning:** For each cluster, selects a main question and randomly
    partitions the remaining sub-questions into two sets using
    `partition_questions`.
3.  **Path Creation:** Structures each set of sub-questions into a linear
    reasoning path (a list of linked steps) using `create_reasoning_path`.
    Each step is assigned a unique paragraph ID placeholder.
4.  **Answer Processing:** Extracts the original ('Memory') answer and uses
    helper functions (`process_counter_answers`) to generate plausible but
    incorrect 'Counter answers' and a 'Distractor answer' based on the answer type
    (numerical, date, entity). This involves calls to LLMs or specific logic.
5.  **Output Formatting:** Assembles the main question, various answer types,
    the two reasoning paths, and other metadata into a structured dictionary
    for each sample.
6.  **Saving & Visualization:** Optionally saves the generated dataset to a JSON
    file and displays visualizations of the distribution of reasoning path
    lengths (`display_reasoning_lengths`) and question complexity per sample
    (`plot_question_distribution_by_type`).

Helper Functions:
- `partition_questions`: Splits questions into main and two paths.
- `get_unique_count_range_for_split`: Calculates valid split points for partitioning.
- `create_reasoning_path`: Formats a list of questions into path steps.
- `process_answer`: Initial processing of the original answer.
- `process_counter_answers`: Orchestrates generation of counter/distractor answers.
- `get_questions`: Extracts all unique questions from a sample's paths.
- `extract_number`: Utility to get numbers from text.
- `wrap_text`: Utility for text formatting.
- `display_qa_sample`: Prints a formatted view of a generated sample.
- `display_reasoning_lengths`: Plots a 2D histogram of path lengths.
- `plot_question_distribution_by_type`: Plots a stacked bar chart of question
  complexity (total path length) per sample, broken down by answer type.

Dependencies:
- Relies on external generation tools (`gen_answers`) for counterfactual data.
- Uses standard libraries like `json`, `random`, `re`, `uuid`, `textwrap`.
- Uses `numpy`, `matplotlib` for plotting, and `tqdm` for progress bars.
"""
import re
import json
import random
import textwrap
from collections import defaultdict, Counter

import numpy as np
from matplotlib import cm
from tqdm import tqdm
from uuid import uuid4
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Any, List, Dict

from knowledge_maze.create_dataset.generation_tools.gen_answers import (get_memory_answer,
                                                                        date_counter_answer,
                                                                        numerical_counter_answers,
                                                                        entity_counter_answers)


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
    plot_question_distribution_by_type(final_results)

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
    if answer[1] == 'NUMERICAL':
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
        if sample['Answer type'] == 'NUMERICAL':
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
    print("Generating counter answers for general & place data:")
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


def plot_question_distribution_by_type(
    data: List[Dict[str, Any]],
    type_key: str = 'Answer type',
) -> None:
    """
    Plots distribution of question counts per sample, stacked by Answer Type.

    Generates a single bar plot where the X-axis represents the number of
    questions in a sample (obtained via `num_questions_func`), and the Y-axis
    shows the frequency (number of samples). Each bar is stacked to visualize
    the contribution of different 'Answer Type' categories to that frequency.

    Args:
        data: A list of dictionaries, where each dictionary represents a sample.
        type_key: The key in each sample dictionary holding the 'Answer Type'
                  string. Defaults to 'Answer Type'.

    Returns:
        None. Displays a matplotlib plot.
    """
    cmap_name = 'tab20'

    if not isinstance(data, list) or not data:
        print("Input data must be a non-empty list of dictionaries.")
        return

    # --- 1. Aggregate Data ---
    # Structure: {num_questions: Counter({'TypeA': count, 'TypeB': count})}
    counts_by_num_questions = defaultdict(Counter)
    all_types = set()
    skipped_samples_numq = 0
    skipped_samples_type = 0
    processed_samples = 0

    print("Aggregating question counts by type...")
    for sample in data:
        try:
            num_questions = len(sample['Reasoning path 1'] + sample['Reasoning path 2'])
            # Skip if num_questions couldn't be determined or is invalid
            if num_questions is None or not isinstance(num_questions, int) or num_questions < 0:
                skipped_samples_numq += 1
                continue

            ans_type = sample.get(type_key, 'Unknown')
            if not isinstance(ans_type, str) or not ans_type: # Ensure type is valid string
                ans_type = 'Unknown'
                skipped_samples_type += 1 # Count samples with invalid/missing type

            counts_by_num_questions[num_questions][ans_type] += 1
            if ans_type != 'Unknown':
                all_types.add(ans_type)
            processed_samples += 1

        except Exception as e:
            print(f"Error processing sample: {e}. Skipping sample.")
            # Decide if this counts as skipped_samples_numq or a general error
            continue # Skip to next sample on error

    if not counts_by_num_questions:
        print("No valid data points found after processing. Cannot generate plot.")
        return

    if skipped_samples_numq > 0:
        print(f"Skipped {skipped_samples_numq} samples due to invalid/missing question count.")
    if skipped_samples_type > 0:
        print(f"Assigned 'Unknown' type to {skipped_samples_type} samples due to invalid/missing '{type_key}'.")


    # --- 2. Prepare for Plotting ---
    # Get unique question counts (x-values) and sort them
    x_values = sorted(counts_by_num_questions.keys())
    # Get unique answer types (excluding 'Unknown' unless it's the only one) and sort
    sorted_types = sorted(list(all_types))
    if 'Unknown' in counts_by_num_questions and not all_types: # If ONLY unknown type exists
         sorted_types = ['Unknown']
    elif 'Unknown' in counts_by_num_questions:
         # Optionally add 'Unknown' to the end if you want to plot it
         # sorted_types.append('Unknown')
         print("Note: 'Unknown' type counts are aggregated but not plotted by default.")
         pass


    num_categories = len(sorted_types)
    if num_categories == 0:
        print("No known answer types found to plot.")
        # Optionally, plot just the totals if counts_by_num_questions is not empty
        return

    # Create a dictionary mapping type to its counts across x_values
    # Example: type_data['PERSON'] = [0, 5, 10, 3, ...] corresponding to x_values
    type_data = {
        t: [counts_by_num_questions[x].get(t, 0) for x in x_values]
        for t in sorted_types
    }

    # Calculate total counts for labels above bars
    total_counts_per_x = [sum(counts_by_num_questions[x].values()) for x in x_values]

    # --- 3. Generate Colors ---
    try:
        colormap = cm.get_cmap(cmap_name)
        colors = colormap(np.linspace(0, 1, num_categories))
        if num_categories > 20 and cmap_name in ['tab10','tab20','tab20b','tab20c']:
            print(f"Warning: More than 20 types ({num_categories}), color distinction from '{cmap_name}' might decrease.")
    except ValueError:
         print(f"Warning: Colormap '{cmap_name}' not found. Using default 'viridis'.")
         colormap = cm.get_cmap('viridis')
         colors = colormap(np.linspace(0, 1, num_categories))

    # --- 4. Create Stacked Bar Plot ---
    print(f"Generating stacked bar plot for {processed_samples} samples...")
    fig, ax = plt.subplots(figsize=(max(8, len(x_values) * 1.6), 6)) # Adjust width based on # bars
    bottoms = np.zeros(len(x_values)) # Keep track of the bottom for stacking

    for i, type_name in enumerate(sorted_types):
        counts = np.array(type_data[type_name]) # Convert to numpy array for addition
        ax.bar(x_values, counts, label=type_name, bottom=bottoms, color=colors[i], edgecolor='grey', linewidth=0.5)
        bottoms += counts # Update bottoms for the next stack

    # --- 5. Add Labels and Final Touches ---
    ax.set_title(f'Distribution of Question Complexity\n(number of paragraphs needed for answer))')
    ax.set_xlabel("Number of Paragraphs per Sample")
    ax.set_ylabel("Number of Samples")

    # Set x-ticks to be integers representing the question counts
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_values) # Ensure labels match ticks
    ax.legend(title=type_key, bbox_to_anchor=(1.02, 1), loc='upper left') # Legend outside plot

    # Add total count labels above each stacked bar
    for i, total in enumerate(total_counts_per_x):
        if total > 0:
            ax.text(x_values[i], total, f'{total}', ha='center', va='bottom', fontsize=9)

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=(0, 0, 0.85, 1)) # Adjust layout to make space for legend
    plt.show()
