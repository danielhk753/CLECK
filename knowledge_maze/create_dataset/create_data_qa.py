"""
---
"""
import re
import random

from tqdm import tqdm
from uuid import uuid4
from typing import Tuple, Optional, Any

from knowledge_maze.create_dataset.generation_tools.gen_answers import (
    get_memory_answer, date_counter_answer, numerical_counter_answers, entity_counter_answers
)


async def create_data_qa(db: list[dict]) -> list[dict]:
    """
    Processes the input database into a structured list of questions, answers, and reasoning paths.

    Args:
        db (list[dict]): List of dictionaries containing answers and corresponding questions.

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
                'Reasoning path 2': reasoning_path_2
            }
            results.append(result)

    # Create Counter answers and add them to the dataset:

    final_results = await process_counter_answers(results)

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
    if answer[1] in ['CARDINAL', 'MONEY', 'PERCENT']:
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
        if sample['Answer type'] in ['CARDINAL', 'MONEY', 'PERCENT']:
            numerical_data.append(sample)
        elif sample['Answer type'] == 'DATE':
            date_data.append(sample)
        else:
            general_data.append(sample)

    # Process numerical data:
    print("Generating counter answers for numerical data:")
    inputs = [{
        'Real answer': data['Answer'][0],
        'Fake answer range': data['Answer'][1],
        'Question': data['Question']
    } for data in numerical_data]

    num_fake_answers = await numerical_counter_answers([str(i) for i in inputs])

    for data, counter_ans in zip(numerical_data, num_fake_answers):
        data['Answer'] = {
            'Memory answer': data['Answer'][0],
            'Counter answer 1': extract_number(counter_ans[0]),
            'Counter answer 2': extract_number(counter_ans[1]),
            'Distractor answer': extract_number(counter_ans[2])
        }

    # Process general data:
    print("Generating counter answers for general data:")
    inputs = [{
        'Real answer': data['Answer'],
        'Questions': get_edge_questions(data)
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
    for data in date_data:
        pass
        # data['Answer'] = date_counter_answer(data['Answer'])   # TODO: Fix this

    return general_data + numerical_data + date_data


def get_edge_questions(sample: dict) -> list[str]:
    """Extracts the edge questions from reasoning paths."""
    return [
        sample['Question'],
        sample['Reasoning path 1'][-1]['Previous question'],
        sample['Reasoning path 2'][-1]['Previous question']
    ]


def extract_number(text: str) -> int:
    """Extracts the first integer from a given text string."""
    match = re.search(r'\d+', text)
    return int(match.group()) if match else 0
