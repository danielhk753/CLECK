"""
Generate paragraphs

Before we start:
- How to generate a counter answer? Per answer type?
- What is the paragraphs styles I want to create?
"""
import re
import json
import random
from typing import Any

import numpy as np
from dotenv.main import load_dotenv

from knowledge_maze.tools.api_tools import *


load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')

# Load the QA :
QA_FILE_PATH = f"{REPO_PATH}\data_files\\merged_dataset.json"
with open(QA_FILE_PATH, "r", encoding="utf-8") as file:
    QA_DATASET = json.load(file)


# Initialize LLM prompts:
GENERATE_ANSWER_PROMPT = """You are an expert at generating short and plausible counter-answers to questions. 
                            Your task is to provide alternative answers that are incorrect but appear valid and 
                            believable. Ensure the counter-answers are concise, contextually relevant to the question, 
                            and different from the correct answer."""
GENERATE_PARAGRAPH_PROMPT = """"""

# Create LLM API call objects:
answer_generator = gptAPI(GENERATE_ANSWER_PROMPT,
                          temperature = 0.2,
                          max_concurrent_requests = 100)
paragraph_generator = gptAPI(GENERATE_PARAGRAPH_PROMPT,
                             temperature = 0.5,
                             max_concurrent_requests = 100)


def entity_counter_answer(question: list, correct_answer: str, num_answers: int = 3) -> str:
    """  """
    pass

# TODO: Change the numerical to be based on an API call, because the numbers must make sense...
def numerical_counter_answer(question: str,
                             answer_range: tuple[float, float],
                             num_answers: int = 3) -> dict[str, str]:
    """
    Finds a numerical memory answer for a given question, generates unique
    counter-answers within the specified range, and formats the results as a dictionary.

    Args:
        question (str): The target question to find the normalized answer for.
        answer_range (tuple[float, float]): The range (inclusive) from which to generate counter-answers.
        num_answers (int, optional): The number of counter-answers to generate. Defaults to 3.

    Returns:
        dict[str, str]: A dictionary containing the memory answer and counter-answers.
    """
    # Handle infinity in the range
    lower_bound = int(answer_range[0])
    upper_bound = int(answer_range[1]) if answer_range[1] != float('inf') else lower_bound * 10

    # Find the normalized memory answer for the given question
    memory_answer = next(
        (sample['Normalized answer'] for sample in QA_DATASET if sample['Question'] == question),
        None
    )

    # Generate counter-answers
    counter_answers = []
    while len(counter_answers) < num_answers:
        candidate = random.randint(lower_bound, upper_bound)
        if candidate != memory_answer:  # Avoid duplicates with the memory answer
            counter_answers.append(candidate)

    # Create the output dictionary
    return {
        'Memory answer': str(memory_answer),
        'Counter answer 1': str(counter_answers[0]) if len(counter_answers) > 0 else None,
        'Counter answer 2': str(counter_answers[1]) if len(counter_answers) > 1 else None,
        'Distractor answer': str(counter_answers[2]) if len(counter_answers) > 2 else None
    }



def date_counter_answer(question: str,
                        num_answers: int = 3,
                        year_range: int = 10) -> dict[str, str]:
    """
    Finds K answers with the same 'Category' as the memory answer in QA_DATASET,
    generates random years close to the memory answer's year (if applicable),
    and replaces years in the original answers with these new random years.

    Args:
        question (str): The target question to find the memory answer for.
        num_answers (int, optional): Number of additional answers to find. Defaults to 3.
        year_range (int, optional): The range within which the years should be adjusted. Defaults to 10.

    Returns:
        dict[str, str]: A dictionary containing the memory answer and counter answers,
        with years replaced by randomly generated years when applicable.
    """
    # Find the memory answer
    memory_answer = next(
        (sample['Normalized answer']['parsed_value'] for sample in QA_DATASET if sample['Question'] == question),
        None
    )

    # Get the memory answer's category
    memory_category = next(
        (sample['Normalized answer']['category'] for sample in QA_DATASET if sample['Question'] == question),
        None
    )

    # Find answers with the same category
    same_category_answers = [
        sample['Normalized answer']['parsed_value']
        for sample in QA_DATASET
        if isinstance(sample.get('Normalized answer'), dict) and
           sample['Normalized answer'].get('category') == memory_category and
           sample['Normalized answer']['parsed_value'] != memory_answer
    ]

    # Extract the memory year using regex
    memory_year_match = re.search(r'\b\d{4}\b', str(memory_answer))
    memory_year = int(memory_year_match.group()) if memory_year_match else None

    # Generate K random years close to the memory year
    if memory_year:
        random_years = random.sample(
            range(memory_year - year_range, memory_year + year_range + 1), num_answers
        )

        # Replace years in the original answers with the randomly generated years
        def replace_year(answer, new_year):
            return re.sub(r'\b\d{4}\b', str(new_year), str(answer), count=1)

        replaced_answers = [
            replace_year(answer, random_years[i])
            for i, answer in enumerate(same_category_answers[:len(random_years)])
        ]
    else:
        # If no year is found, return the original answers
        replaced_answers = [
            str(answer) for answer in same_category_answers[:num_answers]
        ]

    # Ensure consistent format in the output dictionary
    result = {
        'Memory answer': str(memory_answer),
        'Counter answer 1': replaced_answers[0] if len(replaced_answers) > 0 else None,
        'Counter answer 2': replaced_answers[1] if len(replaced_answers) > 1 else None,
        'Distractor answer': replaced_answers[2] if len(replaced_answers) > 2 else None
    }

    return result


def generate_paragraph(question_1: str, question_2: str, answer: str|int, style: str) -> str:
    pass
