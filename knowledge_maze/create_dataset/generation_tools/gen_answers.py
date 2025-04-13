"""
Generates plausible counter-answers for question-answering tasks.

This module provides functions to generate incorrect but believable alternative
answers (counter-answers) for various types of questions (entity, numerical, date).
It utilizes pre-configured Large Language Model (LLM) API clients (`gptAPI`)
from the `knowledge_maze.tools.api_tools` module for generating entity and
numerical counter-answers asynchronously. Date counter-answers are generated
by finding similar-category dates in a preloaded dataset and modifying their years.

The module relies on a pre-existing QA dataset loaded from a specified file path
(defined by `QA_FILE_PATH`, derived from the `REPO_PATH` environment variable set
in a `.env` file) and requires API credentials configured appropriately for the
`gptAPI` tool (likely also via environment variables).

Attributes:
    REPO_PATH (str): The base path to the repository, loaded from environment variables.
    QA_FILE_PATH (str): The full path to the processed QA dataset JSON file.
    QA_DATASET (list): The loaded QA dataset, a list of dictionaries.
    GENERATE_ENT_ANSWER_PROMPT (str): The prompt template used for generating
        entity-based counter-answers via the LLM.
    GENERATE_NUM_ANSWER_PROMPT (str): The prompt template used for generating
        numerical counter-answers via the LLM.
    ent_answer_generator (gptAPI): An initialized `gptAPI` client instance for
        generating entity counter-answers.
    num_answer_generator (gptAPI): An initialized `gptAPI` client instance for
        generating numerical counter-answers.

Functions:
    get_memory_answer: Retrieves the original correct answer structure for a question.
    entity_counter_answers: Asynchronously generates entity counter-answers using an LLM.
    numerical_counter_answers: Asynchronously generates numerical counter-answers using an LLM.
    date_counter_answer: Generates date counter-answers based on dataset lookup and year manipulation.
"""
import re
import os
import json
import random
from dotenv import load_dotenv

from knowledge_maze.tools.api_tools import *

load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')
if not REPO_PATH:
    print("Warning: REPO_PATH environment variable not set.")
    REPO_PATH = "." # Example fallback


# Load the QA :
QA_FILE_PATH = os.path.join(REPO_PATH, "data_files", "processed_data.json")
with open(QA_FILE_PATH, "r", encoding="utf-8") as file:
    QA_DATASET = json.load(file)


# Initialize LLM prompts:
GENERATE_ENT_ANSWER_PROMPT = """You are an expert at generating short and plausible counter-answers to questions. 
                                Your task is to provide 3 alternative answers that are incorrect but appear valid and 
                                believable to all the given questions (e.g. 3 answers in total that can be applied to 
                                all of the given questions). Ensure the counter-answers are concise, contextually 
                                relevant to the question, and different from the correct answer. return a list of the 
                                answers."""

GENERATE_NUM_ANSWER_PROMPT = """You are an expert at generating short and plausible counter-answers to questions. 
                                Your task is to provide 3 alternative numerical integer answers that are incorrect but 
                                appear valid and believable. Ensure the counter-answers are concise, contextually 
                                relevant to the question, and different from the correct answer. Also, make sure the 
                                answers are within the given range. return a list of the answers."""


# Create LLM API call objects:
ent_answer_generator = gptAPI(GENERATE_ENT_ANSWER_PROMPT,
                              output_format='list',
                              temperature = 0.2,
                              max_concurrent_requests = 100,
                              max_rpm=4900)
num_answer_generator = gptAPI(GENERATE_NUM_ANSWER_PROMPT,
                              output_format='list',
                              temperature = 0.2,
                              max_concurrent_requests = 100,
                              max_rpm=4900)


def get_memory_answer(question):
    """
    Retrieves the normalized memory answer for a given question.

    Args:
        question (str): The question to retrieve the memory answer for.

    Returns:
        str or None: The normalized memory answer if found; otherwise, None.
        """
    memory_answer = next(
        (sample['Answer'] for sample in QA_DATASET if sample['Question'] == question),
        None
    )
    return memory_answer


async def entity_counter_answers(inputs: list[str]) -> list:
    """
    Generates a list of plausible entity-based counter-answers for the given inputs.

    Args:
        inputs (list[str]): A list of questions for which counter-answers are to be generated.

    Returns:
        list: A list of counter-answers where each input question has 3 plausible alternatives.
    """
    return await ent_answer_generator.async_run(inputs)


async def numerical_counter_answers(inputs: list[str]) -> list:
    """
    Generates a list of plausible numerical counter-answers for the given inputs.

    Args:
        inputs (list[str]): A list of questions requiring numerical counter-answers.

    Returns:
        list: A list of numerical counter-answers where each input question has 3 plausible alternatives.
    """
    return await num_answer_generator.async_run(inputs)


def date_counter_answer(raw_answer: tuple[dict, list[str]],
                        num_answers: int = 3,
                        year_range: int = 5) -> dict[str, str]:  # TODO: need to refactor and fix
    """
    Finds K answers with the same 'Category' as the memory answer in QA_DATASET,
    generates random years close to the memory answer's year (if applicable),
    and replaces years in the original answers with these new random years.

    Args:
        raw_answer: (tuple[dict, list[str]]): The full answer to be countered.
        num_answers (int, optional): Number of additional answers to find. Defaults to 3.
        year_range (int, optional): The range within which the years should be adjusted. Defaults to 10.

    Returns:
        dict[str, str]: A dictionary containing the memory answer and counter answers,
        with years replaced by randomly generated years when applicable.
    """
    # Find the memory answer
    memory_answer = raw_answer[0]

    # Get the memory answer's category
    memory_category = raw_answer[1]

    # Find answers with the same category
    same_category_answers = [
        sample['Answer']['Original']
        for sample in QA_DATASET
        if sample['Type'][0] == 'DATE' and
           sample['Type'][1] == memory_category[0] and
           sample['Answer']['Normalized'] != memory_answer['Normalized']
    ]

    # Extract the memory year using regex
    memory_year_match = re.search(r'\b\d{4}\b', str(memory_answer))
    memory_year = int(memory_year_match.group()) if memory_year_match else None

    # Generate K random years close to the memory year
    if memory_year:
        start_year = memory_year - year_range
        end_year = memory_year + year_range
        allowed_years = [year for year in range(start_year, end_year + 1) if year != memory_year]
        random_years = random.sample(allowed_years, num_answers)

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
        'Memory answer': str(memory_answer['Normalized']),
        'Counter answer 1': replaced_answers[0] if len(replaced_answers) > 0 else None,
        'Counter answer 2': replaced_answers[1] if len(replaced_answers) > 1 else None,
        'Distractor answer': replaced_answers[2] if len(replaced_answers) > 2 else None
    }

    return result
