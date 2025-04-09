"""
Generate paragraphs

"""
import re
import json
import random
from dotenv import load_dotenv

from knowledge_maze.tools.api_tools import *


load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')

# Load the QA :
QA_FILE_PATH = f"{REPO_PATH}\data_files\\extracted_data.json"
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

MIDDLE_PARAGRAPH_PROMPT = """Write a well-formed paragraph in the style of Wikipedia that conveys, without explicitly 
                             stating, that the following two questions share the same answer. The paragraph should 
                             be self-contained, meaning the reader should be able to infer that both questions have 
                             the same answer without requiring external knowledge. Do not mention the questions 
                             explicitly—rather, incorporate their key details naturally within the paragraph. 
                             *Crucially, do not include the answer itself anywhere in the paragraph*.\n\n
                             Ensure the paragraph flows naturally, avoids unnatural phrasing, and maintains a 
                             coherent narrative. The writing should be engaging while subtly guiding the reader 
                             toward recognizing the shared answer."""

ANSWER_PARAGRAPH_PROMPT = """Write a well-formed paragraph in the style of Wikipedia that answers the given question 
                             with the given answer. The paragraph should be self-contained, meaning the reader should 
                             be able to infer that both questions have the same answer without requiring external 
                             knowledge. Do not mention the questions explicitly—rather, incorporate their key details 
                             naturally within the paragraph. \n\n
                             Ensure the paragraph flows naturally, avoids unnatural phrasing, and maintains a 
                             coherent narrative. The writing should be engaging while subtly guiding the reader 
                             toward recognizing the shared answer."""

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
middle_paragraph_generator = gptAPI(MIDDLE_PARAGRAPH_PROMPT,
                                    temperature = 0.5,
                                    max_concurrent_requests = 100,
                                    max_rpm=4900)
edge_paragraph_generator = gptAPI(ANSWER_PARAGRAPH_PROMPT,
                                  temperature = 0.5,
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
        (sample['Normalized answer'] for sample in QA_DATASET if sample['Question'] == question),
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
    memory_answer = get_memory_answer(question)['parsed_value']

    # Get the memory answer's category
    memory_category = get_memory_answer(question)['category']

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


async def generate_paragraph(inputs: list[str], mode: str) -> list:
    if mode == 'Chain':
        return await middle_paragraph_generator.async_run(inputs)
    if mode == 'Edge':
        return await edge_paragraph_generator.async_run(inputs)
    raise AttributeError