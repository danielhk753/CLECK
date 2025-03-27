import re
import random

import numpy as np
from tqdm import tqdm
from uuid import uuid4

from knowledge_maze.create_dataset.generate_paragraphs import (
    get_memory_answer, date_counter_answer, numerical_counter_answers, entity_counter_answers, generate_paragraph
)


async def process_data_to_list(db: list[dict], answer_type: str) -> list[dict]:
    """
    Processes the input database into a structured list of questions, answers, and reasoning paths.

    Args:
        db (list[dict]): List of dictionaries containing answers and corresponding questions.
        answer_type (str): The answer type ('Wikipedia Entity', 'Numerical', or 'Date & Time').

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

    for cluster in tqdm(db):
        sample_id = str(uuid4())
        if len(cluster['Questions']) >= 2:
            main_question, path1_questions, path2_questions = partition_questions(cluster['Questions'])
            reasoning_path_1 = create_reasoning_path(main_question, path1_questions)
            reasoning_path_2 = create_reasoning_path(main_question, path2_questions)
            answer = process_answer(answer_type, main_question, cluster['Answer'])

            result = {
                'ID': sample_id,
                'Question': main_question,
                'Answer': answer,
                'Answer type': answer_type,
                'Reasoning path 1': reasoning_path_1,
                'Reasoning path 2': reasoning_path_2
            }
            results.append(result)

    return await process_counter_answers(results, answer_type)


def partition_questions(questions: list[str]) -> tuple[str, list[str], list[str]]:
    """Splits the questions into a main question and two reasoning paths."""
    questions = questions.copy()
    main_question = random.choice(questions)
    questions.remove(main_question)

    if len(questions) == 1:
        return main_question, questions, []

    random.shuffle(questions)
    split_point = random.randint(1, len(questions) - 1) if len(questions) > 1 else 1
    return main_question, questions[:split_point], questions[split_point:]


def create_reasoning_path(main_question: str, path_questions: list[str]) -> list[dict]:
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


def process_answer(answer_type: str, question: str, answer):
    """Processes the answer based on its type."""
    if answer_type == 'Numerical':
        return get_memory_answer(question), answer
    if answer_type == 'Date & Time':
        return date_counter_answer(question)
    return answer


async def process_counter_answers(results: list[dict], answer_type: str) -> list[dict]:
    """Processes and updates counter-answers for numerical and Wikipedia entity types."""
    if answer_type == 'Numerical':
        inputs = [{
            'Real answer': sample['Answer'][0],
            'Fake answer range': sample['Answer'][1],
            'Question': sample['Question']
        } for sample in results]

        fake_answers = await numerical_counter_answers([str(i) for i in inputs])

        for sample, counter_ans in zip(results, fake_answers):
            sample['Answer'] = {
                'Memory answer': sample['Answer'][0],
                'Counter answer 1': extract_number(counter_ans[0]),
                'Counter answer 2': extract_number(counter_ans[1]),
                'Distractor answer': extract_number(counter_ans[2])
            }

    if answer_type == 'Wikipedia Entity':
        inputs = [{
            'Real answer': sample['Answer'],
            'Questions': get_edge_questions(sample)
        } for sample in results]

        fake_answers = await entity_counter_answers([str(i) for i in inputs])

        for sample, counter_ans in zip(results, fake_answers):
            sample['Answer'] = {
                'Memory answer': sample['Answer'],
                'Counter answer 1': counter_ans[0],
                'Counter answer 2': counter_ans[1],
                'Distractor answer': counter_ans[2]
            }

    return results


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


async def create_paragraphs(qa_data: list[dict]) -> dict:
    """Create the paragraph dataset"""
    # Collect the id & question pairs lists:
    middle_id_list = []
    middle_pargraph_info_list = []
    edge_id_list = []
    edge_pargraph_info_list = []

    for sample in qa_data:
        middle_id_list += get_middle_paragraph_ids(sample)
        middle_pargraph_info_list += get_middle_paragraph_info(sample)
        edge_id_list += get_edge_paragraph_ids(sample)
        edge_pargraph_info_list += get_edge_paragraph_info(sample)

    # Create all the paragraphs:
    middle_paragraphs = await generate_paragraph(middle_pargraph_info_list, mode='Chain')
    edge_paragraphs = await generate_paragraph(edge_pargraph_info_list, mode='Edge')

    # Pair ids with paragraphs:
    paragraph_data = {}
    for index, paragraph in zip(middle_id_list, middle_paragraphs):
        paragraph_data[index] = paragraph
    for index, paragraph in zip(edge_id_list, edge_paragraphs):
        paragraph_data[index] = paragraph

    return paragraph_data


def get_middle_paragraph_ids(qa_sample: dict) -> list[str]:
    path_1_ids = [sample['Paragraph'] for sample in qa_sample['Reasoning path 1'] if sample['Next question']]
    path_2_ids = [sample['Paragraph'] for sample in qa_sample['Reasoning path 2'] if sample['Next question']]
    return path_1_ids + path_2_ids


def get_edge_paragraph_ids(qa_sample: dict) -> list[str]:
    path_1_ids = [sample['Paragraph'] for sample in qa_sample['Reasoning path 1'] if not sample['Next question']]
    path_2_ids = [sample['Paragraph'] for sample in qa_sample['Reasoning path 2'] if not sample['Next question']]
    return path_1_ids + path_2_ids


def get_middle_paragraph_info(qa_sample: dict) -> list[str]:
    path_1_info = [
        f"""Question 1: {sample['Previous question']} \n
            Question 2: {sample['Next question']} \n
            Answer": {qa_sample['Answer']['Memory answer']}"""
        for sample in qa_sample['Reasoning path 1'] if sample['Next question']
    ]
    path_2_info = [
        f"""Question 1: {sample['Previous question']} \n
            Question 2: {sample['Next question']} \n
            Answer": {qa_sample['Answer']['Memory answer']}"""
        for sample in qa_sample['Reasoning path 2']  if sample['Next question']
    ]
    return path_1_info + path_2_info


def get_edge_paragraph_info(qa_sample: dict) -> list[str]:
    path_1_info = [
        f"""Question: {sample['Previous question']} \n
            Answer": {qa_sample['Answer']['Counter answer 1']}"""
        for sample in qa_sample['Reasoning path 1'] if not sample['Next question']
    ]
    path_2_info = [
        f"""Question: {sample['Previous question']} \n
            Answer": {qa_sample['Answer']['Counter answer 2']}"""
        for sample in qa_sample['Reasoning path 2']  if not sample['Next question']
    ]
    return path_1_info + path_2_info
