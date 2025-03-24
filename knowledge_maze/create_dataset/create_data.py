import re
import random
import asyncio

from tqdm import tqdm
from uuid import uuid4

from knowledge_maze.create_dataset.generate_paragraphs import get_memory_answer, date_counter_answer, numerical_counter_answers

async def process_data_to_list(db: list[dict], answer_type: str) -> list[dict]:
    """
    Processes the input database into a list of dictionaries, each containing
    a randomly chosen question, its answer, and two reasoning paths.

    Args:
        db (list[dict]): List of dictionaries containing answers and corresponding questions.
        answer_type (str): The answer type, can be 'Wikipedia Entity', 'Numerical' or 'Date & Time'.


    Returns:
        list[dict]: A list where each dictionary contains:
            - 'Question': The main question.
            - 'Answer': The associated answer.
            - 'Reasoning path 1': A chain of reasoning sub-questions (partition 1).
            - 'Reasoning path 2': A chain of reasoning sub-questions (partition 2).
    """
    results = []

    for cluster in tqdm(db):
        sample_id = str(uuid4())
        questions = cluster['Questions'].copy()
        answer = cluster['Answer']

        if len(questions) > 1:

            # Randomly select a question to be the main question
            main_question = random.choice(questions)
            questions.remove(main_question)

            # Partition the remaining questions into two reasoning paths
            if len(questions) == 1:
                # Special case: One reasoning path gets the initial and remaining question
                path1_questions = questions
                path2_questions = []
            else:
                random.shuffle(questions)
                split_point = random.randint(1, len(questions) - 1) if len(questions) > 1 else 1
                path1_questions = questions[:split_point]
                path2_questions = questions[split_point:]

            # Generate reasoning paths with chained questions
            def create_reasoning_path(path_questions):
                path = []
                previous_sub_question = main_question
                for question in path_questions:
                    path.append({
                        'Previous question': previous_sub_question,
                        'Next question': question,
                        'Paragraph': str(uuid4())
                    })
                    previous_sub_question = question
                # Add a final step where 'Next question' is None
                path.append({
                    'Previous question': previous_sub_question,
                    'Next question': None,
                    'Paragraph': str(uuid4())
                })
                return path

            reasoning_path_1 = create_reasoning_path(path1_questions)
            reasoning_path_2 = create_reasoning_path(path2_questions)

            if answer_type == 'Numerical':
                answer = (get_memory_answer(main_question), answer)
            if answer_type == 'Date & Time':
                answer = date_counter_answer(main_question)

            # Create the result dictionary for this cluster
            result = {
                'ID': sample_id,
                'Question': main_question,
                'Answer': answer,
                'Answer type': answer_type,
                'Reasoning path 1': reasoning_path_1,
                'Reasoning path 2': reasoning_path_2
            }
            results.append(result)

    if answer_type == 'Numerical':
        # Collect inputs:
        inputs = []
        for sample in results:
            inputs.append(str({
                'Real answer': sample['Answer'][0],
                'Fake answer range': sample['Answer'][1],
                'Question': sample['Question']
            }))

        # Create counter answers:
        fake_answers = await numerical_counter_answers(inputs)

        # Update data:
        for sample, counter_ans in zip(results, fake_answers):
            sample['Answer'] = {
                'Memory answer': sample['Answer'][0],
                'Counter answer 1': int(re.search(r'\d+', counter_ans[0]).group()),
                'Counter answer 2': int(re.search(r'\d+', counter_ans[1]).group()),
                'Distractor answer': int(re.search(r'\d+', counter_ans[2]).group())
            }

    if answer_type == 'Wikipedia Entity':
        pass

    return results

