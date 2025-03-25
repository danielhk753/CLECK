import os
import json
import textwrap

from dotenv import load_dotenv

load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')

# Load the QA :
QA_FILE_PATH = f"{REPO_PATH}\data_files\\merged_dataset.json"
with open(QA_FILE_PATH, "r", encoding="utf-8") as file:
    QA_DATASET = json.load(file)


def wrap_text(text: str, ident: str = '') -> str:
    """Wraps a given text using textwrap."""
    wrapper = textwrap.TextWrapper(width=95)
    return f'\n{ident}'.join(wrapper.wrap(text))


def display_qa_sample(sample: dict) -> None:
    """
    Displays detailed QA sample information with line wrapping using the textwrap module.

    Args:
        sample (dict): A dictionary containing QA sample data.

    Returns:
        None
    """
    original_answer = get_original_answer(sample['Question'])
    sub_questions_chain_1 = [q['Previous question'] for q in sample['Reasoning path 1']]
    sub_questions_chain_2 = [q['Previous question'] for q in sample['Reasoning path 2']]
    print('=' * 100)
    print(f"ID: {sample['ID']}")
    print(f"Type: {sample['Answer type']}")
    print(wrap_text(f"Question: {sample['Question']}", ident=' '*10))
    print(wrap_text(f"Original answer from dataset: {original_answer}"))
    print("\nAnswers:")
    for key, value in sample['Answer'].items():
        print(f"- {value} ({key})")
    print("\nSub-questions path 1:")
    for i, question in enumerate(sub_questions_chain_1):
        print(wrap_text(f" {i+1}. {question} --> {get_original_answer(question)}", ident='    '))
    print("\nSub-questions path 2:")
    for i, question in enumerate(sub_questions_chain_2):
        print(wrap_text(f" {i+1}. {question} --> {get_original_answer(question)}", ident='    '))
    print('=' * 100)


def get_original_answer(question):
    """
    Retrieves the memory answer for a given question.

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