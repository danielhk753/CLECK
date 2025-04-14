"""
Generate paragraphs

"""
import os
import json

from dotenv import load_dotenv

from knowledge_maze.create_dataset.create_data_qa import wrap_text
from knowledge_maze.create_dataset.generation_tools.gen_paragraphs import generate_paragraph


load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')

REASONING_PATH_QUESTIONS = [
    "Counter answers are plausible?",
    "No Similar sub-questions?",
    "Sub-questions are relevant?"
]


# Load the reasoning dataset QA:
QA_FILE_PATH = f"{REPO_PATH}\data_files\\knowledge_maze_qa.json"
with open(QA_FILE_PATH, "r", encoding="utf-8") as file:
    REASONING_DATASET = json.load(file)


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
        paragraph_data[index] = {
            'Text': paragraph,
            'Type': 'Chain'
        }
    for index, paragraph in zip(edge_id_list, edge_paragraphs):
        paragraph_data[index] = {
            'Text': paragraph,
            'Type': 'Edge'
        }

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


def get_paragraph_info(paragraph_id: str) -> dict:
    pass


def display_paragraph(paragraph_id: str, paragraphs_value: dict) -> None:
    print("=" * 120)
    print(f"\033[1m\033[4mParagraph type:\033[0m {paragraphs_value['Type']}")
    for sample in REASONING_DATASET:
        for reasoning_path in ['Reasoning path 1', 'Reasoning path 2']:
            for paragraph in sample[reasoning_path]:
                if paragraph_id == paragraph['Paragraph']:
                    if paragraphs_value['Type'] == 'Chain':
                        print(f"\033[1m\033[4mPrevious question:\033[0m {paragraph['Previous question']}")
                        print(f"\033[1m\033[4mNext question:\033[0m {paragraph['Next question']}")
                        print(f"\033[1m\033[4mShared answer:\033[0m {sample['Answer']['Memory answer']}\n")
                    else:
                        print(f"\033[1m\033[4mQuestion:\033[0m {paragraph['Previous question']}")
                        print(f"\033[1m\033[4mMemory answer:\033[0m {sample['Answer']['Memory answer']}\n")
                    break
    print(wrap_text(f"\033[1m\033[4mParagraph text:\033[0m {paragraphs_value['Text']}"))
    print("=" * 120)