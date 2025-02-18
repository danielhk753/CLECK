"""
Load and classify multiple QA datasets into a unified format.

This module provides functions to:
- Load and process the TriviaQA, Natural Questions, and SQuAD datasets.
- Extract question-answer pairs.
- Filter out unique answers as initial candidates.
- Generate statistics and plots.

Usage:
    from this_module import load_tqa, load_nq, load_sqd

    tqa_data = load_tqa(output_path="processed_triviaqa.json")
    nq_data = load_nq(output_path="processed_nq.json")
    sqd_data = load_sqd(output_path="processed_squad.json")
"""

import os
import json
import ijson
from tqdm import tqdm
from dotenv.main import load_dotenv

# Load datasets path from .env file:
load_dotenv()
TRIVIA_QA_PATH = os.getenv('TRIVIA_QA_PATH')
Natural_Questions_PATH = os.getenv('Natural_Questions_PATH')
SQUAD_PATH = os.getenv('SQUAD_PATH')


def load_tqa(output_path: str | None = None, input_path: str = TRIVIA_QA_PATH) -> list[dict]:
    """
   Load and process the TriviaQA dataset.

   Args:
       output_path (str | None): Path to save the processed dataset (optional).
       input_path (str): Path to the TriviaQA dataset JSON file.

   Returns:
       list[dict]: Processed list of question-answer pairs.
   """
    tqa_dataset = []

    # Count the total number of elements in the dataset:
    with open(input_path, "r", encoding="utf-8") as f:
        total_samples = sum(1 for _ in ijson.items(f, "Data.item"))

    with open(input_path, "r", encoding="utf-8") as f:
        objects = ijson.items(f, "Data.item")
        for obj in tqdm(objects, total=total_samples, desc="Processing TriviaQA"):
            question = obj['Question']
            answer_type = obj['Answer']['Type']

            # Extract answer based on type:
            if answer_type == 'WikipediaEntity':
                answer = obj['Answer']['MatchedWikiEntityName']
            elif answer_type == 'Numerical' or answer_type == 'Range':
                answer = obj['Answer']['Value']
            else:
                continue  # Skip unsupported answer types (Free-form)

            tqa_dataset.append({'Question': question, 'Answer': answer, 'Type': answer_type})

    # Save processed data if output path is given:
    if output_path:
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(tqa_dataset, out_file)

    return tqa_dataset


def load_nq(output_path: str | None = None, input_path: str = Natural_Questions_PATH) -> list[dict]:
    """
    Load and process the Natural Questions dataset.

    Args:
        output_path (str | None): Path to save the processed dataset (optional).
        input_path (str): Path to the Natural Questions JSONL file.

    Returns:
        list[dict]: Processed list of question-answer pairs.
    """
    nq_dataset = []

    # Count the total number of elements in the dataset:
    with open(input_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Processing Natural Questions"):
            try:
                data = json.loads(line.strip())  # Parse JSON

                # Get only questions with short answer:
                if data["annotations"] and data["annotations"][0]["short_answers"]:
                    # Get start and end position of answer:
                    start = data["annotations"][0]["short_answers"][0]["start_token"]
                    end = data["annotations"][0]["short_answers"][0]["end_token"]
                    # Extract answer:
                    answer_text = " ".join(data["document_text"].split()[start:end])

                    nq_dataset.append({"Question": data["question_text"], "Answer": answer_text})

            except json.JSONDecodeError:
                print("Skipping a bad JSON line.")

    # Save processed data if output path is given:
    if output_path:
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(nq_dataset, out_file)

    return nq_dataset


def load_sqd(output_path: str | None = None, input_path: str = SQUAD_PATH) -> list[dict]:
    """
    Load and process the SQuAD dataset.

    Args:
        output_path (str | None): Path to save the processed dataset (optional).
        input_path (str): Path to the SQuAD JSON file.

    Returns:
        list[dict]: Processed list of question-answer pairs.
    """
    sqd_dataset = []

    # Count the total number of elements in the dataset:
    with open(input_path, "r", encoding="utf-8") as f:
        total_samples = sum(1 for _ in ijson.items(f, "data.item"))

    with open(input_path, "r", encoding="utf-8") as f:
        objects = ijson.items(f, "data.item")
        for obj in tqdm(objects, total=total_samples, desc="Processing SQuAD"):
            for paragraph in obj['paragraphs']:
                for qa_pair in paragraph['qas']:
                    if not qa_pair['is_impossible']:
                        sqd_dataset.append({'Question': qa_pair['question'],
                                            'Answer': qa_pair['answers'][0]['text']})

    # Save processed data if output path is given:
    if output_path:
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(sqd_dataset, out_file)

    return sqd_dataset
