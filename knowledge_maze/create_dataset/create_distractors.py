"""
Module for generating distractor paragraphs for a question-answering dataset.

This script provides the functionality to automatically create "distractor" paragraphs
using a large language model API (gptAPI). These paragraphs are designed to be
semantically related to a given question or answer but do not contain the correct
information, making them useful for building robust QA evaluation datasets.

The process is as follows:
1.  Input data structures (`qa_dataset` and `paragraph_dataset`) are provided.
2.  Prompts are constructed from each question-answer pair in the `qa_dataset`.
3.  The `gptAPI` is called in a batch to generate all paragraphs asynchronously.
4.  Each new paragraph is assigned a unique UUID.
5.  The original `qa_dataset` is updated in-place with the new paragraph UUIDs,
    and the `paragraph_dataset` is updated with the UUID-to-text mappings.

There are two types of distractor generation available:
- 'ent': Generates a paragraph about the subject of the question but avoids
         answering the question itself.
- 'num': Generates a paragraph about an unrelated topic that uses the same
         number or date as the answer.
"""
from uuid import uuid4
from typing import Dict, List, Any

from knowledge_maze.tools.api_tools import *

DISTRACTOR_PROMPT_ENTS = """You are a factual writer.\n\n

                            You will be given:\n
                            - A subject (which can be a person, place, organization, or object)\n
                            - A question that must NOT be answered\n\n
                            
                            Your task is to write a concise, factual paragraph about the subject.\n
                            ❗ Do NOT answer the question or include any information that directly or indirectly 
                            reveals its answer.\n
                            ✅ You must still mention the subject explicitly and provide informative content about 
                            it that is unrelated to the question."""
DISTRACTOR_PROMPT_NUMS = """You will be given a question and its correct answer (which is either a number or a date). 
                            Your task is to write a short, factual paragraph about a different subject that shares the 
                            same type of fact — using the same number or date — but without answering the original 
                            question or referring to its subject. \n\n
                            
                            Guidelines:\n
                            - Use the exact number or date from the answer in your paragraph.\n
                            - Do NOT mention or imply anything related to the original question or its subject.\n
                            - Your paragraph must be factual, coherent, and reference a real-world person, place, event, 
                            or organization.\n
                            - Keep it concise (3–5 sentences).\n
                            - You must include the number/date directly in the paragraph."""


def get_paragraph_inputs(dataset: List[Dict[str, Any]]) -> List[str]:
    """
    Flattens the dataset into a single list of prompts,
    one for each answer in each question.
    """
    return [
        f"Answer: {ans},\n Question: {sample['Question']}\n"
        for sample in dataset
        for key, ans in sample['Answer'].items()
        if key != 'Memory answer'
    ]


async def generate_distractors(
        qa_dataset: List[Dict[str, Any]],
        paragraph_dataset: Dict[str, Dict[str, str]],
        para_type: str
) -> None:
    """
    Generates distractor paragraphs for each answer in the qa_dataset,
    and updates both datasets in-place.

    Args:
        qa_dataset: A list of dictionaries, where each dict is a Q&A sample.
        paragraph_dataset: A dictionary mapping paragraph IDs to their text.
        para_type: The type of distractor to generate ('ent' or 'num').
    """
    # 1. Create a flat list of all prompts needed
    prompt_inputs = get_paragraph_inputs(qa_dataset)
    if not prompt_inputs:
        print("No inputs to process.")
        return

    # 2. Set up the correct generator
    if para_type == 'ent':
        distractor_generator = gptAPI(DISTRACTOR_PROMPT_ENTS,
                                      temperature=0.7,
                                      max_concurrent_requests=100,
                                      max_rpm=4900)
    elif para_type == 'num':
        distractor_generator = gptAPI(DISTRACTOR_PROMPT_NUMS,
                                      temperature=0.7,
                                      max_concurrent_requests=100,
                                      max_rpm=4900)
    else:
        raise ValueError("para_type must be either 'ent' or 'num'")

    # 3. Generate all paragraphs in a single, parallelized batch
    generated_texts = await distractor_generator.async_run(prompt_inputs)
    generated_uuids = [str(uuid4()) for _ in generated_texts]

    # 4. Distribute the generated paragraphs back to the data structures
    current_idx = 0
    for sample in qa_dataset:
        # Determine how many paragraphs were generated for this sample
        num_answers = len(sample['Answer'])

        # Slice the results corresponding to the current sample
        end_idx = current_idx + num_answers
        ids_for_sample = generated_uuids[current_idx:end_idx]
        texts_for_sample = generated_texts[current_idx:end_idx]

        # Update the qa_dataset sample with the new paragraph IDs
        sample['Distractor paragraphs'].extend(ids_for_sample)

        # Update the paragraph_dataset with the new IDs and texts
        for para_id, para_text in zip(ids_for_sample, texts_for_sample):
            paragraph_dataset[para_id] = {'Text': para_text,
                                          'Type': 'Distractor'}

        # Move the index to the start of the next sample's block
        current_idx = end_idx
