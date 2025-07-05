"""
Docstring...
"""
from typing import Tuple, Dict

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


def get_paragraph_inputs(dataset: List[Dict]) -> List:
    """Get all the q-a pairs as one list of inputs."""
    structured_input = []
    for sample in dataset:
        answers = sample['Answer'].values()
        question = sample['Question']
        for ans in answers:
            structured_input.append(f"""Answer: {ans},\n Question: {question}\n""")

    return structured_input


async def generate_distractors(qa_dataset, paragraph_dataset, para_type) -> None:
    prompt_inputs = get_paragraph_inputs(qa_dataset)

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
        raise ValueError

    paragraphs =  await distractor_generator.async_run(prompt_inputs)

    # index paragraphs
    # update both datasets and save in memory
