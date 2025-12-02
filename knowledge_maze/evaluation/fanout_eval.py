import random
from collections import defaultdict

from knowledge_maze.tools.api_tools import *

ASK_QUESTION_CONTEXT_PROMPT = """
    You are a logical reasoner. Your task is to answer the question using only the information provided in the context 
    below.\n\n

    Rules:\n
    - Do NOT use any external or real-world knowledge.\n
    - Only rely on statements explicitly made in the context, even if they are factually incorrect or incomplete.\n
    - If the context doesn’t provide enough information or you find conflict and have more than one final answer, respond: ‘No answer’.\n
    - Answer with a short explanation - no more than one line.\n\n
    - Your answer must be a single entity or phrase, taken directly from or unambiguously supported by the context.\n\n
    """
qa_context_model = gptAPI(ASK_QUESTION_CONTEXT_PROMPT,
                         output_format='string',
                         temperature = 1.5,
                         max_concurrent_requests = 100,
                         max_rpm=4900)


def create_prompt_co_located(sample, num_retries=100):
    """
    Create the prompt for question and context, concatenate all the prompts for self-consistency + shuffle order in each iteration.
    """
    all_prompts = []
    context = [sample['context']['direct claim']] + [sample['context']['co-located aggregation']] + sample['context'][
        'distractors']

    for i in range(num_retries):
        random.shuffle(context)
        prompt = f"""Question: {sample['question']}\n\n

            Context:\n
            {context}\n\n        
            """
        all_prompts.append(prompt)
    return all_prompts


def create_prompt_distributed(sample, num_retries=100):
    """
    Create the prompt for question and context, concatenate all the prompts for self-consistency + shuffle order in each iteration.
    """
    all_prompts = []
    context = [sample['context']['direct claim']] + sample['context']['distributed multi-hop'] + sample['context'][
        'distractors']

    for i in range(num_retries):
        random.shuffle(context)
        prompt = f"""Question: {sample['question']}\n\n

            Context:\n
            {context}\n\n        
            """
        all_prompts.append(prompt)
    return all_prompts


async def get_answers_co_located(sample):
    prompt = create_prompt_co_located(sample)
    answers = await qa_context_model.async_run(prompt)
    return {k: v for k, v in sorted(string_frequency_dict(answers).items(), key=lambda item: item[1], reverse=True)}


async def get_answers_distributed(sample):
    prompt = create_prompt_distributed(sample)
    answers = await qa_context_model.async_run(prompt)
    return {k: v for k, v in sorted(string_frequency_dict(answers).items(), key=lambda item: item[1], reverse=True)}


def string_frequency_dict(string_list):
    """
    Converts a list of strings into a dictionary where each string is a key
    and its value is the frequency of that string in the list.

    Args:
        string_list: A list of strings.

    Returns:
        A dictionary with string keys and integer frequency values.
    """
    frequency_dict = defaultdict(int)
    for s in string_list:
        frequency_dict[s] += 1
    return dict(frequency_dict)


async def evaluate(sample):
    answers_co_located = await get_answers_co_located(sample)
    answers_distributed = await get_answers_distributed(sample)

    answer_options = sample['answer']
    memory_answer = answer_options['memory']
    claim_answer = answer_options['claim']
    evidence_answer = answer_options['evidence']

    co_loc = {'memory': 0, 'claim': 0, 'evidence': 0, 'no answer': 0, 'other': 0}
    for key, val in answers_co_located.items():
        if are_related(memory_answer, key):
            co_loc['memory'] += val
        elif are_related(claim_answer, key):
            co_loc['claim'] += val
        elif are_related(evidence_answer, key):
            co_loc['evidence'] += val
        elif are_related('No answer', key):
            co_loc['no answer'] += val
        else:
            co_loc['other'] += val

    distrib = {'memory': 0, 'claim': 0, 'evidence': 0, 'no answer': 0, 'other': 0}
    for key, val in answers_distributed.items():
        if are_related(memory_answer, key):
            distrib['memory'] += val
        elif are_related(claim_answer, key):
            distrib['claim'] += val
        elif are_related(evidence_answer, key):
            distrib['evidence'] += val
        elif are_related('No answer', key):
            distrib['no answer'] += val
        else:
            distrib['other'] += val

    return co_loc, distrib


async def get_final_eval(dataset):
    final_results_co_loc = []
    final_results_dist = []
    for sample in dataset:
        res_co_loc, res_dist = await evaluate(sample)
        final_results_co_loc.append(res_co_loc)
        final_results_dist.append(res_dist)

    return final_results_co_loc, final_results_dist


def are_related(s1, s2):
    # Cleans strings by keeping only alphanumeric characters and making them lowercase
    c1 = "".join(c for c in s1 if c.isalnum()).lower()
    c2 = "".join(c for c in s2 if c.isalnum()).lower()

    # Check for direct substring relationship (original logic)
    if c1 in c2 or c2 in c1:
        return True

    # --- New Logic: Check for > 90% character overlap ---

    # Combine the two cleaned strings into a set of unique characters
    set1 = set(c1)
    set2 = set(c2)

    # Calculate the intersection (common characters) and union (all unique characters)
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # Avoid division by zero if both strings are empty after cleaning
    if not union:
        return True  # Or False, depending on desired behavior for empty strings. Assuming True for equality/perfect overlap.

    # Calculate the Jaccard similarity (measure of overlap)
    # Jaccard Index = |Intersection| / |Union|
    overlap_ratio = len(intersection) / len(union)

    # The condition is that the overlap (Jaccard Index) is more than 90% (0.9)
    return overlap_ratio > 0.9