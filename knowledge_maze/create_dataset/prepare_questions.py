import re
import json
import random
import logging

import numpy as np

from knowledge_maze.tools.api_tools import gptAPI
from knowledge_maze.create_dataset.counter_answers import (create_counterfactuals,
                                                           replace_sub_answers_with_original)

# Try to import nest_asyncio, install if needed
try:
    import nest_asyncio

    nest_asyncio.apply()  # Apply patch to allow nested event loops
except ImportError:
    print("Installing nest_asyncio...")
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "nest_asyncio"])
    import nest_asyncio

    nest_asyncio.apply()

REASONING_TYPES = {
    "R1": "Extremum reasoning (max/min/highest/lowest) — select entity based on a single attribute across multiple entities",
    "R2": "Temporal ordering (earliest/latest/first/last) — select entity based on chronological events or releases",
    "R3": "Filtered selection (constraint-based elimination) — identify entity/entities that satisfy multiple non-trivial constraints; requires multi-step reasoning",
    "R4": "Comparative multi-attribute reasoning — compare entities across two or more attributes to select the answer",
    "R5": "Aggregation-based reasoning (count/frequency/mode) — determine answer via counting, classification aggregation, or distribution patterns",
    "NOISE": "Invalid / non-deterministic / binary / trivial question — return error if a single definitive answer cannot be determined"
}


async def transform_question_to_specific(complex_question_list):
    """
    Transforms complex multi-answer factual questions into single-answer reasoning questions,
    adhering to specified transformation rules and structure guidelines. This function handles
    input processing, API interaction, and post-response validation and formatting.

    Args:
        complex_question_list (list[dict[str, str]]): List of dictionaries where each dictionary contains
            a question-answer pair under the keys `question` and `answer`.

    Returns:
        list[dict]: A list of dictionaries containing transformed questions, counter-factual answers,
            and other metadata curated from the original input and API response, based on specific
            transformation rules.

    Raises:
        JSONDecodeError: If the API response cannot be parsed into a valid JSON structure.
        KeyError: If required keys are missing in the API response or during filtering operations.
    """
    # System prompt instructing the LLM how to transform the question
    system_prompt = """SYSTEM PROMPT:

    You are an expert question reformulation engine. Your goal is to transform multi-answer factual questions into a single-answer reasoning question.

    You must follow all rules below.

    ---------------------
    REQUIRED OUTPUT FORMAT
    ---------------------

    Always respond using exactly this structure:

    {
      "revised_question": "...",
      "original answer": "...",
      "counter-factual answers": ["...", "...", ...]
      "reasoning_type": "R1-R5",
      "status": "OK"
    }

    Instructions for counter-factual answers:
    - Generate 2 or more counter-factual answers that are plausible but **different from the true answer**.
    - Counter-factuals should **still make sense given the domain/entities**, but must be incorrect.
    - They should be inferable from the original QA sample (e.g., using other entities/values from the original input).
    - Do not invent entirely unrelated entities or facts.
    
    If transformation is not allowed (see rules), respond with exactly:

    {
      "status": "TRANSFORMATION_FAILED: REASON_CODE"
    }

    No additional text is allowed.

    ---------------------
    TRANSFORMATION RULES
    ---------------------

    1. The revised question must require reasoning that uses ALL the provided entity→attribute mappings from the original input.

    2. The revised question must lead to ONE final factual answer — not a list.

    3. The new question must NOT become:
       - yes/no
       - true/false
       - a binary choice (e.g., left vs right, male vs female)
       - a trivial wording variant of the original question
       - a question that still requires multiple answers

    4. If your revised question logically permits only TWO meaningful possible outcomes (binary), you MUST return:

    {
      "status": "TRANSFORMATION_FAILED: BINARY_OUTCOME"
    }

    5. The question must be mapped to exactly one reasoning category from this list:

    "R1": "Extremum reasoning (max/min/highest/lowest) — select entity based on a single attribute across multiple entities",
    "R2": "Temporal ordering (earliest/latest/first/last) — select entity based on chronological events or releases",
    "R3": "Filtered selection (constraint-based elimination) — identify entity/entities that satisfy multiple non-trivial constraints; requires multi-step reasoning",
    "R4": "Comparative multi-attribute reasoning — compare entities across two or more attributes to select the answer",
    "R5": "Aggregation-based reasoning (count/frequency/mode) — determine answer via counting, classification aggregation, or distribution patterns",
    "NOISE": "Invalid / non-deterministic / binary / trivial question — return error if a single definitive answer cannot be determined"

    If the category is "NOISE", you MUST return:

    {
      "status": "TRANSFORMATION_FAILED: NOISE_REASONING_CATEGORY"
    }

    6. The revised question should be simple, direct, and human readable.

    7. Do NOT change or invent facts. Only change the structure of the question.

    ---------------------
    FEW-SHOT EXAMPLES
    ---------------------

    EXAMPLE 1
    Input:
    "What were box office values of the Star Wars films in the prequel and sequel trilogies?"

    Output:
    {
      "revised_question": "Which Star Wars trilogy earned the highest total global box office revenue?",
      "answer": "The sequel trilogy",
      "reasoning_type": "R5",
      "status": "OK"
    }

    ---

    EXAMPLE 2
    Input:
    "Who won the World Car of the Year Award from 2019 to 2023, and what was the height of each winning car?"

    Output:
    {
      "revised_question": "Among the World Car of the Year winners from 2019 to 2023, which model was the tallest?",
      "answer": "Kia Telluride",
      "reasoning_type": "R1",
      "status": "OK"
    }

    ---

    EXAMPLE 3
    Input:
    "What is the batting hand of each of the first five picks in the 1998 MLB draft?"

    Output:
    {
      "status": "TRANSFORMATION_FAILED: BINARY_OUTCOME"
    }

    ---

    Your job: Apply these rules to the next input question.
    """

    # Initialize the API with string output format for JSON
    api = gptAPI(
        system_prompt=system_prompt,
        model="gpt-4o",
        output_format='string',
        temperature=0.2
    )

    # Extract question-answer pairs
    qa_pairs = [(sample['question'], sample['answer']) for sample in complex_question_list]
    sub_question_list = [sample['sub-questions'] for sample in complex_question_list]

    # Create prompts
    prompts = [
        f"Here is the question: {question}\n\nThis is the original answer: {answer}\n\n"
        for question, answer in qa_pairs
    ]

    # Get API responses
    string_responses = await api.async_run(prompts)

    # Parse string responses to dictionaries
    parsed_responses = []
    for i, response_str in enumerate(string_responses):
        try:
            # Attempt to parse the string as JSON
            parsed_dict = json.loads(response_str)
            parsed_responses.append(parsed_dict)
        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse response {i} as JSON: {e}")
            # Add the original string if parsing fails
            parsed_responses.append({"raw_response": response_str, "error": "JSON parsing failed"})

    # After parsing, update the 'reasoning_type' and validate 'counter-factual answers' length.
    for sample, original_answer, sub_q in zip(parsed_responses, qa_pairs, sub_question_list):
        ###########################################################################################
        # Patch for dev:
        if 'reasoning_type' in sample and (sample['reasoning_type'] == "R3"
                                           or sample['reasoning_type'] == "R4"
                                           or sample['reasoning_type'] == "R5"):
            sample['reasoning_type'] = "DISCARDED"
            sample['status'] = "NOT_OK"
        ###########################################################################################
        # Check if the reasoning type is valid, otherwise mark it as DISCARDED
        if 'reasoning_type' in sample and sample['reasoning_type'] in REASONING_TYPES:
            sample['reasoning_type'] = re.split(" — ", REASONING_TYPES[sample['reasoning_type']])[0]
            try:
                sample['all options'] = list(original_answer[1])
                sample['sub-questions'] = sub_q
                sample['full answer'] = original_answer
            except:
                sample['reasoning_type'] = "DISCARDED"
                sample['status'] = "NOT_OK"
        else:
            sample['reasoning_type'] = "DISCARDED"
            sample['status'] = "NOT_OK"

        # Validate the length of counter-factual answers
        if 'counter-factual answers' in sample and len(sample['counter-factual answers']) < 2:
            sample['reasoning_type'] = "DISCARDED"
            sample['status'] = "NOT_OK"

    # Filter only 'status': 'OK':
    filtered_responses = [{key: value for key, value in sample.items() if key != 'status'} for sample in
                          parsed_responses if sample.get('status') == 'OK']

    # Process 'counter-factual answers' and create the new keys
    for item in filtered_responses:
        if 'counter-factual answers' in item:
            counter_answers = item.pop('counter-factual answers')  # Remove the old key
            if counter_answers:
                item['counter answer A'] = random.choice(counter_answers)
                item['counter answer B'] = random.choice(
                    [ans for ans in counter_answers if ans != item['counter answer A']])
                item['other answers'] = [ans for ans in counter_answers if
                                         ans not in {item['counter answer A'], item['counter answer B']}]
            else:
                item['counter answer A'] = item['counter answer B'] = None
                item['other answers'] = []

    visualize_reasoning_distribution(parsed_responses)

    processed_responses = process_answers_with_claims(filtered_responses)

    return replace_sub_answers_with_original(create_counterfactuals(processed_responses))


def visualize_reasoning_distribution(parsed_responses):
    """
    Visualize the reasoning type distribution of the parsed responses.

    Args:
        parsed_responses (list): List of parsed response dictionaries containing reasoning types.

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    # Count reasoning types
    reasoning_counts = {}
    for response in parsed_responses:
        reasoning_type = response.get('reasoning_type', 'DISCARDED')
        reasoning_counts[reasoning_type] = reasoning_counts.get(reasoning_type, 0) + 1

    # Calculate percentages
    total_samples = sum(reasoning_counts.values())
    reasoning_percentages = {
        reasoning: (count / total_samples) * 100 for reasoning, count in reasoning_counts.items()
    }

    # Create a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        reasoning_percentages.values(),
        labels=reasoning_percentages.keys(),
        autopct=lambda p: f'{p:.1f}%\n({int(np.round(p / 100 * total_samples))})',
        startangle=140,
        colors=plt.cm.Paired.colors
    )
    plt.title('Reasoning Types Percentage Distribution')
    plt.tight_layout()

    # Display the plot
    plt.show()


def process_answers_with_claims(samples):
    """
    Process the samples to create a new 'answer' key containing 'memory', 'claim', and 'evidence'.

    Args:
        samples (list): List of response dictionaries containing answers and other data.

    Returns:
        list: List of samples with the modified 'answer' key.
    """
    processed_samples = []
    for sample in samples:
        new_answer = {
            'memory': sample.get('original answer'),
            'claim': sample.get('counter answer A'),
            'evidence': sample.get('counter answer B')
        }

        processed_samples.append({'question': sample['revised_question'],
                                  'answer': new_answer,
                                  'reasoning_type': sample['reasoning_type'],
                                  'original answer': sample.get('full answer')[1],
                                  'all options': sample.get('all options'),
                                  'sub-questions': sample.get('sub-questions')
                                  })


    return processed_samples


async def create_claim(question, answer):
    """
    Generates a claim based on the provided question and answer, allowing for counter-factuals.

    Args:
        question (str): The question text.
        answer (str): The answer text (factual or counter-factual).

    Returns:
        dict: A dictionary containing the generated claim or an error message if the process fails.
    """
    system_prompt = """SYSTEM PROMPT:

    You are an expert claim generation engine. Your goal is to construct a concise, factual or counter-factual claim 
    based on a given question and answer. The answer may be factually incorrect, and you must ignore factual correctness 
    when generating the claim.

    ---------------------
    REQUIRED OUTPUT FORMAT
    ---------------------

    Always respond using exactly this structure:

    {
      "claim": "..."
    }

    Rules for generating the claim:
    - The claim must be a single, complete sentence.
    - It must represent the relationship between the question and the provided answer exactly as given.
    - Do not fact-check or validate the provided answer; treat it as given, even if counter-factual.

    ---------------------
    EXAMPLE

    Input Question: "What is the capital of France?"
    Input Answer: "Berlin"

    Output Claim: "The capital of France is Berlin."
    """

    api = gptAPI(
        system_prompt=system_prompt,
        model="gpt-4o",
        output_format='string',
        temperature=0.2
    )

    prompt = f"Input Question: \"{question}\"\nInput Answer: \"{answer}\"\n"

    try:
        # Generate the claim using the API
        response = api.run(prompt)
        claim_response = json.loads(response)
        return claim_response.get("claim", "No claim generated.")
    except Exception as e:
        # Log and return error details
        logging.error(f"Error during claim generation: {e}")
        return {"error": "Claim generation failed."}


async def prepare_llm_rewrite_request(sample: dict):
    """
    Transforms the dataset sample into the correct format for the LLM rewrite step.
    """

    question = sample["question"]

    # First sub-question remains factual
    first_fact = sample["sub-questions"][0]

    # Remaining sub-questions become rewrite targets
    rewrite_targets = [
        sq for sq in sample["sub-questions"][1:]
    ]

    # The counterfactual final answer that rewriting must support
    target_answer = sample["answer"]["evidence"]

    return {
        "original_question": question,
        "fixed_fact": await create_claim(first_fact["sub-question"], first_fact["sub-answer"]),
        "rewrite_sub_questions": [
            sq["sub-question"] for sq in rewrite_targets
        ],
        "target_final_answer": target_answer
    }


async def create_counter_answers(sample):
    system_prompt = """You are modifying factual supporting evidence to make it consistent with a counterfactual answer.

                        The input contains:
                        - A main question.
                        - One fixed true fact (must stay unchanged).
                        - A list of sub-questions whose answers should be rewritten.
                        - A target counterfactual final answer the rewritten evidence must logically support.
                        
                        Your task is to generate new answers for the rewrite sub-questions.
                        
                        Rules:
                        1. The rewritten answers must be factually coherent with each other.
                        2. They must support the target final answer as if it were true.
                        3. Do NOT mention the target answer explicitly or reference reasoning steps.
                        4. Do NOT expose that the data is modified or counterfactual.
                        5. Keep the output grounded, natural, and plausible.
                        6. The first fact remains unchanged and should be included as-is in the final output.
                        7. Maintain original answer types (numbers stay numbers, names remain names, lists remain lists, etc.).
                        
                        Return the output as a list of answers in the same order. The answers must be as short as possible.
    """

    api = gptAPI(
        system_prompt=system_prompt,
        model="gpt-4o",
        output_format='list',
        temperature=0.2
    )

    prompt = f"{await prepare_llm_rewrite_request(sample)}"

    try:
        # Generate the claim using the API
        return api.run(prompt)
    except Exception as e:
        # Log and return error details
        logging.error(f"Error during claim generation: {e}")
        return {"error": "Claim generation failed."}


from typing import List, Dict, Any


async def create_claim_dataset(dataset: List[Dict[Any, Any]]) -> List[Dict[Any, Any]]:
    """
    Creates a new dataset with claims generated from the original dataset.

    Args:
        dataset: Original dataset containing questions and answers

    Returns:
        A new dataset with claims generated from each sample
    """
    transformed_dataset = []

    for sample in dataset:
        # Extract necessary data from sample
        prepared_data = await prepare_llm_rewrite_request(sample)
        factual_claim = prepared_data['fixed_fact']
        rewrite_questions = prepared_data['rewrite_sub_questions']
        counter_answers = await create_counter_answers(sample)

        # Create new sample with initial fact
        transformed_sample = {
            'question': sample['question'],
            'answer': sample['answer'],
            'reasoning_type': sample['reasoning_type'],
            'all options': sample['all options'],
            'facts': [factual_claim]
        }

        # Add generated claims for each question-answer pair
        for question, answer in zip(rewrite_questions, counter_answers):
            claim = await create_claim(question, answer)
            transformed_sample['facts'].append(claim)

        transformed_dataset.append(transformed_sample)

    return transformed_dataset
