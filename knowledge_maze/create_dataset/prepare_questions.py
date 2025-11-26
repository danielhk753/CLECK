import re
import json
import random
import logging

from knowledge_maze.tools.api_tools import gptAPI

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
    for sample in parsed_responses:
        # Check if the reasoning type is valid, otherwise mark it as DISCARDED
        if 'reasoning_type' in sample and sample['reasoning_type'] in REASONING_TYPES:
            sample['reasoning_type'] = re.split(" — ", REASONING_TYPES[sample['reasoning_type']])[0]

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

    return process_answers_with_claims(filtered_responses)


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
        all_options = [sample.get('original answer'),
                       sample.get('counter answer A'),
                       sample.get('counter answer B')] + sample.get('other answers')

        processed_samples.append({'question': sample['revised_question'],
                                  'answer': new_answer,
                                  'reasoning_type': sample['reasoning_type'],
                                  'all options': all_options
                                  })


    return processed_samples


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
        autopct=lambda p: f'{p:.1f}%\n({int(p / 100 * total_samples)})',
        startangle=140,
        colors=plt.cm.Paired.colors
    )
    plt.title('Reasoning Types Percentage Distribution')
    plt.tight_layout()

    # Display the plot
    plt.show()
