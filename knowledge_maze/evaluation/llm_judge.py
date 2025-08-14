"""
Provides a highly efficient, batch-oriented function for classifying model answers.

This module is designed to take a large number of different answer lists, each
with its own set of valid options, and classify them using a single API call to
an LLM judge. This approach avoids messy, interleaved progress bars from
concurrent calls and maximizes throughput by creating one large, optimized batch.

The main function is `batch_judge_answers`, which replaces the need for looping
or concurrent calls to a single-item processing function.

Key Components:
- LLM_JUDGE_PROMPT: A constant system prompt that defines the classification rules.
- judge_model: An initialized API client for making the LLM calls.
- batch_judge_answers: The primary public function for batch processing.
"""
from collections import Counter
from typing import List, Dict, Any

from knowledge_maze.tools.api_tools import gptAPI


# The system prompt is defined once at the module level.
LLM_JUDGE_PROMPT = """You are an impartial and precise judge. Your task is to classify an 'Answer to Judge' into one of several 
                      categories based on the 'Valid Options' provided.

                      ---
                      Judging Rules: Respond with ONLY ONE of the following, based on these rules:

                      1.  **Direct Match:** If the 'Answer to Judge' clearly corresponds to one of the 'Valid Options', respond with 
                      the text of that matching option.
                          - Example: If options are ['Paris', 'London'] and the answer is 'The capital of France is Paris', respond 
                          with 'Paris'.

                      2.  **No Answer:** If the 'Answer to Judge' explicitly states an inability to answer (e.g., "I don't know", 
                      "The context does not provide this information", "Cannot answer"), respond with the exact phrase 'No Answer'.

                      3.  **Other Valid Answer:** If the 'Answer to Judge' is a plausible, different answer that is NOT in the 'Valid 
                      Options' and is NOT a refusal to answer, respond with the exact phrase 'Other Answer'.
                          - Example: If options are ['Paris', 'London'] and the answer is 'Berlin', respond with 'Other Answer'.

                      4.  **Invalid:** If the 'Answer to Judge' is gibberish, irrelevant, a fragment, or fails all other rules, 
                      respond with the exact phrase 'Invalid'.
                      """

# The API client is initialized once. Its async_run method will be used.
judge_model = gptAPI(LLM_JUDGE_PROMPT,
                     output_format='list',
                     temperature=0.2,
                     max_concurrent_requests=100,
                     max_rpm=4900)


def _calculate_probabilities(normalized_answers_chunk: List[str], options: Dict[str, str]) -> Dict[str, float]:
    """
    Calculates the probability distribution for a given chunk of judged answers.
    This is a private helper function for the main batch processor.
    """
    all_keys = list(options.keys()) + ['No Answer', 'Other Answer', 'Invalid']
    result_probabilities = {key: 0.0 for key in all_keys}

    if not normalized_answers_chunk:
        return result_probabilities

    value_to_key_map = {value: key for key, value in options.items()}
    total_count = len(normalized_answers_chunk)
    answer_counts = Counter(normalized_answers_chunk)

    for value, count in answer_counts.items():
        key = value_to_key_map.get(value)
        if key:
            result_probabilities[key] = count / total_count
        else:
            result_probabilities[value] = count / total_count

    return result_probabilities


async def batch_judge_answers(batch_input: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    Asynchronously judges answers for multiple, different questions in a single,
    highly efficient batch API call.

    Args:
        batch_input (List[Dict[str, Any]]): A list of "job" dictionaries.
            Each dictionary must contain two keys:
            - 'answers' (List[str]): A list of answers for a single question.
            - 'options' (Dict[str, str]): The options dict for that question.

    Returns:
        List[Dict[str, float]]: A list of probability summary dictionaries,
            with each result corresponding to an item in the batch_input.
    """
    if not batch_input:
        return []

    # --- Step 1: Flatten all jobs into a single list of prompts ---
    all_prompts = []
    # Keep track of how many answers each job has, for reconstruction later.
    answers_per_job = []

    for job in batch_input:
        answers = job.get('answers', [])
        options = job.get('options', {})
        option_values = list(options.values())

        answers_per_job.append(len(answers))

        for answer in answers:
            # The prompt sent to the API is just the variable part.
            prompt = f"""
            Valid Options:
            {option_values}

            Answer to Judge:
            "{answer}"

            Your verdict:
            """
            all_prompts.append(prompt)

    # --- Step 2: Make a single, large API call ---
    # This will show just ONE tqdm bar for the entire operation.
    judged_responses = await judge_model.async_run(all_prompts)
    normalized_answers = [result[0].strip() for result in judged_responses if result]

    # --- Step 3: Reconstruct the results ---
    # We "un-flatten" the single list of results back into chunks for each job.
    final_results = []
    current_index = 0
    for i, job in enumerate(batch_input):
        num_answers_for_this_job = answers_per_job[i]

        # Slice the portion of normalized_answers that belongs to this job.
        chunk = normalized_answers[current_index : current_index + num_answers_for_this_job]

        # Calculate the summary for this job's results.
        summary = _calculate_probabilities(chunk, job['options'])
        final_results.append(summary)

        # Move the index to the start of the next chunk.
        current_index += num_answers_for_this_job

    return final_results