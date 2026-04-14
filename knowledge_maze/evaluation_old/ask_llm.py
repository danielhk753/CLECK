"""Get the question and context, then answer using LLM of our choosing

- get full context or just reasoning path 1/2 + make random order for the paragraphs.
- ask a question and get answer
"""
from typing import List

from knowledge_maze.tools.api_tools import *

# Initialize LLM prompts:
ASK_QUESTION_MEMORY_PROMPT = """
    You are a knowledgeable AI assistant with a vast internal knowledge base.\n
    Your task is to answer the following question based on your general knowledge.\n\n

    Rules:\n
    - Use your own knowledge to provide the answer.\n
    - Your answer must be a single, concise entity or phrase.\n\n
    """
ASK_QUESTION_CONTEXT_PROMPT = """
    You are a logical reasoner. Your task is to answer the question using only the information provided in the context 
    below.\n\n

    Rules:\n
    - Do NOT use any external or real-world knowledge.\n
    - Only rely on statements explicitly made in the context, even if they are factually incorrect or incomplete.\n
    - If the context does not contain enough information to answer, respond clearly that the answer cannot be determined.\n
    - Your answer must be a single entity or phrase, taken directly from or unambiguously supported by the context.\n\n
    """

# Create LLM API call objects:
qa_memory_model = gptAPI(ASK_QUESTION_MEMORY_PROMPT,
                         output_format='list',
                         temperature = 1.5,
                         max_concurrent_requests = 100,
                         max_rpm=4900)
qa_context_model = gptAPI(ASK_QUESTION_CONTEXT_PROMPT,
                         output_format='list',
                         temperature = 1.5,
                         max_concurrent_requests = 100,
                         max_rpm=4900)


async def batch_ask_llm_from_memory(
        questions: List[str],
        num_retries: int
) -> List[List[str]]:
    """
    Asynchronously asks multiple, different questions from memory in a single,
    highly efficient batch API call.

    Args:
        questions (List[str]): A list of different questions to ask the model.
        num_retries (int): The number of times to ask EACH question.

    Returns:
        List[List[str]]: A list of answer lists. The structure mirrors the
            input, where the i-th list of answers corresponds to the i-th
            question in the input `questions` list.
    """
    if not questions:
        return []

    # --- Step 1: Flatten all questions into a single list of prompts ---
    all_prompts = []
    for question in questions:
        prompt = f"""Question: {question}"""
        # Add the prompt 'num_retries' times for this specific question
        all_prompts.extend([prompt] * num_retries)

    # --- Step 2: Make a single, large API call ---
    # This will show just ONE tqdm bar (from qa_model's config) for the entire operation.
    nested_answers = await qa_memory_model.async_run(all_prompts)

    # Flatten the result into a single list of answer strings
    all_generated_answers = [result[0] for result in nested_answers if result]

    # --- Step 3: Reconstruct the results into a list of lists ---
    # "Un-flatten" the single list of answers back into chunks, one for each question.
    final_results = []
    current_index = 0

    for _ in questions:
        # Slice the portion of answers that belongs to the current question.
        chunk = all_generated_answers[current_index: current_index + num_retries]
        final_results.append(chunk)

        # Move the index to the start of the next chunk.
        current_index += num_retries

    return final_results


async def batch_ask_llm_with_context(
        data: List[dict],
        num_retries: int
) -> List[List[str]]:
    """
    Asynchronously generates alternative answers for a batch of questions.

    For each data sample, it constructs a prompt containing the question,
    context, and an instruction to AVOID the provided 'Memory answer'. It
    then runs all prompts in a single, efficient batch API call.

    Args:
        data (List[Dict[str, str]]): A list of dictionaries, where each
            dictionary represents a sample and must contain the keys:
            'Question', 'Context', and 'Memory answer'.
        num_retries (int): The number of times to generate an answer for
            EACH data sample.

    Returns:
        List[List[str]]: A list of answer lists. The structure mirrors the
            input `data`. The i-th inner list contains `num_retries` generated
            answers for the i-th sample in the input `data` list.
    """
    if not data:
        return []

    # --- Step 1: Flatten all questions into a single list of prompts ---
    all_prompts = []
    for sample in data:
        prompt = f"""
        Question: {sample['Question']}\n\n
        
        Context:\n
        {sample['Context']}\n\n        

        Important note:\n
        The answer is *not* {sample['Answer']['Memory answer']}
        """
        # Add the prompt 'num_retries' times for this specific question
        all_prompts.extend([prompt] * num_retries)

    # --- Step 2: Make a single, large API call ---
    # This will show just ONE tqdm bar (from qa_model's config) for the entire operation.
    nested_answers = await qa_context_model.async_run(all_prompts)

    # Flatten the result into a single list of answer strings
    all_generated_answers = [result[0] for result in nested_answers if result]

    # --- Step 3: Reconstruct the results into a list of lists ---
    # "Un-flatten" the single list of answers back into chunks, one for each question.
    final_results = []
    current_index = 0

    for _ in data:
        # Slice the portion of answers that belongs to the current question.
        chunk = all_generated_answers[current_index: current_index + num_retries]
        final_results.append(chunk)

        # Move the index to the start of the next chunk.
        current_index += num_retries

    return final_results

# ===================================================================================================================
# ===================================================================================================================
# ===================================================================================================================
# ===================================================================================================================
# ===================================================================================================================


# async def ask_llm(question: str, context: str, answer: str, num_retries: int) -> List[str]:
#     """
#     Uses the qa_model to ask the same question multiple times in parallel to
#     generate a variety of answers.
#
#     Args:
#         question (str): The question to be answered.
#         context (str): The context to use for answering the question.
#         answer (str): The answer to the question.
#         num_retries (int): The number of times to ask the question.
#
#     Returns:
#         List[str]: A list of all answers generated by the model.
#     """
#     prompt = f"""
#     You are a logical reasoner. Your task is to answer the question using only the information found in the context below.
#     \n\n
#     Rules:\n
#     - Do NOT use any external or real-world knowledge.\n
#     - Only rely on statements explicitly made in the context, even if they are factually incorrect.\n
#     - If the context contains contradictions, choose the answer that is best supported by the majority of consistent clues.\n
#     - Your answer must be a single entity or phrase clearly supported by the context.\n\n
#
#     Context:\n
#     {context}\n\n
#
#     Question:\n
#     {question}\n\n
#
#     Important note:
#     The answer is *not* {answer}
#     """
#     # Create a list of identical prompts for batch processing.
#     prompts = [prompt] * num_retries
#
#     # qa_model's output_format is 'list', so async_run returns a list of lists,
#     # e.g., [['answer 1'], ['answer 2'], ['answer 1']].
#     nested_answers = await qa_model.async_run(prompts)
#
#     # Flatten the list of lists into a single list of answer strings.
#     # We take the first element [0] from each sublist and handle potential empty responses.
#     answers = [result[0] for result in nested_answers if result]
#
#     return answers
#
#
# async def ask_llm_from_memory(question: str, num_retries: int) -> List[str]:
#     """
#     Uses the qa_model to ask the same question multiple times from memory,
#     generating a variety of answers while excluding a specific one.
#
#     Args:
#         question (str): The question to be answered.
#         num_retries (int): The number of times to ask in parallel.
#
#     Returns:
#         List[str]: A list of all answers generated by the model.
#     """
#     # The prompt is rewritten to encourage using general knowledge and to remove context.
#     prompt = f"""
#     You are a knowledgeable AI assistant with a vast internal knowledge base.
#     Your task is to answer the following question based on your general knowledge.
#     \n\n
#     Rules:
#     - Use your own knowledge to provide the answer.
#     - Your answer must be a single, concise entity or phrase.
#     \n\n
#     Question:
#     {question}
#     """
#     # Create a list of identical prompts for batch processing.
#     prompts = [prompt] * num_retries
#
#     # qa_model's output_format is 'list', so async_run returns a list of lists,
#     # e.g., [['answer 1'], ['answer 2'], ['answer 1']].
#     nested_answers = await qa_model.async_run(prompts)
#
#     # Flatten the list of lists into a single list of answer strings.
#     # We take the first element [0] from each sublist and handle potential empty responses.
#     answers = [result[0] for result in nested_answers if result]
#
#     return answers
#
#
# def plot_aggregated_pie(
#         data: List[Dict[str, float]],
#         title: str = "Overall Answer Distribution",
#         output_path: Optional[str] = None
# ) -> None:
#     """
#     Aggregates results from a list of score dictionaries and plots a single pie chart.
#
#     The function calculates the average score for each category across all
#     dictionaries in the list and displays these averages as a pie chart.
#
#     Args:
#         data: A list of dictionaries, where each dictionary maps an answer
#               category (str) to its score (float).
#         title: The title for the pie chart.
#         output_path: Optional file path to save the generated chart image.
#     """
#     if not data:
#         print("Data is empty. Nothing to plot.")
#         return
#
#     # --- Step 1: Aggregate the data ---
#     # Use defaultdict to easily handle sums without checking if a key exists.
#     sums = defaultdict(float)
#     num_samples = len(data)
#
#     # Sum up the scores for each category across all dictionaries.
#     for result_dict in data:
#         for key, value in result_dict.items():
#             sums[key] += value
#
#     # Calculate the average score for each category.
#     averages = {key: total_sum / num_samples for key, total_sum in sums.items()}
#
#     # --- Step 2: Prepare data for plotting ---
#     labels = list(averages.keys())
#     values = list(averages.values())
#
#     # Define a consistent color map for better visual clarity.
#     color_map = {
#         'Memory answer': 'skyblue',
#         'Counter answer 1': 'lightgreen',
#         'Counter answer 2': 'gold',
#         'Distractor answer': 'salmon',
#         'Invalid': 'silver',
#         'Other': 'lightgray'  # Fallback color
#     }
#     colors = [color_map.get(label, color_map['Other']) for label in labels]
#
#     # --- Step 3: Create the Pie Chart ---
#     fig, ax = plt.subplots(figsize=(12, 8))
#
#     # Use autopct to display the percentage on each slice.
#     # The return values (wedges, texts) are used for the legend.
#     wedges, texts, autotexts = ax.pie(
#         values,
#         autopct='%1.1f%%',
#         startangle=90,
#         colors=colors,
#         wedgeprops=dict(edgecolor='w')  # Add white lines between slices
#     )
#
#     # Improve the look of the percentage text.
#     plt.setp(autotexts, size=10, weight="bold", color="white")
#
#     # Ensure the pie chart is a circle.
#     ax.axis('equal')
#
#     # Add a title to the chart.
#     ax.set_title(title, fontsize=16, pad=20)
#
#     # Add a legend outside the pie chart for clarity.
#     ax.legend(
#         wedges,
#         labels,
#         title="Answer Categories",
#         loc="center left",
#         bbox_to_anchor=(1, 0, 0.5, 1)  # Position legend to the right
#     )
#
#     # --- Step 4: Display and save the plot ---
#     plt.tight_layout()
#
#     if output_path:
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         print(f"Chart saved to {output_path}")
#
#     plt.show()
#
#
# async def batch_ask_llm(batch_input: List[Dict[str, Any]], num_retries: int) -> List[str]:
#     """
#     Creates and runs a single large batch of prompts for the qa_model.
#
#     Args:
#         batch_input: A list of dictionaries, where each dict contains 'question',
#                      'context', and 'answer' for one sample.
#         num_retries: The number of times to generate a prompt for EACH item
#                      in the batch_input.
#
#     Returns:
#         A single flat list containing all generated answers from the model.
#     """
#     all_prompts = []
#     for item in batch_input:
#         prompt = f"""
#     You are a logical reasoner. Your task is to answer the question using only the information found in the context below.
#     \n\n
#     Rules:\n
#     - Do NOT use any external or real-world knowledge.\n
#     - Only rely on statements explicitly made in the context, even if they are factually incorrect.\n
#     - If the context contains contradictions, choose the answer that is best supported by the majority of consistent clues.\n
#     - Your answer must be a single entity or phrase clearly supported by the context.\n\n
#
#     Context:\n
#     {item['context']}\n\n
#
#     Question:\n
#     {item['question']}\n\n
#
#     Important note:
#     The answer is *not* {item['answer']}
#     """
#         # Add the same prompt 'num_retries' times for this sample
#         all_prompts.extend([prompt] * num_retries)
#
#     if not all_prompts:
#         return []
#
#     # A single, large API call for all prompts from all samples
#     nested_answers = await qa_model.async_run(all_prompts)
#
#     # Flatten the result into a single list
#     answers = [result[0] for result in nested_answers if result]
#     return answers
#
#
# # async def batch_judge_answers(generated_answers: List[str], answer_options_map: List[Dict[str, str]]) -> List[str]:
# #     """
# #     Creates and runs a single large batch of judgment prompts for the judge_model.
# #
# #     Args:
# #         generated_answers: A flat list of answers produced by batch_ask_llm.
# #         answer_options_map: A parallel list where each element is the 'options'
# #                             dictionary corresponding to the answer at the same
# #                             index in generated_answers.
# #
# #     Returns:
# #         A flat list of normalized/judged answers (e.g., 'Motown', 'Invalid').
# #     """
# #     if not generated_answers:
# #         return []
# #
# #     all_prompts = []
# #     # Use zip to iterate through the parallel lists
# #     for answer, options in zip(generated_answers, answer_options_map):
# #         option_values = list(options.values())
# #         prompt = f"""
# #         You are an impartial judge. Your task is to determine which of the following options an answer corresponds to.
# #
# #         Valid Options:
# #         {option_values}
# #
# #         Answer to Judge:
# #         "{answer}"
# #
# #         Which option does the answer correspond to? Respond with ONLY the text of the matching option from the list.
# #         If the answer does not correspond to any of the options, respond with the single word 'Invalid'.
# #         """
# #         all_prompts.append(prompt)
# #
# #     # A single, large API call for all judgment prompts
# #     judged_responses = await judge_model.async_run(all_prompts)
# #
# #     # Flatten and clean the results
# #     normalized_answers = [result[0].strip() for result in judged_responses if result]
# #     return normalized_answers
#
#
# def _calculate_probabilities(normalized_answers_chunk: List[str], options: Dict[str, str]) -> Dict[str, float]:
#     """Helper to calculate probabilities for a chunk of judged answers."""
#     if not normalized_answers_chunk:
#         return {key: 0.0 for key in options.keys()}
#
#     value_to_key_map = {value: key for key, value in options.items()}
#     result_probabilities = {key: 0.0 for key in options.keys()}
#
#     total_count = len(normalized_answers_chunk)
#     answer_counts = Counter(normalized_answers_chunk)
#
#     for value, count in answer_counts.items():
#         key = value_to_key_map.get(value)
#         if key:
#             result_probabilities[key] = count / total_count
#         else:
#             result_probabilities[value] = count / total_count
#
#     return result_probabilities
#
#
# async def run_batch_evaluation(dataset: List[Dict[str, Any]], num_retries: int) -> List[List[Any]]:
#     """
#     Efficiently processes a dataset by batching all LLM calls.
#
#     Args:
#         dataset: A list of QA sample dictionaries.
#         num_retries: The number of answers to generate for each sample.
#
#     Returns:
#         A list of results, where each item is in the format
#         [answer_1_length, answer_2_length, summary_results_dict].
#     """
#     print(f"Starting batch processing for {len(dataset)} samples...")
#
#     # --- Stage 1: Generation ---
#     # Prepare the input for the generation stage
#     ask_llm_input = [
#         {
#             "question": sample['Question'],
#             "context": sample['Context'],
#             "answer": sample['Answer']['Memory answer']
#         }
#         for sample in dataset
#     ]
#     print(f"Generating {len(ask_llm_input) * num_retries} answers in a single batch...")
#     all_generated_answers = await batch_ask_llm(ask_llm_input, num_retries)
#
#     # --- Stage 2: Judgment ---
#     # Create a parallel list that maps each generated answer back to its original options
#     options_map = [sample['Answer'] for sample in dataset for _ in range(num_retries)]
#
#     # The map might be longer than the actual answers if some failed, so trim it.
#     trimmed_options_map = options_map[:len(all_generated_answers)]
#
#     print(f"Judging {len(all_generated_answers)} answers in a single batch...")
#     all_normalized_answers = await batch_judge_answers(all_generated_answers, trimmed_options_map)
#
#     # --- Stage 3: Reconstruction ---
#     # Now, chunk the flat list of normalized answers back into groups for each sample.
#     final_results = []
#     answers_processed_so_far = 0
#     for i, sample in enumerate(dataset):
#         # We need to know how many answers were successfully generated for *this specific sample*
#         # by looking at the original API output structure. Let's assume for simplicity
#         # it's always num_retries, but a more robust solution might need to count them.
#         # This is a simplification that works if the API is reliable.
#         num_answers_for_this_sample = num_retries  # Simplified assumption
#
#         chunk_start = i * num_retries
#         chunk_end = chunk_start + num_retries
#
#         # We slice the judged answers that correspond to this sample
#         normalized_answers_chunk = all_normalized_answers[chunk_start:chunk_end]
#
#         # Calculate the final probability summary for this chunk
#         summary_results = _calculate_probabilities(normalized_answers_chunk, sample['Answer'])
#
#         # Append the final formatted result
#         final_results.append([sample['answer 1 length'], sample['answer 2 length'], summary_results])
#
#     print("Batch processing complete.")
#     return final_results