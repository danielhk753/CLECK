from typing import List, Dict, Any, Callable, Awaitable

from knowledge_maze.evaluation_old.llm_judge import batch_judge_answers
from knowledge_maze.evaluation_old.ask_llm import batch_ask_llm_from_memory, batch_ask_llm_with_context

# async def run_memory_evaluation(
#         dataset: List[Dict[str, Any]],
#         num_retries: int
# ) -> list[Any] | list[str]:
#     """
#     Orchestrates a full, batch-processed memory evaluation_old on a given dataset.
#
#     This function performs the entire evaluation_old pipeline:
#     1. Extracts all questions from the dataset.
#     2. Calls `batch_ask_llm_from_memory` to generate answers for all questions
#        in a single, efficient API call.
#     3. Prepares the generated answers for the judging step.
#     4. Calls `batch_judge_answers` to classify all answers in another single
#        API call.
#     5. Reconstructs the final results into the desired output format.
#
#     Args:
#         dataset (List[Dict[str, Any]]): A list of QA sample dictionaries. Each
#             dictionary must contain 'Question', 'Answer', 'answer 1 length',
#             and 'answer 2 length'.
#         num_retries (int): The number of answers to generate for each question.
#
#     Returns:
#         List[List[Any]]: A list of final results, where each item is in the
#             format `[answer_1_length, answer_2_length, summary_results_dict]`.
#     """
#     if not dataset:
#         return []
#
#     print(f"--- Starting Memory Evaluation for {len(dataset)} Samples ---")
#
#     # --- Stage 1: Batch Generation ---
#     # Extract all questions to prepare for the first batch call.
#     all_questions = [sample['Question'] for sample in dataset]
#     print(f"\n[Stage 1/2] Generating {len(all_questions) * num_retries} answers...")
#
#     # This single await call generates all answers for all questions.
#     # It will show one tqdm bar from the underlying qa_model.
#     all_answer_lists = await batch_ask_llm_from_memory(all_questions, num_retries=num_retries)
#     print("Answer generation complete.")
#
#     # --- Stage 2: Batch Judging ---
#     # Prepare the input for the batch_judge_answers function.
#     batch_jobs_for_judging = []
#     for i, sample in enumerate(dataset):
#         batch_jobs_for_judging.append({
#             'answers': all_answer_lists[i],  # The list of answers for this sample
#             'options': sample['Answer']  # The options dict for this sample
#         })
#
#     print(f"\n[Stage 2/2] Judging {sum(len(job['answers']) for job in batch_jobs_for_judging)} answers...")
#
#     # This single await call judges all the answers.
#     all_summary_results = await batch_judge_answers(batch_jobs_for_judging)
#     print("Judging complete.")
#
#     # --- Stage 3: Final Reconstruction ---
#     # Combine the original sample metadata with the final summary results.
#     final_output = []
#     for i, sample in enumerate(dataset):
#         final_output.append(all_summary_results[i])
#
#     print("\n--- Memory Evaluation Complete ---")
#     return final_output
#
#
# async def run_context_evaluation(
#     dataset: List[Dict[str, Any]],
#     num_retries: int
# ) -> List[Dict[str, Any]]:
#     """
#     Orchestrates a full, batch-processed context evaluation_old on a given dataset.
#
#     This function performs the entire evaluation_old pipeline:
#     1. Calls `batch_ask_llm_with_context` to generate alternative answers
#        based on the provided context, while being instructed to avoid the
#        'Memory answer'. This is done in a single, efficient API call.
#     2. Prepares the generated answers for the judging step.
#     3. Calls `batch_judge_answers` to classify all generated answers in another
#        single API call.
#     4. Returns the final judging results.
#
#     Args:
#         dataset (List[Dict[str, Any]]): A list of sample dictionaries. Each
#             dictionary must contain the keys: 'Question', 'Context',
#             'Memory answer', and 'Answer' (for the judging options).
#         num_retries (int): The number of alternative answers to generate for
#             each question.
#
#     Returns:
#         List[Dict[str, Any]]: A list of summary result dictionaries, one for
#             each sample in the input dataset.
#     """
#     if not dataset:
#         return []
#
#     print(f"--- Starting Context Evaluation for {len(dataset)} Samples ---")
#
#     # --- Stage 1: Batch Generation ---
#     total_answers_to_gen = len(dataset) * num_retries
#     print(f"\n[Stage 1/2] Generating {total_answers_to_gen} alternative answers...")
#
#     # This single await call generates all answers for all samples.
#     # It requires the full dataset, not just the questions.
#     all_answer_lists = await batch_ask_llm_with_context(dataset, num_retries=num_retries)
#     print("Answer generation complete.")
#
#     # --- Stage 2: Batch Judging ---
#     # Prepare the input for the batch_judge_answers function.
#     batch_jobs_for_judging = []
#     for i, sample in enumerate(dataset):
#         batch_jobs_for_judging.append({
#             'answers': all_answer_lists[i],  # The list of generated answers for this sample
#             'options': sample['Answer']      # The judging options dict for this sample
#         })
#
#     total_answers_to_judge = sum(len(job['answers']) for job in batch_jobs_for_judging)
#     print(f"\n[Stage 2/2] Judging {total_answers_to_judge} answers...")
#
#     # This single await call judges all the generated answers.
#     all_summary_results = await batch_judge_answers(batch_jobs_for_judging)
#     print("Judging complete.")
#
#     # --- Stage 3: Final Reconstruction ---
#     # The output from batch_judge_answers is already in the desired final format.
#     # We simply return it directly.
#     print("\n--- Context Evaluation Complete ---")
#     return all_summary_results

async def _run_evaluation_pipeline(
    dataset: List[Dict[str, Any]],
    num_retries: int,
    evaluation_name: str,
    generation_function: Callable[..., Awaitable[List[List[str]]]],
    input_preparer: Callable[[List[Dict[str, Any]]], Any]
) -> List[Dict[str, Any]]:
    """
    A generic, internal pipeline for running a batch evaluation_old.

    This function abstracts the common steps of generation and judging.
    """
    if not dataset:
        return []

    print(f"--- Starting {evaluation_name} Evaluation for {len(dataset)} Samples ---")

    # --- Stage 1: Batch Generation (Customizable) ---
    # Prepare the specific input required by the generation function.
    generation_input = input_preparer(dataset)
    total_answers_to_gen = len(dataset) * num_retries
    print(f"\n[Stage 1/2] Generating {total_answers_to_gen} answers...")

    # Call the provided generation function.
    all_answer_lists = await generation_function(generation_input, num_retries=num_retries)
    print("Answer generation complete.")

    # --- Stage 2: Batch Judging (Common) ---
    batch_jobs_for_judging = []
    for i, sample in enumerate(dataset):
        batch_jobs_for_judging.append({
            'answers': all_answer_lists[i],
            'options': sample['Answer']
        })

    total_answers_to_judge = sum(len(job['answers']) for job in batch_jobs_for_judging)
    print(f"\n[Stage 2/2] Judging {total_answers_to_judge} answers...")

    all_summary_results = await batch_judge_answers(batch_jobs_for_judging)
    print("Judging complete.")

    print(f"\n--- {evaluation_name} Evaluation Complete ---")
    return all_summary_results


async def run_memory_evaluation(
    dataset: List[Dict[str, Any]],
    num_retries: int
) -> List[Dict[str, Any]]:
    """
    Orchestrates a full, batch-processed memory evaluation_old on a given dataset.

    This is a wrapper that configures and runs the generic evaluation_old pipeline
    for a "memory-based" answer generation task.

    Args:
        dataset (List[Dict[str, Any]]): A list of sample dictionaries. Each
            dictionary must contain 'Question' and 'Answer'.
        num_retries (int): The number of answers to generate for each question.

    Returns:
        List[Dict[str, Any]]: A list of summary result dictionaries, one for
            each sample in the input dataset.
    """
    return await _run_evaluation_pipeline(
        dataset=dataset,
        num_retries=num_retries,
        evaluation_name="Memory",
        generation_function=batch_ask_llm_from_memory,
        # For memory, the generator needs only a list of the questions.
        input_preparer=lambda data: [sample['Question'] for sample in data]
    )


async def run_context_evaluation(
    dataset: List[Dict[str, Any]],
    num_retries: int
) -> List[Dict[str, Any]]:
    """
    Orchestrates a full, batch-processed context evaluation_old on a given dataset.

    This is a wrapper that configures and runs the generic evaluation_old pipeline
    for a "context-based" answer generation task.

    Args:
        dataset (List[Dict[str, Any]]): A list of sample dictionaries. Each
            dictionary must contain 'Question', 'Context', 'Memory answer',
            and 'Answer'.
        num_retries (int): The number of alternative answers to generate.

    Returns:
        List[Dict[str, Any]]: A list of summary result dictionaries, one for
            each sample in the input dataset.
    """
    return await _run_evaluation_pipeline(
        dataset=dataset,
        num_retries=num_retries,
        evaluation_name="Context",
        generation_function=batch_ask_llm_with_context,
        # For context, the generator needs the full dataset dictionary.
        input_preparer=lambda data: data
    )