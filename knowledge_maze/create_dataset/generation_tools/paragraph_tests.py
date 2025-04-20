"""
Analyzes paragraph quality in relation to QA sample answers.

This module evaluates paragraphs generated for a question-answering dataset
by checking for the presence or absence of specific answer types within the
paragraph text. It loads the QA dataset (`REASONING_DATASET`) and the
corresponding paragraph dataset (`PARAGRAPH_DATASET`) from specified file paths
derived from environment variables.

The primary function, `paragraph_preview_test`, performs a detailed analysis:
- It iterates through each QA sample and its associated paragraphs.
- It checks if the original ('Memory') answer appears in the text of 'Chain'
  and 'Edge' paragraphs separately, using case-insensitive and
  punctuation-normalized comparisons.
- It checks if the counter answer appears in the text of 'Edge' paragraphs.
- It identifies "Ideal Counter Samples" where the original answer is absent
  from all paragraphs, but both counter answers are present in their
  respective 'Edge' paragraphs.
- It calculates and prints summary statistics for these checks.
- It displays examples of paragraphs that failed checks (e.g., contained
  the original answer or missed the counter answer) using the imported
  `display_paragraph` function.

Helper functions are used internally for text normalization and formatting the
output summary and examples.
"""
import os
import json
import string
from typing import List, Dict, Optional

from tqdm import tqdm
from dotenv import load_dotenv

from knowledge_maze.create_dataset.create_paragraphs import display_paragraph

load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')


# Load the reasoning dataset QA:
QA_FILE_PATH = os.path.join(REPO_PATH, "data_files", "knowledge_maze_qa.json")
with open(QA_FILE_PATH, "r", encoding="utf-8") as file:
    REASONING_DATASET = json.load(file)

# Load the reasoning dataset paragraphs:
PARAGRAPH_FILE_PATH = os.path.join(REPO_PATH, "data_files", "knowledge_maze_paragraphs.json")
with open(PARAGRAPH_FILE_PATH, "r", encoding="utf-8") as file:
    PARAGRAPH_DATASET = json.load(file)


def normalize_text_for_comparison(text: Optional[str]) -> Optional[str]:
    """Converts text to lowercase and removes punctuation for comparison."""
    if text is None:
        return None
    translator = str.maketrans('', '', string.punctuation)
    return str(text).lower().translate(translator)


def paragraph_preview_test(max_examples_to_show: int = 2) -> None:
    """
    Analyzes paragraphs for answer containment, providing detailed stats and examples.

    Iterates through QA samples, checking each associated paragraph for:
    1. Presence of the original ('Memory') answer, tracked separately for
       'Chain' and 'Edge' paragraphs.
    2. Presence of the correct counter answer in 'Edge' paragraphs.

    Calculates and prints summary statistics, including the breakdown of original
    answer containment by paragraph type. Also identifies and displays random
    examples of problematic paragraphs.

    Args:
        max_examples_to_show (int): Max examples per problematic category. Defaults to 2.

    Returns:
        None. Prints results and examples directly.
    """
    if not REASONING_DATASET:
        print("⚠️ REASONING_DATASET is empty. Cannot perform analysis.")
        return
    if not PARAGRAPH_DATASET:
        print("⚠️ PARAGRAPH_DATASET is empty. Cannot perform analysis.")
        return

    # --- Counters ---
    stats = {
        "total_paragraphs_processed": 0,
        "total_chain_paragraphs_processed": 0, # Added
        "total_edge_paragraphs_processed": 0,
        "chain_para_contains_original": 0,   # Added
        "edge_para_contains_original": 0,    # Added
        "edge_para_contains_counter": 0,
        "ideal_counter_samples": 0,
        "samples_analyzed": 0,
        "errors_paragraph_lookup": 0,
        "errors_sample_structure": 0,
    }

    # --- Lists for example display ---
    paras_with_original_ans_ids: List[str] = []
    edge_paras_missing_counter_ids: List[str] = []

    print("Analyzing paragraphs and samples for answer containment...")

    # --- Main Analysis Loop ---
    for sample_idx, sample in enumerate(tqdm(REASONING_DATASET, desc="Analyzing Samples")):
        try:
            sample_had_successful_paragraph = False

            # --- 1. Extract and Normalize Sample Answers ---
            original_answer = sample.get('Answer', {}).get('Memory answer')
            counter_answer_1 = sample.get('Answer', {}).get('Counter answer 1')
            counter_answer_2 = sample.get('Answer', {}).get('Counter answer 2')

            norm_original_answer = normalize_text_for_comparison(original_answer)
            norm_counter_1 = normalize_text_for_comparison(counter_answer_1)
            norm_counter_2 = normalize_text_for_comparison(counter_answer_2)

            # --- 2. Initialize Sample-Level Flags ---
            sample_flags = {
                "p1_has_original": False, "p2_has_original": False,
                "p1_edge_has_counter": False, "p2_edge_has_counter": False,
                "p1_edge_exists": False, "p2_edge_exists": False
            }

            # --- 3. Process Paragraphs in Both Paths ---
            path_details = [
                ('Reasoning path 1', norm_counter_1, 1),
                ('Reasoning path 2', norm_counter_2, 2)
            ]

            for path_key, norm_path_counter_ans, path_num in path_details:
                reasoning_path = sample.get(path_key, [])
                if not reasoning_path: continue

                for step in reasoning_path:
                    paragraph_id = step.get('Paragraph')
                    if not paragraph_id: continue

                    # --- 3a. Look Up and Normalize Paragraph ---
                    paragraph_entry = PARAGRAPH_DATASET.get(paragraph_id)
                    if not paragraph_entry or not isinstance(paragraph_entry, dict):
                        stats["errors_paragraph_lookup"] += 1
                        if stats["errors_paragraph_lookup"] <= 10:
                             print(f"⚠️ Missing PARAGRAPH_DATASET entry for ID '{paragraph_id}' in sample {sample_idx}. Skipping.")
                        continue

                    paragraph_text = paragraph_entry.get('Text')
                    paragraph_type = paragraph_entry.get('Type') # Expect 'Chain' or 'Edge'
                    if not paragraph_text or not isinstance(paragraph_text, str) or not paragraph_type:
                        continue # Skip paragraphs with invalid text or missing type

                    stats["total_paragraphs_processed"] += 1
                    sample_had_successful_paragraph = True
                    norm_paragraph_text = normalize_text_for_comparison(paragraph_text)
                    if not norm_paragraph_text: continue

                    is_edge = (paragraph_type == 'Edge')

                    # --- Update type-specific processed counts ---
                    if is_edge:
                        stats["total_edge_paragraphs_processed"] += 1
                        if path_num == 1: sample_flags["p1_edge_exists"] = True
                        else: sample_flags["p2_edge_exists"] = True
                    else: # Assumed 'Chain'
                        stats["total_chain_paragraphs_processed"] += 1


                    # --- 3b. Check Original Answer Containment ---
                    original_found_here = False # Flag for this specific paragraph
                    if norm_original_answer and norm_original_answer in norm_paragraph_text:
                        original_found_here = True
                        # Store for example display (regardless of type)
                        paras_with_original_ans_ids.append(paragraph_id)
                        # Sample-Level Check Update
                        if path_num == 1: sample_flags["p1_has_original"] = True
                        else: sample_flags["p2_has_original"] = True

                    # --- Update type-specific original answer counts ---
                    if original_found_here:
                        if is_edge:
                            stats["edge_para_contains_original"] += 1
                        else: # Assumed 'Chain'
                            stats["chain_para_contains_original"] += 1


                    # --- 3c. Check Counter Answer Containment (Edge Only) ---
                    if is_edge:
                        counter_found_here = False
                        if norm_path_counter_ans and norm_path_counter_ans in norm_paragraph_text:
                            stats["edge_para_contains_counter"] += 1
                            if path_num == 1: sample_flags["p1_edge_has_counter"] = True
                            else: sample_flags["p2_edge_has_counter"] = True
                            counter_found_here = True

                        # Store for example display if counter was NOT found
                        if not counter_found_here and norm_path_counter_ans:
                            edge_paras_missing_counter_ids.append(paragraph_id)
            # --- End loop for steps ---
        # --- End loop for paths ---

            # --- 4. Evaluate Sample-Level Criteria ---
            if sample_flags["p1_edge_exists"] and sample_flags["p2_edge_exists"]:
                is_ideal_counter_sample = (
                    not sample_flags["p1_has_original"] and
                    not sample_flags["p2_has_original"] and
                    sample_flags["p1_edge_has_counter"] and
                    sample_flags["p2_edge_has_counter"]
                )
                if is_ideal_counter_sample:
                    stats["ideal_counter_samples"] += 1

            # Increment analyzed count if at least one paragraph was processed
            if sample_had_successful_paragraph:
                stats["samples_analyzed"] += 1

        except Exception as e:
            stats["errors_sample_structure"] += 1
            print(f"\n⚠️ Error processing sample index {sample_idx}: {e}. Skipping sample.")

    # --- 5. Calculate and Print Statistics ---
    _print_analysis_summary(stats) # Pass the updated stats dict

    # --- 6. Display Bad Examples ---
    _display_bad_paragraph_examples(
        paras_with_original_ans_ids,
        edge_paras_missing_counter_ids,
        max_examples_to_show
    )


def _print_analysis_summary(stats: Dict[str, int]) -> None:
    """Helper function to print the calculated statistics."""
    print("\n" + "=" * 60)
    print("------------ Paragraph & Sample Analysis Results -----------")
    print("=" * 60)
    print(f"Total QA samples in dataset: {len(REASONING_DATASET)}")
    print(f"Samples analyzed: {stats['samples_analyzed']}")
    if stats["errors_sample_structure"] > 0:
         print(f"Errors during sample processing (structure/answers): {stats['errors_sample_structure']}")
    if stats["errors_paragraph_lookup"] > 0:
         print(f"Errors during paragraph lookup (missing IDs): {stats['errors_paragraph_lookup']}")

    print("\n------------------ Paragraph-Level Stats  ------------------")
    print(f"Total paragraphs processed: {stats['total_paragraphs_processed']}")

    # --- Detailed Original Answer Stats ---
    print("-" * 60)
    print("Original Answer Containment:")
    if stats["total_chain_paragraphs_processed"] > 0:
        percent_chain_original = (stats["chain_para_contains_original"] / stats["total_chain_paragraphs_processed"]) * 100
        print(f"  - In 'Chain' Paras: {stats['chain_para_contains_original']} / {stats['total_chain_paragraphs_processed']} ({percent_chain_original:.2f}%)")
    else:
        print("  - No 'Chain' paragraphs processed.")

    if stats["total_edge_paragraphs_processed"] > 0:
         percent_edge_original = (stats["edge_para_contains_original"] / stats["total_edge_paragraphs_processed"]) * 100
         print(f"  - In 'Edge' Paras:  {stats['edge_para_contains_original']} / {stats['total_edge_paragraphs_processed']} ({percent_edge_original:.2f}%)")
    else:
         print("  - No 'Edge' paragraphs processed.")
    # --- END CHANGE ---

    print("-" * 60)
    print("Counter Answer Containment (Edge Paras Only):")
    if stats["total_edge_paragraphs_processed"] > 0:
        percent_edge_counter = (stats["edge_para_contains_counter"] / stats["total_edge_paragraphs_processed"]) * 100
        # print(f"Total 'Edge' paragraphs processed: {stats['total_edge_paragraphs_processed']}") # Redundant now
        print(f"  - 'Edge' Paras containing Counter Answer: {stats['edge_para_contains_counter']} / {stats['total_edge_paragraphs_processed']}")
        print(f"  - Percentage containing Counter Answer:   {percent_edge_counter:.2f}%")
    else:
        print("  - No 'Edge' paragraphs found or processed.")

    print("\n" + "=" * 60)
    print("-------------------- Sample-Level Stats --------------------")
    print("=" * 60)
    if stats["samples_analyzed"] > 0:
        percent_ideal_counter = (stats["ideal_counter_samples"] / stats["samples_analyzed"]) * 100
        print(f"Samples meeting 'Ideal Counter' criteria*: {stats['ideal_counter_samples']}")
        print(f"Percentage of analyzed samples meeting criteria: {percent_ideal_counter:.2f}%")
        print("* (Original ans absent in all path paras AND both edge paras\n   contain respective counter ans)")
    else:
         print("No samples were analyzed for the 'Ideal Counter' criteria.")


def _display_bad_paragraph_examples(
    paras_with_original_ids: List[str],
    edge_missing_counter_ids: List[str],
    max_examples: int
) -> None:
    """Helper function to display random examples of problematic paragraphs."""
    # --- This helper function remains unchanged ---
    print("\n" + "=" * 60)
    print("---------- Bad Paragraph Examples (Random Sample) ----------")
    print("=" * 60)

    # --- Examples: Original Answer Found ---
    print("\n--- Examples: Paragraphs Containing Original Answer ---")
    if paras_with_original_ids:
        print(f"(Showing {len(paras_with_original_ids)} found)")
        # Use set to get unique IDs before sampling if duplicates are possible and unwanted in examples
        unique_ids = list(set(paras_with_original_ids))
        for pid in unique_ids:
            try:
                display_paragraph(pid) # Assuming display_paragraph exists
            except Exception as e:
                print(f"\nError displaying example paragraph '{pid}': {e}")
    else:
        print("No examples found where original answer was in the paragraph text.")

    # --- Examples: Edge Paragraph Missing Counter Answer ---
    print("\n--- Examples: Edge Paragraphs MISSING Counter Answer ---")
    if edge_missing_counter_ids:
        print(f"(Showing {len(edge_missing_counter_ids)} found)")
        unique_ids = list(set(edge_missing_counter_ids))
        for pid in unique_ids:
             try:
                display_paragraph(pid) # Assuming display_paragraph exists
             except Exception as e:
                print(f"\nError displaying example paragraph '{pid}': {e}")
    else:
        print("No examples found where edge paragraphs were missing their counter answer.")