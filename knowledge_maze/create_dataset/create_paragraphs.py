"""
Manages paragraph generation, analysis, and visualization for QA reasoning paths.

This module provides functionalities to:
1.  Generate paragraph text based on information extracted from QA reasoning paths
    using an external generation tool (`create_paragraphs`).
2.  Retrieve detailed context (questions, answers) for a given paragraph ID
    by searching through preloaded QA and paragraph datasets (`get_paragraph_info`).
3.  Display a formatted preview of a specific paragraph and its context
    (`display_paragraph`).
4.  Analyze the generated paragraphs to check for the presence of original and
    counter answers within their text (`paragraph_preview_test`).
5.  Calculate the total token count (using `tiktoken`) for all paragraphs
    associated with each QA sample and plot a histogram of these counts
    (`plot_hist_token_size_per_sample`).

The module loads the primary QA data (`knowledge_maze_qa.json`) and the
generated/pre-existing paragraph data (`knowledge_maze_paragraphs.json`) into
global variables (`REASONING_DATASET`, `PARAGRAPH_DATASET`) upon import.
File paths are constructed using a base repository path loaded from environment
variables (`REPO_PATH` via `dotenv`).
"""
import os
import json
from typing import List, Dict, Any
from collections import defaultdict

import tiktoken
from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from knowledge_maze.create_dataset.create_data_qa import wrap_text
from knowledge_maze.create_dataset.generation_tools.gen_paragraphs import generate_paragraph


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


async def create_paragraphs(
    qa_data: List[Dict[str, Any]],
    output_path: str = None
) -> Dict[str, Dict[str, str]]:
    """
    Generates paragraph text, processing 'Chain' types separately based on answer type.

    Processes QA samples, categorizing steps into 'Chain' or 'Edge'.
    Collects 'Chain' paragraph information grouped by the sample's 'Answer type'.
    Calls `generate_paragraph` asynchronously for each 'Answer type' group of
    'Chain' paragraphs, passing the type information within the input.
    Calls `generate_paragraph` once for all 'Edge' paragraphs.
    Combines results into a final dictionary mapping paragraph IDs to their
    generated text and original type ('Chain' or 'Edge').

    Args:
        qa_data: A list of QA sample dictionaries, expected to contain keys
            like 'Answer type', 'Answer' (with 'Counter answer 1/2'),
            'Reasoning path 1'/'Reasoning path 2'. Each path step should
            contain 'Paragraph', 'Previous question', and maybe 'Next question'.
        output_path: Optional file path to save the generated paragraph data.

    Returns:
        A dictionary mapping paragraph IDs (str) to {'Text': str, 'Type': str}.

    Raises:
        Any exceptions from `generate_paragraph` or KeyErrors from data structure.
    """
    # --- Data Structures ---
    # Group middle paragraph info by the answer type of the sample
    # Keep edge paragraphs separate (processed in one batch)
    middle_data_by_type = defaultdict(lambda: {'ids': [], 'info': []})
    edge_ids: List[str] = []
    edge_info: List[str] = [] # Edge info remains just the prompt string

    # --- 1. Extract and Group Paragraph Info ---
    print(f"Processing {len(qa_data)} QA samples to extract and group paragraph info...")
    for sample in qa_data:
        # Ensure 'Answer type' is a valid string, default otherwise
        answer_type = sample.get('Answer type', 'Unknown Type')
        if not isinstance(answer_type, str) or not answer_type:
             answer_type = 'Unknown Type'

        answers = sample.get('Answer', {})
        path_details = [
            ('Reasoning path 1', answers.get('Counter answer 1', 'N/A')),
            ('Reasoning path 2', answers.get('Counter answer 2', 'N/A'))
        ]

        for path_key, counter_answer in path_details:
            reasoning_path = sample.get(path_key, [])
            for step in reasoning_path:
                paragraph_id = step.get('Paragraph')
                prev_question = step.get('Previous question')
                next_question = step.get('Next question')

                if not paragraph_id or not prev_question: continue

                if next_question:
                    # Middle ('Chain') step - group by answer_type
                    info_dict = {
                        'Text': (f"Questions:\n"
                                 f"- {prev_question}\n"
                                 f"- {next_question}\n"),
                        'Type': answer_type # Include the type here
                    }
                    middle_data_by_type[answer_type]['ids'].append(paragraph_id)
                    middle_data_by_type[answer_type]['info'].append(info_dict)
                else:
                    # Edge ('Edge') step - collect normally
                    info_str = (f"Question: {prev_question}\n"
                                f"Answer: {counter_answer}\n")
                    edge_ids.append(paragraph_id)
                    edge_info.append(info_str)

    # --- 2. Generate Paragraphs Asynchronously ---
    all_middle_paragraphs_map: Dict[str, str] = {} # {paragraph_id: generated_text}

    print("Generating 'Chain' paragraphs (grouped by Answer Type)...")
    for answer_type, type_data in middle_data_by_type.items():
        ids_batch = type_data['ids']
        info_batch = [text['Text'] for text in type_data['info']] # This is now List[Dict[str, Any]]

        if not info_batch: continue # Skip empty batches

        print(f"--> Generating {len(info_batch)} paragraphs for type: '{answer_type}'")
        try:
            # *** Call generate_paragraph with the list of dictionaries for this type ***
            generated_texts = await generate_paragraph(info_batch, mode='Chain', ans_type=answer_type)

            if len(ids_batch) == len(generated_texts):
                for pid, text in zip(ids_batch, generated_texts):
                    all_middle_paragraphs_map[pid] = text
            else:
                print(f"Warning: Mismatch for type '{answer_type}'. IDs: {len(ids_batch)}, Texts: {len(generated_texts)}. Mapping based on shorter length.")
                for i in range(min(len(ids_batch), len(generated_texts))):
                    all_middle_paragraphs_map[ids_batch[i]] = generated_texts[i]
        except Exception as e:
             print(f"Error generating paragraphs for type '{answer_type}': {e}")


    print(f"--> Generating {len(edge_info)} 'Edge' paragraphs...")
    edge_paragraphs: List[str] = []
    if edge_info:
        try:
            edge_paragraphs = await generate_paragraph(edge_info, mode='Edge', ans_type=None)
            if len(edge_ids) != len(edge_paragraphs):
                print(f"Warning: Mismatch in edge IDs ({len(edge_ids)}) and generated edge paragraphs ({len(edge_paragraphs)})")
        except Exception as e:
             print(f"Error generating edge paragraphs: {e}")


    # --- 3. Combine Results into Final Dictionary ---
    print("Combining generated paragraphs with IDs...")
    paragraph_data: Dict[str, Dict[str, str]] = {}

    # Add middle paragraphs
    for pid, text in all_middle_paragraphs_map.items():
        paragraph_data[pid] = {'Text': text, 'Type': 'Chain'}

    # Add edge paragraphs (handle potential length mismatch)
    for i in range(min(len(edge_ids), len(edge_paragraphs))):
        paragraph_data[edge_ids[i]] = {'Text': edge_paragraphs[i], 'Type': 'Edge'}

    # --- 4. Save if Path Provided ---
    if output_path:
        print(f"Saving paragraph data ({len(paragraph_data)} entries) to {output_path}...")
        try:
            with open(output_path, "w", encoding="utf-8") as out_file:
                json.dump(paragraph_data, out_file, indent=4)
            print("Save complete.")
        except IOError as e:
            print(f"Error saving file to {output_path}: {e}")
        except Exception as e:
             print(f"An unexpected error occurred during saving: {e}")


    return paragraph_data


def get_paragraph_info(paragraph_id: str) -> dict:
    """
    Retrieves detailed information associated with a specific paragraph ID.

    Fetches the basic type and text of the paragraph from PARAGRAPH_DATASET.
    Then, it searches through REASONING_DATASET to find the specific reasoning
    path and sample context this paragraph belongs to. This context includes
    the preceding/succeeding questions and the associated original and
    counter answers for that path within the sample.

    Args:
        paragraph_id: The unique identifier string for the paragraph.

    Returns:
        A dictionary containing information about the paragraph:
            - 'Type' (str): The type of the paragraph (e.g., 'Chain', 'Answer')
              obtained from PARAGRAPH_DATASET.
            - 'Text' (str): The text content of the paragraph from
              PARAGRAPH_DATASET.
            - 'Questions' (List[str] | None): A list containing the
              'Previous question' and potentially the 'Next question' if the
              paragraph type is 'Chain'. Contains only 'Previous question'
              otherwise. Is None if the paragraph context isn't found in
              REASONING_DATASET.
            - 'Original answer' (Any | None): The 'Memory answer' from the
              sample the paragraph belongs to. Is None if context isn't found.
            - 'Counter answer' (Any | None): The specific counter answer
              ('Counter answer 1' or 'Counter answer 2') corresponding to the
              reasoning path the paragraph was found in. Is None if context
              isn't found.
    """
    info = {
        # Initialize with basic info and context placeholders
        'Type': PARAGRAPH_DATASET[paragraph_id]['Type'],
        'Text': PARAGRAPH_DATASET[paragraph_id]['Text'],
        'Questions': None,
        'Original answer': None,
        'Counter answer': None
    }

    # Search for the paragraph's context within the reasoning dataset
    found_context = False
    for sample in REASONING_DATASET:
        for reasoning_path_key in ['Reasoning path 1', 'Reasoning path 2']:
            if reasoning_path_key not in sample: continue  # Skip if path missing

            for paragraph_step in sample[reasoning_path_key]:
                if paragraph_id == paragraph_step.get('Paragraph'):
                    # Found the context for the paragraph ID
                    paragraph_type = info['Type']  # Use type already fetched

                    # Determine questions based on type
                    if paragraph_type == 'Chain':
                        info['Questions'] = [
                            paragraph_step.get('Previous question'),
                            paragraph_step.get('Next question')  # Might be None
                        ]
                    else:  # Assume 'Answer' type or similar single-question context
                        info['Questions'] = [
                            paragraph_step.get('Previous question')
                        ]

                    # Get answers associated with the sample and path
                    info['Original answer'] = sample.get('Answer', {}).get('Memory answer')
                    if reasoning_path_key == 'Reasoning path 1':
                        info['Counter answer'] = sample.get('Answer', {}).get('Counter answer 1')
                    else:  # Assumed to be 'Reasoning path 2'
                        info['Counter answer'] = sample.get('Answer', {}).get('Counter answer 2')

                    found_context = True
                    break  # Stop searching steps within this path
            if found_context:
                break  # Stop searching paths within this sample
        if found_context:
            break  # Stop searching samples

    # If context wasn't found, 'Questions', 'Original answer', 'Counter answer'
    # will retain their initial None values.
    return info


def display_paragraph(paragraph_id: str) -> None:
    """
    Fetches paragraph information and displays it in a formatted manner.

    Calls `get_paragraph_info` to retrieve details about the paragraph specified
    by `paragraph_id`. It then prints these details, including the type,
    associated questions, answers (original and counter, depending on type),
    and the paragraph text itself, using ANSI escape codes for bold/underline
    formatting and `wrap_text` for text wrapping.

    Args:
        paragraph_id: The unique identifier string for the paragraph to display.

    Returns:
        None. Prints directly to standard output.
    """
    print("=" * 120)
    try:
        paragraph_info = get_paragraph_info(paragraph_id)

        para_type = paragraph_info.get('Type', 'N/A')
        questions = paragraph_info.get('Questions')  # Could be None or list
        original_answer = paragraph_info.get('Original answer', 'N/A')
        counter_answer = paragraph_info.get('Counter answer', 'N/A')
        text = paragraph_info.get('Text', 'N/A')

        print(f"\033[1m\033[4mParagraph type:\033[0m {para_type}\n")

        if para_type == 'Chain':
            print(f"\033[1m\033[4mQuestions:\033[0m")
            print(f"- {questions[0]}")
            print(f"- {questions[1]}\n")
            print(f"\033[1m\033[4mShared answer:\033[0m {original_answer}\n")
        else:
            print(f"\033[1m\033[4mQuestion:\033[0m")
            print(f"- {questions[0]}\n")
            print(f"\033[1m\033[4mOriginal answer:\033[0m {original_answer}")
            print(f"\033[1m\033[4mCounter answer:\033[0m {counter_answer}\n")

        # Use wrap_text for the main paragraph body
        print(wrap_text(f"\033[1m\033[4mParagraph text:\033[0m {text}"))

    except KeyError as e:
        print(f"\nError: Could not retrieve basic info for paragraph ID '{paragraph_id}': {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred while displaying paragraph '{paragraph_id}': {e}")

    print("=" * 120)


def plot_hist_token_size_per_sample(tokenizer_name: str = "cl100k_base") -> None:
    """Calculates total paragraph tokens per QA sample and plots a histogram.

    Iterates through each QA sample in the global `REASONING_DATASET`. For each
    sample, it identifies all associated paragraph IDs from its reasoning paths.
    It checks if at least one of these paragraph IDs exists in the global
    `PARAGRAPH_DATASET`.

    If one or more paragraphs exist for the sample, it retrieves their text,
    counts the tokens using the specified tiktoken tokenizer, and sums these
    counts. This total token count is added to a list for histogram generation.
    Samples for which *no* corresponding paragraphs are found in
    `PARAGRAPH_DATASET` are skipped.

    Finally, it reports how many samples were processed and displays a histogram
    showing the distribution of total token counts for the processed samples.

    Args:
        tokenizer_name (str, optional): The name of the tiktoken tokenizer to use
            (e.g., "cl100k_base", "p50k_base", "gpt2").
            Defaults to "cl100k_base".

    Returns:
        None. Displays a matplotlib histogram.

    Raises:
        KeyError: If expected keys ('Reasoning path 1', 'Reasoning path 2',
            'Paragraph') are missing in `REASONING_DATASET` items.
        Exception: Can potentially raise exceptions from `tiktoken.get_encoding`
            if an invalid tokenizer name is provided.
    """
    if not REASONING_DATASET:
        print("⚠️ REASONING_DATASET is empty. Cannot generate histogram.")
        return
    if not PARAGRAPH_DATASET:
        print("⚠️ PARAGRAPH_DATASET is empty. All samples will be skipped.")
        return

    try:
        encoder = tiktoken.get_encoding(tokenizer_name)
        print(f"Using tokenizer: {tokenizer_name}")
    except Exception as e:
        print(f"❌ Error initializing tokenizer '{tokenizer_name}': {e}")
        print("Please use a valid tiktoken encoding name (e.g., 'cl100k_base').")
        return

    sample_total_token_counts: List[int] = []
    processed_samples_count = 0
    total_initial_samples = len(REASONING_DATASET)
    missing_paragraph_ids_log = set() # To limit warnings about missing IDs

    print("Calculating total paragraph tokens per QA sample...")
    for sample in tqdm(REASONING_DATASET, desc="Processing Samples"):
        paragraph_ids_in_sample = set()
        has_at_least_one_paragraph = False
        sample_tokens = 0

        # 1. Collect all paragraph IDs for this sample
        for path_key in ['Reasoning path 1', 'Reasoning path 2']:
            reasoning_path = sample.get(path_key, [])
            for step in reasoning_path:
                p_id = step.get('Paragraph')
                if p_id:
                    paragraph_ids_in_sample.add(p_id)

        # If no paragraph IDs were even mentioned in this sample, skip it.
        if not paragraph_ids_in_sample:
            continue

        # 2. Check existence and sum tokens for found paragraphs
        found_paragraph_texts: List[str] = []
        for p_id in paragraph_ids_in_sample:
            paragraph_entry = PARAGRAPH_DATASET.get(p_id)
            if paragraph_entry and isinstance(paragraph_entry, dict):
                paragraph_text = paragraph_entry.get('Text')
                if paragraph_text and isinstance(paragraph_text, str):
                    # Found a valid paragraph text for this sample
                    has_at_least_one_paragraph = True
                    found_paragraph_texts.append(paragraph_text)
                # Optionally warn if text is missing/empty for an existing ID
                # else:
                #     print(f"⚠️ Missing or empty text for paragraph ID '{p_id}'.")
            else:
                 # Log missing IDs only once per run, limit warnings
                if p_id not in missing_paragraph_ids_log:
                    missing_paragraph_ids_log.add(p_id)
                    if len(missing_paragraph_ids_log) <= 10:
                         print(f"⚠️ Paragraph ID '{p_id}' found in reasoning path but not in PARAGRAPH_DATASET.")
                    elif len(missing_paragraph_ids_log) == 11:
                        print("⚠️ (Further missing paragraph ID warnings suppressed)...")

        # 3. Only proceed if at least one paragraph was found for this sample
        if not has_at_least_one_paragraph:
            continue # Skip this sample entirely

        # 4. Calculate total tokens for the found paragraphs
        processed_samples_count += 1
        for text in found_paragraph_texts:
             try:
                 tokens = encoder.encode(text)
                 sample_tokens += len(tokens)
             except Exception as e:
                 # This error is less likely here if text is confirmed string, but good practice
                 print(f"⚠️ Error tokenizing text chunk: {e}")

        sample_total_token_counts.append(sample_tokens)

    # Reporting and Plotting:
    print("-" * 30)
    print(f"Analysis complete.")
    print(f"Processed {processed_samples_count} out of {total_initial_samples} samples for histogram.")
    if missing_paragraph_ids_log:
         print(f"Encountered {len(missing_paragraph_ids_log)} unique paragraph IDs referenced in samples but missing from PARAGRAPH_DATASET.")
    print("-" * 30)


    if not sample_total_token_counts:
        print("❌ No token counts generated (all samples might have been skipped). Cannot plot histogram.")
        return

    # Plotting:
    print("Generating histogram...")
    plt.figure(figsize=(10, 6))
    max_tokens = max(sample_total_token_counts) if sample_total_token_counts else 1
    plt.hist(sample_total_token_counts, bins=min(30, max(10, max_tokens//100)), color='skyblue', edgecolor='black') # Slightly dynamic bins
    plt.title(f'Distribution of Total Paragraph Tokens per QA Sample ({processed_samples_count} Samples)')
    plt.xlabel(f'Total Tokens per Sample')
    plt.ylabel('Number of QA Samples')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()
