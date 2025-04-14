"""..."""
import os
import time
import json
import base64
import random

import pandas as pd
import ipywidgets as widgets
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from IPython.display import display, HTML

from knowledge_maze.create_dataset.create_data_qa import display_qa_sample

load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')

REASONING_PATH_QUESTIONS = [
    "Counter answers are plausible?",
    "No Similar sub-questions?",
    "Sub-questions are relevant?"
]


# Load the reasoning dataset QA:
QA_FILE_PATH = f"{REPO_PATH}\data_files\\knowledge_maze_qa.json"
with open(QA_FILE_PATH, "r", encoding="utf-8") as file:
    REASONING_DATASET = json.load(file)


def create_download_button(df, download_button_area):
    """Creates a client-side download link for the DataFrame."""
    if df is None or df.empty:
         with download_button_area:
             download_button_area.clear_output(wait=True)
             # Optionally display a message that no data is ready yet
             # print("...")
         return
    try:
        df_copy = df.copy()
        # Ensure 'Sample ID' (or the index name) is treated as string for safety
        if df_copy.index.name:
            df_copy.index = df_copy.index.astype(str)
        else: # If it became a regular column after reset_index
             if 'Sample ID' in df_copy.columns:
                 df_copy['Sample ID'] = df_copy['Sample ID'].astype(str)

        csv_data = df_copy.to_csv(index=df_copy.index.name is not None) # Include index only if it has a name
        b64 = base64.b64encode(csv_data.encode('utf-8')).decode('utf-8')
        filename = "tagging_results_live.csv" # Changed filename slightly
        button_html = f'''
        <a download="{filename}" href="data:text/csv;charset=utf-8;base64,{b64}" target="_blank" style="text-decoration: none;">
            <button style="padding: 6px 12px; font-size: 14px; color: #fff; background-color: #007bff; border-color: #007bff; border: 1px solid transparent; border-radius: .25rem; cursor: pointer;">
                Download Current CSV ({len(df)} rows)
            </button>
        </a>
        '''
        with download_button_area:
            download_button_area.clear_output(wait=True)
            display(HTML(button_html))
    except Exception as e:
         with download_button_area:
             download_button_area.clear_output(wait=True)
             print(f"❌ Error creating download link: {e}")


def interactive_tagging(size: int = 100):
    """
    Creates an interactive tagging system, updating the DataFrame incrementally.

    Displays results in wide format after completion. The download button
    is updated after each item is fully tagged.

    Args:
        size (int): size of the tagged subset.

    Returns:
        dict: A dictionary {'results_list': list, 'final_df': pd.DataFrame | None}.
              'final_df' contains the wide DataFrame, updated incrementally.
              Check the UI for 'Tagging Complete!' before assuming it's final.
    """

    # --- State Variables ---
    start_time = time.time()
    # Ensure sample size is not larger than dataset size
    actual_size = min(size, len(REASONING_DATASET))
    if actual_size == 0:
         print("⚠️ REASONING_DATASET is empty or size is 0. Cannot start tagging.")
         return {'results_list': [], 'final_df': None}
    if size > len(REASONING_DATASET):
        print(f"⚠️ Requested size {size} is larger than dataset size {len(REASONING_DATASET)}. Using {actual_size} samples.")

    dataset = random.sample(REASONING_DATASET, k=actual_size)
    results_list = [] # Raw results stored here
    current_index = 0
    # Using a dictionary allows modification by the nested function.
    # Initialize final_df as None
    final_result_holder = {'results_list': results_list, 'final_df': None}

    # --- Output areas ---
    text_display_area = widgets.Output()
    question_display_areas = [widgets.Output() for _ in range(len(REASONING_PATH_QUESTIONS))]
    final_result_area = widgets.Output()
    download_button_area = widgets.Output() # For the dynamic button

    # --- Buttons ---
    buttons = []
    for i in range(len(REASONING_PATH_QUESTIONS)):
        yes_button = widgets.Button(description="Yes", button_style="success", layout=widgets.Layout(width='auto'))
        no_button = widgets.Button(description="No", button_style="danger", layout=widgets.Layout(width='auto'))
        buttons.append((yes_button, no_button))

    # --- Helper: Process Results & Update DF/Button ---
    def process_and_update():
        """Internal function to process results, update final_df, and refresh button."""
        if not results_list:
            return # Nothing to process

        try:
            # Create, Pivot, and Clean DataFrame from *all* results so far
            df_long = pd.DataFrame(results_list, columns=["Sample ID", "Question", "Answer"])
            # Ensure Sample ID is suitable for indexing/joining (string is safest)
            df_long['Sample ID'] = df_long['Sample ID'].astype(str)

            # Pivot
            df_wide = df_long.pivot_table(index='Sample ID', columns='Question', values='Answer', aggfunc='first')
            # Reset index to make 'Sample ID' a column
            df_wide = df_wide.reset_index()
            # Ensure correct column order (Sample ID first)
            cols = ['Sample ID'] + [col for col in df_wide.columns if col != 'Sample ID']
            df_wide = df_wide[cols]
            df_wide.columns.name = None # Clean column index name

            # Store the *current* wide DataFrame
            final_result_holder['final_df'] = df_wide

            # Update download button with the current DataFrame
            create_download_button(df_wide, download_button_area)

        except Exception as e:
            # Handle potential errors during pivot (e.g., duplicate entries if logic flawed)
             with final_result_area: # Display error prominently
                final_result_area.clear_output(wait=True)
                print(f"❌ Error processing results during pivot: {e}")
                print("Current raw results:")
                print(results_list[-len(REASONING_PATH_QUESTIONS):]) # Show recent results

    # --- Core Logic Functions ---

    def show_final_display():
        """Internal function to display the final message and results."""
        # Clear interactive elements
        text_display_area.clear_output()
        for output_area in question_display_areas:
            output_area.clear_output()
        final_result_area.clear_output()
        # Keep the download button area as it holds the latest button

        df_final = final_result_holder['final_df']

        with final_result_area:
            if df_final is not None and not df_final.empty:
                print("✅ Tagging Complete! Here are your final answers:\n")
                with pd.option_context('display.max_rows', None, 'display.max_colwidth', 100, 'display.width', 1000):
                    display(df_final)
                # Optional: Create one last definitive download button based on the final DF
                create_download_button(df_final, download_button_area)
            elif not results_list:
                 print("⚠️ No results were recorded during the session.")
            else:
                 print("❌ Tagging completed, but there was an error generating the final table.")
                 print("Raw results are available in the returned dictionary['results_list'].")


    def on_button_click(q_index, answer):
        nonlocal current_index # Required to modify current_index
        # Prevent clicks after completion or if index is out of bounds
        if current_index >= len(dataset):
             return

        # Record the answer
        current_data_item_id = dataset[current_index]['ID']
        results_list.append((current_data_item_id, REASONING_PATH_QUESTIONS[q_index], answer))

        # Update UI for the answered question
        buttons[q_index][0].disabled = True
        buttons[q_index][1].disabled = True
        with question_display_areas[q_index]:
            question_display_areas[q_index].clear_output(wait=True)
            print(f"{REASONING_PATH_QUESTIONS[q_index]} → {answer}")

        # Check if all questions for the *current* item are answered
        # Need to count results ONLY for the current_data_item_id
        answered_count = sum(1 for res in results_list if res[0] == current_data_item_id)

        if answered_count == len(REASONING_PATH_QUESTIONS):
            # Process results and update DF/Button
            process_and_update()

            # Move to the next item
            current_index += 1
            if current_index < len(dataset):
                show_next_text()
            else:
                # All items tagged, show final display
                show_final_display()


    def show_next_text():
        # Clear previous item's display and enable buttons
        text_display_area.clear_output(wait=True)
        for i, output_area in enumerate(question_display_areas):
            output_area.clear_output(wait=True) # Clear previous question/buttons
            buttons[i][0].disabled = False
            buttons[i][1].disabled = False

        # Display current item info
        with text_display_area:
            time_elapsed = time.time() - start_time
            time_minutes = f"{time_elapsed // 60:.0f}"
            time_seconds = f"{time_elapsed % 60:.0f}" if time_elapsed % 60 > 9 else f"0{time_elapsed % 60:.0f}"
            print(f"\n--- Item {current_index + 1} of {len(dataset)} --- ({time_minutes}:{time_seconds})")
            try:
                display_qa_sample(dataset[current_index])
            except Exception as e:
                print(f"❌ Error displaying sample {dataset[current_index].get('ID', 'N/A')}: {e}")


        # Display questions and buttons for the current item
        for i, output_area in enumerate(question_display_areas):
            with output_area:
                print(REASONING_PATH_QUESTIONS[i])
                display(widgets.HBox([buttons[i][0], buttons[i][1]]))


    # --- Setup Button Events ---
    for q_idx, (yes_btn, no_btn) in enumerate(buttons):
        # Use lambda directly or keep nested function for clarity
        yes_btn.on_click(lambda b, index=q_idx, ans="Yes": on_button_click(index, ans))
        no_btn.on_click(lambda b, index=q_idx, ans="No": on_button_click(index, ans))

    # --- Initial Display ---
    # Display layout containers first
    display(text_display_area)
    for area in question_display_areas:
        display(area)
    display(final_result_area)
    display(download_button_area) # Display area for the button

    # Start the process by showing the first item
    if len(dataset) > 0:
        show_next_text()

    # --- Return the holder ---
    # The function returns *immediately* after setup.
    # 'final_df' in the returned dict will be updated dynamically by the UI interactions.
    return final_result_holder


def plot_tagging_summary(df_wide: pd.DataFrame, text_col='Sample ID'):
    """
    Generates and displays a bar plot summarizing tagging results from a wide DataFrame.

    Args:
        df_wide (pd.DataFrame): The input DataFrame where rows are samples,
                                'text_col' identifies the sample, and other
                                columns represent questions with 'Yes'/'No' answers.
        text_col (str, optional): The name of the column identifying the text/sample.

    Returns:
        dict: A dictionary containing the calculated percentages for each category plotted.
              Returns None if the input DataFrame is invalid or empty.
    """
    # --- Input Validation ---
    if not isinstance(df_wide, pd.DataFrame) or df_wide.empty:
        print("❌ Error: Input must be a non-empty pandas DataFrame.")
        return None
    if text_col not in df_wide.columns:
        print(f"❌ Error: Text column '{text_col}' not found in DataFrame.")
        return None

    # --- Identify Question Columns ---
    question_cols = [col for col in df_wide.columns if col != text_col]
    if not question_cols:
        print("❌ Error: No question columns found (columns other than '{text_col}').")
        return None

    total_samples = df_wide.shape[0]
    if total_samples == 0:
         print("❌ Error: DataFrame has zero rows.")
         return None

    # --- Calculate Percentages ---
    percentages = {}
    plot_labels = []
    plot_values = []

    # 1-N: Percentage 'Yes' for each question
    for col in question_cols:
        if col not in df_wide:
            print(f"⚠️ Warning: Expected question column '{col}' not found, skipping.")
            continue
        # Ensure the column contains comparable values (handle potential NaN or other types)
        try:
            # Count 'Yes', treating potential non-strings/NaNs as not 'Yes'
            yes_count = (df_wide[col].astype(str).str.strip().str.lower() == 'yes').sum()
            percentage = (yes_count / total_samples) * 100
            percentages[f"{col} == Yes"] = percentage
            plot_labels.append(col) # Use original column name for plot label
            plot_values.append(percentage)
        except Exception as e:
            print(f"⚠️ Warning: Could not process column '{col}'. Error: {e}. Skipping.")


    # N+1: Percentage of samples where ALL answers are 'Yes'
    try:
        # Create a boolean DataFrame checking for 'Yes' (case-insensitive, stripped)
        is_yes_df = df_wide[question_cols].apply(lambda x: x.astype(str).str.strip().str.lower() == 'yes')
        # Check if all values in a row are True
        all_yes_mask = is_yes_df.all(axis=1)
        all_yes_count = all_yes_mask.sum()
        all_yes_percentage = (all_yes_count / total_samples) * 100

        category_name = "All Questions == Yes"
        percentages[category_name] = all_yes_percentage
        plot_labels.append("Good QA Samples") # Shorter label for plot
        plot_values.append(all_yes_percentage)
    except Exception as e:
         print(f"⚠️ Warning: Could not calculate 'All Questions == Yes'. Error: {e}")


    # --- Create Bar Plot ---
    if not plot_labels:
        print("\n❌ Error: No data available to plot.")
        return percentages # Return potentially partial percentages


    fig, ax = plt.subplots(figsize=(max(6, int(len(plot_labels) * 1.5)), 5)) # Adjust width based on # bars

    bars = ax.bar(plot_labels, plot_values, color='skyblue')

    # Add percentage text on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', # Add 1 for slight offset
                va='bottom', ha='center', fontsize=9) # vertical alignment: bottom

    # Customize plot
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Tagging Session Summary - QA samples metadata')
    ax.set_ylim(0, 105) # Extend y-limit slightly for text visibility
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=30, ha='right') # Rotate labels if they might overlap

    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()


def get_good_and_bad_sample_ids(df_wide: pd.DataFrame, id_col: str = 'Sample ID') -> tuple[list[str], list[str]]:
    """
    Categorizes sample IDs into 'good' (all 'Yes') and 'bad' (at least one 'No').

    Args:
        df_wide (pd.DataFrame): The wide-format DataFrame resulting from the
                                tagging process. Expected to have an ID column
                                and question columns with 'Yes'/'No' (or similar)
                                string values.
        id_col (str, optional): The name of the column containing the unique
                                sample identifiers. Defaults to 'Sample ID'.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
            - good_ids: A list of unique sample IDs where all answers were 'Yes'.
            - bad_ids: A list of unique sample IDs where at least one answer was 'No'.
        Returns ([], []) if the input is invalid or empty.
    """
    # --- Input Validation ---
    if not isinstance(df_wide, pd.DataFrame):
        print("⚠️ Warning: Input is not a pandas DataFrame.")
        return [], []
    if df_wide.empty:
        # Valid case: empty input means empty output lists
        return [], []
    if id_col not in df_wide.columns:
        print(f"❌ Error: ID column '{id_col}' not found in DataFrame.")
        return [], []

    # --- Identify Question Columns ---
    question_cols = [col for col in df_wide.columns if col != id_col]

    if not question_cols:
        print("⚠️ Warning: No question columns found (besides ID column). Cannot determine good/bad.")
        # Return all IDs as neither good nor bad can be determined, or maybe return all as 'bad'?
        return [], []

    try:
        # --- Find Rows with 'No' (Bad Samples) ---
        is_no_df = df_wide[question_cols].map(lambda x: str(x).strip().lower() == 'no')
        # Check if *any* column in a row contains True
        has_at_least_one_no = is_no_df.any(axis=1)

        # --- Extract Bad IDs ---
        bad_samples_df = df_wide.loc[has_at_least_one_no]
        # Ensure IDs are strings for consistent comparison later
        bad_ids = bad_samples_df[id_col].astype(str).tolist()
        bad_ids_set = set(bad_ids) # Use a set for efficient lookup

        # --- Determine Good IDs ---
        all_ids = df_wide[id_col].astype(str).unique().tolist()
        all_ids_set = set(all_ids)

        # Good IDs are those in all_ids but not in bad_ids
        good_ids_set = all_ids_set - bad_ids_set
        good_ids = sorted(list(good_ids_set)) # Sort for consistent output order

        # Sort bad_ids as well for consistency if desired
        bad_ids = sorted(list(bad_ids_set))

        return good_ids, bad_ids

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return [], []


def display_annotated_samples(df_wide: pd.DataFrame, num_samples: int = 3) -> None:
    """Displays random samples categorized as 'good' and 'bad' based on tagging.

    This function first uses `get_good_and_bad_sample_ids` to classify samples
    from the input DataFrame (`df_wide`) based on their 'Yes'/'No' tags.
    It then randomly selects a specified number (`num_samples`) of IDs from both
    the 'good' (all 'Yes') and 'bad' (at least one 'No') categories.

    Finally, it retrieves the full sample data corresponding to these selected IDs
    from the global `REASONING_DATASET` and prints them using the
    `display_qa_sample` function, visually separated into 'Good Samples' and
    'Bad Samples' sections.

    Args:
        df_wide (pd.DataFrame): The DataFrame containing the tagging results.
            It's expected to have an ID column (matching IDs in
            `REASONING_DATASET`) and question columns containing 'Yes'/'No'
            string answers used by `get_good_and_bad_sample_ids`.
        num_samples (int, optional): The number of random samples to display
            from each category (good and bad). Defaults to 3.

    Returns:
        None: This function prints directly to the output and does not return
              any value.

    Raises:
        ValueError: If `num_samples` is greater than the number of available
            sample IDs in either the 'good' or 'bad' category returned by
            `get_good_and_bad_sample_ids`.
        KeyError: If items retrieved from `REASONING_DATASET` do not contain
            the expected 'ID' key (or if `get()` fails gracefully but results
            in filtering issues).
        Any errors raised by `get_good_and_bad_sample_ids` or
            `display_qa_sample`.
    """
    good_ids, bad_ids = get_good_and_bad_sample_ids(df_wide)

    print("=" * 120)
    print("=" * 52, " Good Samples ", "=" * 52)
    # Ensure we don't request more samples than available
    num_good_to_show = min(num_samples, len(good_ids))
    if num_good_to_show > 0:
        random_good_ids = random.sample(good_ids, k=num_good_to_show)
        # Use a set for faster lookup when filtering REASONING_DATASET
        good_ids_set = set(random_good_ids)
        for sample in [item for item in REASONING_DATASET if item.get("ID") in good_ids_set]:
            # Ensure the sample ID is actually one we requested (handles potential ID mismatches/case issues if any)
            if str(sample.get("ID")) in good_ids_set: # Extra check for safety
                 display_qa_sample(sample)
    else:
        print("No 'good' samples found or requested.")

    print("=" * 52, " Bad Samples  ", "=" * 52)
     # Ensure we don't request more samples than available
    num_bad_to_show = min(num_samples, len(bad_ids))
    if num_bad_to_show > 0:
        random_bad_ids = random.sample(bad_ids, k=num_bad_to_show)
        # Use a set for faster lookup
        bad_ids_set = set(random_bad_ids)
        for sample in [item for item in REASONING_DATASET if item.get("ID") in bad_ids_set]:
             if str(sample.get("ID")) in bad_ids_set: # Extra check for safety
                 display_qa_sample(sample)
    else:
         print("No 'bad' samples found or requested.")

    print("=" * 120)
