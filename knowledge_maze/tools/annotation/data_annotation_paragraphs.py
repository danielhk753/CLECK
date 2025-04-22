"""Provides tools for interactive tagging and analysis of QA reasoning paths.

This module facilitates the manual annotation (tagging) of generated Question-Answer
samples, specifically focusing on the quality and validity of reasoning paths
and their constituent paragraphs. It leverages `ipywidgets` for creating an
interactive user interface within a Jupyter environment.

Core Functionality (`interactive_paragraph_tagging`):
- Pre-filters the main QA dataset to include only samples for which all
  associated paragraph data exists.
- Presents these valid QA samples one by one (`display_qa_sample`).
- Iterates through the paragraphs associated with each reasoning path in the sample.
- For each paragraph:
    - Displays relevant context (preceding/next questions).
    - Displays the paragraph text (`display_paragraph`).
    - Asks the user to classify the paragraph's quality using buttons:
        - Yes (Good)
        - No (Reason: Contains real answer)
        - No (Reason: Doesn't lead to next question)
        - No (Reason: Sub-questions irrelevant)
- After evaluating all paragraphs in a path, asks if the path itself is valid.
- After evaluating both paths, asks if the overall sample is valid.
- Records all user annotations cumulatively within a sample's display.
- Provides a live-updating download button for the results (`create_download_button`).

Analysis & Visualization (Separate utilities, may need data from tagging):
- `plot_tagging_summary`: Generates a bar chart summarizing tagging results.
- `get_good_and_bad_sample_ids`: Classifies sample IDs based on 'Yes'/'No' tags.
- `display_annotated_samples`: Shows random examples of 'good' and 'bad' tagged
  samples.

Dependencies:
- Loads QA (`REASONING_DATASET`) and Paragraph (`PARAGRAPH_DATASET`) data from
  files specified via environment variables (`dotenv`, `REPO_PATH`).
- Uses `ipywidgets`, `IPython.display` for the interactive UI.
- Uses `pandas` for data handling and `matplotlib` for plotting.
- Relies on helper display functions (`display_qa_sample`, `display_paragraph`,
 `get_paragraph_info`) from other modules in the `knowledge_maze` package.
"""

import os
import time
import json
import base64
import random
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import cm
from tqdm import tqdm
import ipywidgets as widgets
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from IPython.display import display, HTML, clear_output

from knowledge_maze.create_dataset.create_data_qa import display_qa_sample
from knowledge_maze.create_dataset.create_paragraphs import display_paragraph, get_paragraph_info


# --- Module Initialization ---
load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')
if not REPO_PATH:
    print("Warning: REPO_PATH environment variable not set. File loading might fail.")
    REPO_PATH = "." # Example fallback

# Load QA data
try:
    # Use os.path.join for better path compatibility
    QA_FILE_PATH = os.path.join(REPO_PATH, "data_files", "knowledge_maze_qa.json")
    with open(QA_FILE_PATH, "r", encoding="utf-8") as file:
        REASONING_DATASET: List[Dict[str, Any]] = json.load(file)
    print(f"Loaded {len(REASONING_DATASET)} QA samples from {QA_FILE_PATH}")
except FileNotFoundError:
    print(f"❌ Error: QA data file not found at {QA_FILE_PATH}")
    REASONING_DATASET = []
except Exception as e:
    print(f"❌ Error loading QA data from {QA_FILE_PATH}: {e}")
    REASONING_DATASET = []

# Load Paragraph data
try:
    PARAGRAPHS_FILE_PATH = os.path.join(REPO_PATH, "data_files", "knowledge_maze_paragraphs.json")
    with open(PARAGRAPHS_FILE_PATH, "r", encoding="utf-8") as file:
        PARAGRAPH_DATASET: Dict[str, Dict[str, str]] = json.load(file)
    print(f"Loaded {len(PARAGRAPH_DATASET)} paragraphs from {PARAGRAPHS_FILE_PATH}")
except FileNotFoundError:
    print(f"❌ Error: Paragraph data file not found at {PARAGRAPHS_FILE_PATH}")
    PARAGRAPH_DATASET = {}
except Exception as e:
    print(f"❌ Error loading Paragraph data from {PARAGRAPHS_FILE_PATH}: {e}")
    PARAGRAPH_DATASET = {}

# --- Constants for Tagging ---
PARAGRAPH_ANSWERS = {
    0: "Yes",
    1: "No - Real Answer Present",
    2: "No - Doesn't Lead to Next",
    3: "No - Irrelevant Sub-Questions",
    4: "No - Other Reason"
}
PARAGRAPH_BUTTON_STYLES = {
    0: "success", 1: "danger", 2: "danger", 3: "danger", 4: "danger"
}
YES_NO_ANSWERS = { 0: "Yes", 1: "No" }
YES_NO_BUTTON_STYLES = { 0: "success", 1: "danger" }
# --- End Constants ---


# --- Utility Function Definitions ---

def create_download_button(df, download_button_area):
    """Creates a client-side download link for the DataFrame."""
    if df is None or df.empty:
         with download_button_area:
             download_button_area.clear_output(wait=True)
         return
    try:
        df_copy = df.copy()
        # Ensure ID column is string if it exists
        id_col = 'Sample ID' # Assuming this is the ID column name after potential pivot/reset
        if id_col in df_copy.columns:
            df_copy[id_col] = df_copy[id_col].astype(str)
        elif df_copy.index.name == id_col: # Check if it's the index name
             df_copy.index = df_copy.index.astype(str)


        # Determine if index should be included based on its name
        include_index = df_copy.index.name is not None and df_copy.index.name != id_col

        csv_data = df_copy.to_csv(index=include_index)
        b64 = base64.b64encode(csv_data.encode('utf-8')).decode('utf-8')
        filename = "paragraph_tagging_results_live.csv"
        button_html = f'''
        <a download="{filename}" href="data:text/csv;charset=utf-8;base64,{b64}" target="_blank" style="text-decoration: none;">
            <button style="padding: 6px 12px; font-size: 14px; color: #fff; background-color: #007bff; border-color: #007bff; border: 1px solid transparent; border-radius: .25rem; cursor: pointer;">
                Download Current CSV ({len(df)} rows tagged)
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


def plot_tagging_summary_para(df_long: pd.DataFrame) -> None:
    """
    Generates summary plots with custom color schemes.

    Takes the long-format DataFrame from interactive_paragraph_tagging.
    Creates two plots:
    1. Bar chart: Shows % Good Paragraphs, % Good Paths, % Good Samples.
       Bars are colored based on their value (percentage) using a Red->Orange->Green gradient.
    2. Pie chart: Shows distribution of Paragraph tagging answers. 'Yes' slice
       is green, 'No - ...' slices are shades of red/orange.

    Args:
        df_long (pd.DataFrame): Long-format DataFrame with columns
            'ID_Type', 'ID', 'Response_Value', 'Response_Text'.
    """
    # --- Input Validation ---
    required_cols = ['ID_Type', 'ID', 'Response_Value', 'Response_Text']
    if not isinstance(df_long, pd.DataFrame) or df_long.empty:
        print("❌ Error: Input must be a non-empty pandas DataFrame.")
        return
    if not all(col in df_long.columns for col in required_cols):
        print(f"❌ Error: DataFrame missing one or more required columns: {required_cols}")
        return

    # --- Calculate Metrics ---
    metrics = {}
    bar_plot_labels = []
    bar_plot_values = []
    # bar_colors will be determined later based on values

    # 1. % Good Paragraphs
    para_df = df_long[df_long['ID_Type'] == 'Paragraph'].copy()
    total_paragraphs = len(para_df)
    percent_good_para = 0.0 # Default
    if total_paragraphs > 0:
        good_paragraphs = len(para_df[para_df['Response_Value'] == 0]) # Assuming 0 is 'Yes'
        percent_good_para = (good_paragraphs / total_paragraphs) * 100
        metrics['Good Paragraphs'] = percent_good_para
        bar_plot_labels.append('% Good\nParagraphs')
        bar_plot_values.append(percent_good_para)
    else:
        print("⚠️ No 'Paragraph' type entries found.")

    # 2. % Good Paths (marked 'Yes')
    path_df = df_long[df_long['ID_Type'] == 'Path'].copy()
    total_paths = len(path_df)
    percent_good_path = 0.0 # Default
    if total_paths > 0:
        good_paths = len(path_df[path_df['Response_Value'] == 0]) # Assuming 0 is 'Yes'
        percent_good_path = (good_paths / total_paths) * 100
        metrics['Good Paths'] = percent_good_path
        bar_plot_labels.append('% Good\nPaths')
        bar_plot_values.append(percent_good_path)
    else:
         print("⚠️ No 'Path' type entries found.")

    # 3. % Good Samples (marked 'Yes')
    sample_df = df_long[df_long['ID_Type'] == 'Sample'].copy()
    total_samples = len(sample_df)
    percent_good_sample = 0.0 # Default
    if total_samples > 0:
        good_samples = len(sample_df[sample_df['Response_Value'] == 0]) # Assuming 0 is 'Yes'
        percent_good_sample = (good_samples / total_samples) * 100
        metrics['Good Samples'] = percent_good_sample
        bar_plot_labels.append('% Good\nSamples')
        bar_plot_values.append(percent_good_sample)
    else:
         print("⚠️ No 'Sample' type entries found.")

    # --- Create Plots ---
    if not bar_plot_labels:
        print("\n❌ Error: No data available to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Bar Chart with Value-Based Colors ---

    # Normalize values (percentages) to 0-1 for colormap
    norm = mcolors.Normalize(vmin=0, vmax=100)
    # Choose a colormap: Red -> Yellow -> Green ('RdYlGn') is common
    # Other options: 'coolwarm' (Blue-Red), 'viridis', 'plasma' etc.
    cmap = cm.get_cmap('RdYlGn')
    # Generate colors based on normalized values
    bar_colors = [cmap(norm(value)) for value in bar_plot_values]

    bars = ax1.bar(bar_plot_labels, bar_plot_values, color=bar_colors) # Use calculated colors
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Tagging Summary Metrics')
    ax1.set_ylim(0, 105)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # Add percentage text on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%',
                 va='bottom', ha='center', fontsize=9)

    # --- Plot 2: Pie Chart with Custom Category Colors ---
    if total_paragraphs > 0:
        para_answer_counts = para_df['Response_Text'].value_counts()
        pie_labels_ordered = para_answer_counts.index.tolist() # Keep order from value_counts
        pie_sizes = para_answer_counts.values.tolist()

        # *** Define Custom Colors for Pie Chart ***
        # Define your desired colors explicitly
        color_map_pie = {
            "Yes": "lightgreen",
            "No - Real Answer Present": "red", # Example Red/Orange shades
            "No - Doesn't Lead to Next": "indianred",
            "No - Irrelevant Sub-Questions": "darksalmon",
            "No - Other Reason": "crimson",
            # Add more mappings if PARAGRAPH_ANSWERS changes
        }
        # Get default color for any unexpected labels
        default_pie_color = 'grey'
        # Create the list of colors in the correct order based on pie_labels_ordered
        pie_colors = [color_map_pie.get(label, default_pie_color) for label in pie_labels_ordered]
        # *** End Custom Pie Colors ***

        wedges, texts, autotexts = ax2.pie(
            pie_sizes,
            labels=None,
            autopct='%1.1f%%',
            startangle=90,
            colors=pie_colors, # Apply custom colors
            pctdistance=0.85,
             wedgeprops={'edgecolor': 'white'}
        )
        plt.setp(autotexts, size=8, weight="bold", color="black")

        ax2.set_title('Distribution of Paragraph Tagging Answers')
        # Create legend labels with counts
        legend_pie_labels = [f"{l} ({s})" for l, s in zip(pie_labels_ordered, pie_sizes)]
        ax2.legend(wedges, legend_pie_labels,
                   title="Paragraph Answers (Count)",
                   loc="center left",
                   bbox_to_anchor=(1.05, 0.5),
                   fontsize='small')
        ax2.axis('equal')

    else:
        ax2.set_title('Distribution of Paragraph Tagging Answers')
        ax2.text(0.5, 0.5, "No Paragraph Data Found",
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes, fontsize=12, color='grey')

    plt.tight_layout(rect=(0, 0, 0.85, 0.95)) # Adjust layout
    plt.show()


def get_good_and_bad_sample_ids(df_wide: pd.DataFrame, id_col: str = 'ID') -> tuple[list[str], list[str]]:
    """
    (Placeholder/Example) Categorizes IDs based on 'Yes'/'No' columns.
    NOTE: Assumes df_wide has specific columns from a *different* tagging schema.
          Needs adaptation if used with interactive_paragraph_tagging results.
    """
    print("--- Getting Good/Bad Sample IDs (Placeholder) ---")
    print("NOTE: This function assumes a specific DataFrame structure.")
    if not isinstance(df_wide, pd.DataFrame) or df_wide.empty or id_col not in df_wide.columns:
        print("Invalid DataFrame for get_good_and_bad_sample_ids.")
        return [], []
    # --- Add actual logic here ---
    # Example: Assume a column 'Overall Validation' exists
    # if 'Overall Validation' in df_wide.columns:
    #     good_ids = df_wide[df_wide['Overall Validation'] == 'Yes'][id_col].tolist()
    #     bad_ids = df_wide[df_wide['Overall Validation'] != 'Yes'][id_col].tolist()
    #     return good_ids, bad_ids
    # else:
    #     print("Required column 'Overall Validation' not found.")
    #     return [], []
    return df_wide[id_col].tolist(), [] # Return all as good, empty bad for placeholder


def display_annotated_samples(df_wide: pd.DataFrame, num_samples: int = 3) -> None:
    """
    (Placeholder/Example) Displays random good/bad samples based on prior classification.
    NOTE: Assumes df_wide has specific columns from a *different* tagging schema.
          Needs adaptation if used with interactive_paragraph_tagging results.
    """
    print("--- Displaying Annotated Samples (Placeholder) ---")
    good_ids, bad_ids = get_good_and_bad_sample_ids(df_wide) # Assumes this works

    print("=" * 120)
    print("=" * 52, " Good Samples ", "=" * 52)
    num_good_to_show = min(num_samples, len(good_ids))
    if num_good_to_show > 0:
        # Find samples in REASONING_DATASET matching good_ids
        good_samples = [s for s in REASONING_DATASET if s.get('ID') in good_ids]
        sampled_good = random.sample(good_samples, k=num_good_to_show)
        for sample in sampled_good: display_qa_sample(sample)
    else: print("No 'good' samples found or requested.")

    print("=" * 52, " Bad Samples  ", "=" * 52)
    num_bad_to_show = min(num_samples, len(bad_ids))
    if num_bad_to_show > 0:
        # Find samples in REASONING_DATASET matching bad_ids
        bad_samples = [s for s in REASONING_DATASET if s.get('ID') in bad_ids]
        sampled_bad = random.sample(bad_samples, k=num_bad_to_show)
        for sample in sampled_bad: display_qa_sample(sample)
    else: print("No 'bad' samples found or requested.")
    print("=" * 120)

# --- END Utility Function Definitions ---


# --- Main Interactive Paragraph Tagging Function ---

def interactive_paragraph_tagging(size: int = 50, start_idx: int = 0):
    """
    Creates an interactive UI for tagging paragraphs within QA reasoning paths,
    displaying content cumulatively within a sample. Only samples with existing
    paragraph data are included.

    Args:
        size (int): Max number of QA samples to tag. Defaults to 50.
        start_idx (int): Starting index within the *pre-filtered* dataset. Defaults to 0.

    Returns:
        dict: {'results_list': list, 'current_df': pd.DataFrame | None}.
    """

    # --- *** NEW: Pre-filter REASONING_DATASET *** ---
    print("Pre-filtering QA samples to ensure all paragraphs exist...")
    valid_reasoning_samples = []
    initial_count = len(REASONING_DATASET)
    for sample in tqdm(REASONING_DATASET, desc="Checking Samples"):
        all_paras_exist = True
        # Collect all paragraph IDs needed by this sample
        pids_in_sample = set()
        for path_key in ['Reasoning path 1', 'Reasoning path 2']:
            for step in sample.get(path_key, []):
                p_id = step.get('Paragraph')
                if p_id:
                    pids_in_sample.add(p_id)

        if not pids_in_sample: # Skip samples with no paragraphs referenced
            all_paras_exist = False
        else:
            # Check if all collected IDs exist in PARAGRAPH_DATASET
            for p_id in pids_in_sample:
                if p_id not in PARAGRAPH_DATASET:
                    all_paras_exist = False
                    break # No need to check further for this sample

        if all_paras_exist:
            valid_reasoning_samples.append(sample)

    filtered_count = len(valid_reasoning_samples)
    print(f"Pre-filtering complete. Found {filtered_count} samples (out of {initial_count}) with all necessary paragraphs.")
    # --- *** END NEW: Pre-filter *** ---


    # --- Input Validation & Setup (using filtered data) ---
    if not valid_reasoning_samples or not PARAGRAPH_DATASET:
        print("❌ Error: No valid QA samples with existing paragraphs found, or PARAGRAPH_DATASET not loaded.")
        return {'results_list': [], 'current_df': None}

    # Adjust size calculation based on the filtered list
    effective_start_idx = min(start_idx, filtered_count) # Ensure start_idx isn't out of bounds
    actual_size = min(size, filtered_count - effective_start_idx)

    if actual_size <= 0:
        print(f"⚠️ No samples available to tag from index {effective_start_idx} in the filtered dataset.")
        return {'results_list': [], 'current_df': None}
    if size > filtered_count - effective_start_idx:
         print(f"⚠️ Requested size {size} exceeds available samples ({filtered_count - effective_start_idx}) from index {effective_start_idx} in filtered data. Using {actual_size}.")

    # Sample from the pre-filtered list
    dataset_subset = random.sample(valid_reasoning_samples[effective_start_idx:], k=actual_size)


    # --- State Variables ---
    start_time = time.time()
    results_list = [] # Stores tuples: (ID_Type, ID, Response_Value, Response_Text)
    current_df = None
    tagging_state = {
        'dataset_subset_idx': 0, # Index within the dataset_subset being tagged
        'path_idx': 0,
        'paragraph_step_idx': 0,
        'stage': 'sample_start' # Stages: sample_start, paragraph, path_valid, sample_valid, done
    }
    active_widgets = {'buttons': None, 'output_area': None}
    final_result_holder = {'results_list': results_list, 'current_df': current_df}

    # --- Output Areas ---
    main_display_area = widgets.VBox()
    title_area = widgets.Output()
    progress_area = widgets.Output()
    download_button_area = widgets.Output()
    completion_area = widgets.Output()


    # --- Helper: Save and Update DF ---
    def record_and_update_df(id_type, record_id, response_value, response_text):
        nonlocal current_df
        if id_type:
            # Store both value and text for easier analysis later
            results_list.append((id_type, record_id, response_value, response_text))
        try:
            if results_list:
                # Include Response_Text in DataFrame
                temp_df = pd.DataFrame(results_list, columns=['ID_Type', 'ID', 'Response_Value', 'Response_Text'])
                current_df = temp_df
                final_result_holder['current_df'] = current_df
                create_download_button(current_df, download_button_area)
        except Exception as e:
            print(f"❌ Error updating DataFrame: {e}")


    # --- Core Logic: Display Stages ---
    def display_current_stage():
        nonlocal tagging_state, active_widgets

        subset_idx = tagging_state['dataset_subset_idx'] # Use index within the subset

        # --- Handle Completion ---
        if tagging_state['stage'] == 'done':
            main_display_area.children = []
            title_area.clear_output()
            with completion_area:
                clear_output()
                print("✅ Tagging Complete!")
                df_final = final_result_holder['current_df']
                if df_final is not None and not df_final.empty:
                    print("\nFinal Results Summary (Tail):")
                    with pd.option_context('display.max_rows', 10):
                        display(df_final.tail()) # Show tail
                else:
                    print("\nNo results were recorded.")
            record_and_update_df(None, None, None, None) # Final button update
            return

        # --- Handle Start of New Sample ---
        if tagging_state['stage'] == 'sample_start':
            main_display_area.children = []
            title_area.clear_output()
            completion_area.clear_output()

            current_sample = dataset_subset[subset_idx]
             # Calculate original index if needed for display/reference
            original_dataset_index = "N/A" # Placeholder
            try:
                # This assumes IDs are unique and present in original REASONING_DATASET
                original_idx_list = [i for i, item in enumerate(REASONING_DATASET) if item.get('ID') == current_sample.get('ID')]
                if original_idx_list:
                     original_dataset_index = original_idx_list[0]
            except Exception:
                pass # Ignore errors finding original index


            with title_area:
                 # Display index within the current tagging session AND original index
                print(f"--- Tagging Sample {subset_idx + 1} / {len(dataset_subset)} (Original Index: {original_dataset_index}) ---")
            new_sample_output = widgets.Output()
            with new_sample_output:
                display_qa_sample(current_sample)
            main_display_area.children += (new_sample_output,)
            tagging_state['stage'] = 'paragraph'

        # --- Update Progress ---
        progress_area.clear_output()
        with progress_area:
             time_elapsed = time.time() - start_time
             time_minutes = f"{time_elapsed // 60:.0f}"
             time_seconds = f"{time_elapsed % 60:.0f}" if time_elapsed % 60 > 9 else f"0{time_elapsed % 60:.0f}"
             print(f"Time Elapsed: {time_minutes}:{time_seconds}")


        # --- Display Content Based on Stage ---
        current_sample = dataset_subset[subset_idx]
        stage = tagging_state['stage']
        path_idx = tagging_state['path_idx']
        step_idx = tagging_state['paragraph_step_idx']
        path_key = 'Reasoning path 1' if path_idx == 0 else 'Reasoning path 2'
        current_path = current_sample.get(path_key, [])

        current_step_output = widgets.Output()
        active_widgets['output_area'] = current_step_output

        if stage == 'paragraph':
            if step_idx < len(current_path):
                step = current_path[step_idx]
                paragraph_id = step.get('Paragraph')
                prev_q = step.get('Previous question', 'N/A')
                next_q = step.get('Next question', 'N/A (End of path?)')

                with current_step_output:
                    print("=" * 120)
                    print("-" * 33 + f" Evaluating Paragraph in {path_key}, Step {step_idx + 1} / {len(current_path)} " + "-" * 33)

                    # Paragraph ID should exist due to pre-filtering
                    if paragraph_id and paragraph_id in PARAGRAPH_DATASET:
                         display_paragraph(paragraph_id)
                    elif paragraph_id: # Should not happen if pre-filtering worked
                         print(f"\n❌ Internal Error: Paragraph ID '{paragraph_id}' not found despite pre-filtering.")
                    else:
                         print("\n⚠️ Paragraph ID missing for this step.")

                    print("\n\033[1mParagraph qualify for the dataset?\033[0m")

                buttons_for_step = {
                    value: widgets.Button(description=desc, button_style=PARAGRAPH_BUTTON_STYLES[value], layout=widgets.Layout(width='auto', margin='2px'))
                    for value, desc in PARAGRAPH_ANSWERS.items()
                }
                active_widgets['buttons'] = list(buttons_for_step.values())
                for value, button in buttons_for_step.items():
                    button.on_click(lambda b, v=value: handle_click(v))
                button_box = widgets.HBox(active_widgets['buttons'])
                main_display_area.children += (current_step_output, button_box)

            else:
                tagging_state['stage'] = f"path_{path_idx + 1}_valid"
                display_current_stage() # Re-render

        elif stage == 'path_1_valid' or stage == 'path_2_valid':
            path_num_str = "1" if path_idx == 0 else "2"
            with current_step_output:
                print("=" * 120)
                print(f"\033[1mIs Reasoning Path {path_num_str} valid overall?\033[0m")

            buttons_for_step = { value: widgets.Button(description=desc, button_style=YES_NO_BUTTON_STYLES[value], layout=widgets.Layout(width='auto', margin='2px'))
                 for value, desc in YES_NO_ANSWERS.items() }
            active_widgets['buttons'] = list(buttons_for_step.values())
            for value, button in buttons_for_step.items(): button.on_click(lambda b, v=value: handle_click(v))
            button_box = widgets.HBox(active_widgets['buttons'])
            main_display_area.children += (current_step_output, button_box)
            print("=" * 120)

        elif stage == 'sample_valid':
            with current_step_output:
                print("=" * 120)
                print("\033[1mIs this QA Sample valid overall?\033[0m")

            buttons_for_step = { value: widgets.Button(description=desc, button_style=YES_NO_BUTTON_STYLES[value], layout=widgets.Layout(width='auto', margin='2px'))
                 for value, desc in YES_NO_ANSWERS.items() }
            active_widgets['buttons'] = list(buttons_for_step.values())
            for value, button in buttons_for_step.items(): button.on_click(lambda b, v=value: handle_click(v))
            button_box = widgets.HBox(active_widgets['buttons'])
            main_display_area.children += (current_step_output, button_box)


    # --- Button Click Handler ---
    def handle_click(response_value):
        nonlocal tagging_state, active_widgets

        subset_idx = tagging_state['dataset_subset_idx']
        if tagging_state['stage'] == 'done': return

        current_sample = dataset_subset[subset_idx]
        sample_id = current_sample.get('ID', f'UnknownID_{start_idx + subset_idx}') # Use subset index if ID missing
        stage = tagging_state['stage']
        response_text = ""
        id_type = ""
        record_id = ""

        # --- Determine Response Text and ID details ---
        if stage == 'paragraph':
             response_text = PARAGRAPH_ANSWERS.get(response_value, f"Val:{response_value}")
             id_type = "Paragraph"
             path_key = 'Reasoning path 1' if tagging_state['path_idx'] == 0 else 'Reasoning path 2'
             current_path = current_sample.get(path_key, [])
             step_idx = tagging_state['paragraph_step_idx']
             if step_idx < len(current_path):
                 record_id = current_path[step_idx].get('Paragraph', 'MISSING_PID')
        elif stage == 'path_1_valid':
             response_text = YES_NO_ANSWERS.get(response_value, f"Val:{response_value}")
             id_type = "Path"
             record_id = f"{sample_id}_Path1"
        elif stage == 'path_2_valid':
             response_text = YES_NO_ANSWERS.get(response_value, f"Val:{response_value}")
             id_type = "Path"
             record_id = f"{sample_id}_Path2"
        elif stage == 'sample_valid':
             response_text = YES_NO_ANSWERS.get(response_value, f"Val:{response_value}")
             id_type = "Sample"
             record_id = sample_id

        # --- Disable buttons and show chosen text ---
        if active_widgets.get('buttons'):
            # 1. Disable all buttons in the current active set
            for btn in active_widgets['buttons']:
                btn.disabled = True

            # 2. Create an HTML widget to display the choice
            choice_html = widgets.HTML(
                value=f"<p style='margin-top: 10px;'>➡️ <b>Your choice:</b> {response_text}</p>"
            )

            # 3. Find the HBox containing the buttons in the main display children
            #    and replace it with the HTML choice widget.
            current_children = list(main_display_area.children)
            button_box_found = False
            for i in range(len(current_children) - 1, -1, -1):  # Iterate backwards
                child = current_children[i]
                # Check if it's an HBox and contains the buttons we just disabled
                if isinstance(child, widgets.HBox) and active_widgets['buttons'] and child.children[0] in \
                        active_widgets['buttons']:
                    current_children[i] = choice_html  # Replace HBox with HTML
                    button_box_found = True
                    break  # Stop after replacing

            if button_box_found:
                main_display_area.children = tuple(current_children)  # Update the display area
            else:
                # Fallback: If we couldn't find the button box to replace,
                # append the choice text to the last output area (less ideal layout)
                print(f"Debug: Button HBox not found for replacement. Appending choice.")
                if active_widgets.get('output_area'):
                    with active_widgets['output_area']:
                        print(f"\n -> Your choice: \033[1m{response_text}\033[0m")

            # 4. Clear the references to the now-handled widgets
            active_widgets['buttons'] = None
            active_widgets['output_area'] = None

        # --- Record Response ---
        if id_type and record_id:
             record_and_update_df(id_type, record_id, response_value, response_text)

        # --- Advance State ---
        if stage == 'paragraph':
            tagging_state['paragraph_step_idx'] += 1
        elif stage == 'path_1_valid':
             tagging_state['path_idx'] = 1
             tagging_state['paragraph_step_idx'] = 0
             tagging_state['stage'] = 'paragraph'
        elif stage == 'path_2_valid':
             tagging_state['stage'] = 'sample_valid'
        elif stage == 'sample_valid':
             tagging_state['dataset_subset_idx'] += 1 # Move to next sample index
             if tagging_state['dataset_subset_idx'] >= len(dataset_subset):
                 tagging_state['stage'] = 'done'
             else:
                 tagging_state['path_idx'] = 0
                 tagging_state['paragraph_step_idx'] = 0
                 tagging_state['stage'] = 'sample_start' # Trigger clear and display new sample

        # --- Display Next Step ---
        display_current_stage()


    # --- Initial Display ---
    display(
        title_area,
        progress_area,
        main_display_area,
        download_button_area,
        completion_area
    )
    tagging_state['stage'] = 'sample_start'
    display_current_stage()

    # --- Return Holder ---
    return final_result_holder
