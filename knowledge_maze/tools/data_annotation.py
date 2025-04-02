import os
import json
import base64
import textwrap

import pandas as pd
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display, HTML

from dotenv import load_dotenv

from knowledge_maze.create_dataset.create_data import get_edge_paragraph_ids, get_middle_paragraph_ids

load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')

REASONING_PATH_QUESTIONS = [
    "Correct answer type?",
    "Counter answers are relevant?",
    "No Similar sub-questions?",
    "Sub-questions are relevant?"
]

# Load original data :
QA_FILE_PATH = f"{REPO_PATH}\data_files\\merged_dataset.json"
with open(QA_FILE_PATH, "r", encoding="utf-8") as file:
    QA_DATASET = json.load(file)

# Load the reasoning :
PARAGRAPHS_FILE_PATH = f"{REPO_PATH}\data_files\\knowledge_maze_qa.json"
with open(PARAGRAPHS_FILE_PATH, "r", encoding="utf-8") as file:
    REASONING_DATASET = json.load(file)

# Load the paragraphs :
PARAGRAPHS_FILE_PATH = f"{REPO_PATH}\data_files\\knowledge_maze_paragraphs.json"
with open(PARAGRAPHS_FILE_PATH, "r", encoding="utf-8") as file:
    PARAGRAPHS_DATASET = json.load(file)


def wrap_text(text: str, ident: str = '') -> str:
    """Wraps a given text using textwrap."""
    wrapper = textwrap.TextWrapper(width=95)
    return f'\n{ident}'.join(wrapper.wrap(text))


def display_qa_sample(sample: dict) -> None:
    """
    Displays detailed QA sample information with line wrapping using the textwrap module.

    Args:
        sample (dict): A dictionary containing QA sample data.

    Returns:
        None
    """
    original_answer = get_original_answer(sample['Question'])
    sub_questions_chain_1 = [q['Previous question'] for q in sample['Reasoning path 1']]
    sub_questions_chain_2 = [q['Previous question'] for q in sample['Reasoning path 2']]

    paragraphs = get_middle_paragraph_ids(sample) + get_edge_paragraph_ids(sample)

    print('=' * 100)
    print(f"ID: {sample['ID']}")
    print(f"Type: {sample['Answer type']}")
    print(wrap_text(f"Question: {sample['Question']}", ident=' '*10))
    print(wrap_text(f"Original answer from dataset: {original_answer}"))
    print("\nAnswers:")
    for key, value in sample['Answer'].items():
        print(f"- {value} ({key})")
    print("\nSub-questions path 1:")
    for i, question in enumerate(sub_questions_chain_1):
        print(wrap_text(f" {i+1}. {question} --> {get_original_answer(question)}", ident='    '))
    print("\nSub-questions path 2:")
    for i, question in enumerate(sub_questions_chain_2):
        print(wrap_text(f" {i+1}. {question} --> {get_original_answer(question)}", ident='    '))

    # print("\nParagraphs:")
    # for i, paragraph in enumerate(paragraphs):
    #     print(wrap_text(f" {i+1}. {PARAGRAPHS_DATASET[paragraph]}", ident='    '), '\n')
    print('=' * 100)


def get_original_answer(question):
    """
    Retrieves the memory answer for a given question.

    Args:
        question (str): The question to retrieve the memory answer for.

    Returns:
        str or None: The normalized memory answer if found; otherwise, None.
        """
    memory_answer = next(
        (sample['Answer'] for sample in QA_DATASET if sample['Question'] == question),
        None
    )
    return memory_answer


# --- Placeholder for the download button function ---
def create_download_button(df, download_button_area):
    """Creates a client-side download link for the DataFrame."""
    try:
        df_copy = df.copy()
        try:
            df_copy['Text'] = df_copy['Text'].astype(str)
        except Exception:
            pass # Ignore conversion errors, proceed anyway

        csv_data = df_copy.to_csv(index=False)
        b64 = base64.b64encode(csv_data.encode('utf-8')).decode('utf-8')
        filename = "tagging_results.csv"
        button_html = f'''
        <a download="{filename}" href="data:text/csv;charset=utf-8;base64,{b64}" target="_blank" style="text-decoration: none;">
            <button style="padding: 6px 12px; font-size: 14px; color: #fff; background-color: #007bff; border-color: #007bff; border: 1px solid transparent; border-radius: .25rem; cursor: pointer;">
                Download CSV
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

# --- Main Interactive Function ---

def interactive_tagging():
    """
    Creates an interactive tagging system. Displays results in wide format.

    Returns:
        dict: A dictionary {'results_list': list, 'final_df': None}.
              'final_df' will be populated with the wide DataFrame *after*
              the user completes the tagging process in the UI.
              Check the UI for 'Tagging Complete!' before accessing 'final_df'.
    """

    # --- State Variables ---
    results_list = [] # Raw results stored here
    current_index = 0
    # ** NEW: Holder for the final DataFrame **
    # Using a dictionary allows modification by the nested function.
    final_result_holder = {'results_list': results_list, 'final_df': None}

    # --- Output areas ---
    text_display_area = widgets.Output()
    question_display_areas = [widgets.Output() for _ in range(len(REASONING_PATH_QUESTIONS))]
    final_result_area = widgets.Output()
    download_button_area = widgets.Output()

    # --- Buttons ---
    buttons = []
    for i in range(len(REASONING_PATH_QUESTIONS)):
        yes_button = widgets.Button(description="Yes", button_style="success", layout=widgets.Layout(width='auto'))
        no_button = widgets.Button(description="No", button_style="danger", layout=widgets.Layout(width='auto'))
        buttons.append((yes_button, no_button))

    # --- Core Logic Functions ---

    def show_final_results():
        """Internal function to process and display final results."""
        # Clear interactive elements
        text_display_area.clear_output()
        for output_area in question_display_areas:
            output_area.clear_output()
        final_result_area.clear_output()
        download_button_area.clear_output()

        if not results_list:
            with final_result_area:
                print("⚠️ No results were recorded.")
            return

        try:
            # Create, Pivot, and Clean DataFrame
            df_long = pd.DataFrame(results_list, columns=["Sample ID", "Question", "Answer"])
            df_long['Text_Str'] = df_long['Sample ID'].astype(str) # Assume string conversion works

            df_wide = df_long.pivot_table(index='Text_Str', columns='Question', values='Answer', aggfunc='first')
            df_wide.columns.name = None
            df_wide = df_wide.reset_index().rename(columns={'Text_Str': 'Sample ID'})
            cols = ['Sample ID'] + [col for col in df_wide.columns if col != 'Sample ID']
            df_wide = df_wide[cols]

            # ** NEW: Store the final DataFrame in the holder **
            final_result_holder['final_df'] = df_wide

            # Display results
            with final_result_area:
                print("✅ Tagging Complete! Here are your answers:\n")
                with pd.option_context('display.max_rows', None, 'display.max_colwidth', 100, 'display.width', 1000):
                    display(df_wide)

            # Create download button
            create_download_button(df_wide, download_button_area)

        except Exception as e:
            with final_result_area:
                print(f"❌ Error processing or pivoting results: {e}")
                print("\nRaw results list:")
                print(results_list)
            # Ensure final_df remains None or handle appropriately
            final_result_holder['final_df'] = None # Indicate error


    def on_button_click(q_index, answer):
        nonlocal current_index # Required to modify current_index
        if current_index >= len(REASONING_DATASET):
             return

        current_data_item = REASONING_DATASET[current_index]['ID']
        results_list.append((current_data_item, REASONING_PATH_QUESTIONS[q_index], answer))

        buttons[q_index][0].disabled = True
        buttons[q_index][1].disabled = True
        with question_display_areas[q_index]:
            question_display_areas[q_index].clear_output(wait=True)
            print(f"{REASONING_PATH_QUESTIONS[q_index]} → {answer}")

        answered_count = sum(1 for res in results_list if res[0] is current_data_item)

        if answered_count == len(REASONING_PATH_QUESTIONS):
            current_index += 1
            if current_index < len(REASONING_DATASET):
                show_next_text()
            else:
                # Call the internal final results function
                show_final_results()


    def show_next_text():
        text_display_area.clear_output(wait=True)
        for i, output_area in enumerate(question_display_areas):
            output_area.clear_output(wait=True)
            buttons[i][0].disabled = False
            buttons[i][1].disabled = False

        with text_display_area:
            print(f"\n--- Item {current_index + 1} of {len(REASONING_DATASET)} ---")
            display_qa_sample(REASONING_DATASET[current_index])


        for i, output_area in enumerate(question_display_areas):
            with output_area:
                print(REASONING_PATH_QUESTIONS[i])
                display(widgets.HBox([buttons[i][0], buttons[i][1]]))


    # --- Setup Button Events ---
    for q_idx, (yes_btn, no_btn) in enumerate(buttons):
        def create_handler(index, ans):
            def handler(b):
                on_button_click(index, ans)
            return handler
        yes_btn.on_click(create_handler(q_idx, "Yes"))
        no_btn.on_click(create_handler(q_idx, "No"))

    # --- Initial Display ---
    display(text_display_area)
    for area in question_display_areas:
        display(area)
    display(final_result_area)
    display(download_button_area)

    if len(REASONING_DATASET) > 0:
        show_next_text()
    else:
        with final_result_area:
            print("⚠️ The dataset (REASONING_DATASET) is empty. Nothing to tag.")

    # --- Return the holder ---
    # The function returns *immediately* after setup.
    # 'final_df' in the returned dict is still None at this point.
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
    ax.set_title('Tagging Session Summary')
    ax.set_ylim(0, 105) # Extend y-limit slightly for text visibility
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=30, ha='right') # Rotate labels if they might overlap

    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()