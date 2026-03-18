import re
import random
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from urllib.parse import unquote
from IPython.display import display, HTML

from knowledge_maze.create_dataset.create_questions.wikipedia_mining import ENTITY_GROUPS


def plot_strategy_sunburst(valid_results):
    """
    Generates a two-layered (Sunburst-style) pie chart from a list of samples.

    :param valid_results: List of dicts, each containing a 'Selected strategy' key.
    """
    # 1. Define the Category Mapping
    mapping = {
        "Numerical": [
            "AGGREGATION", "EXTREME_NUMERICAL", "TOP_K_RANKING",
            "COMPARATIVE_DIFFERENCE", "FILTER_COUNT_NUMERICAL"
        ],
        "Categorical": [
            "UNIQUE_DIVERSITY", "FREQUENCY_MODE",
            "CATEGORICAL_FILTER_LOOKUP", "COUNT_TOTAL"
        ],
        "Temporal": [
            "EXTREME_TEMPORAL", "FILTER_COUNT_TEMPORAL"
        ]
    }

    # 2. Tally the results from the input list
    strat_counts = Counter([sample.get('Selected strategy', 'UNKNOWN') for sample in valid_results])

    # Check for strategies not in the mapping and add to 'Other'
    known_strats = [s for sublist in mapping.values() for s in sublist]
    other_strats = [s for s in strat_counts if s not in known_strats]
    if other_strats:
        mapping["Other"] = other_strats

    # 3. Organize data for the layers
    inner_labels, inner_sizes, inner_colors = [], [], []
    outer_labels, outer_sizes, outer_colors = [], [], []

    # Define color maps for visual distinction
    colors_map = {
        "Numerical": plt.cm.Blues,
        "Categorical": plt.cm.Reds,
        "Temporal": plt.cm.Greens,
        "Other": plt.cm.Greys
    }

    for group, strats in mapping.items():
        group_total = sum(strat_counts[s] for s in strats)
        if group_total > 0:
            inner_labels.append(group)
            inner_sizes.append(group_total)
            inner_colors.append(colors_map.get(group, plt.cm.Purples)(0.6))

            # Build the outer ring (Sub-strategies)
            for i, s in enumerate(strats):
                count = strat_counts[s]
                if count > 0:
                    outer_labels.append(s)
                    outer_sizes.append(count)
                    # Create a gradient effect for sub-strategies
                    base_color = colors_map.get(group, plt.cm.Purples)
                    outer_colors.append(base_color(0.4 - (i * 0.05)))

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#f8fafc')

    # Outer Ring (Specific Strategies)
    # Using wedgeprops to create the 'donut' look
    ax.pie(outer_sizes, labels=outer_labels, radius=1, colors=outer_colors,
           wedgeprops=dict(width=0.3, edgecolor='w'),
           pctdistance=0.85, autopct='%1.1f%%',
           textprops={'fontsize': 9})

    # Inner Ring (Parent Categories)
    ax.pie(inner_sizes, labels=inner_labels, radius=0.7, colors=inner_colors,
           labeldistance=0.4, wedgeprops=dict(width=0.3, edgecolor='w'),
           textprops={'fontsize': 11, 'fontweight': 'bold', 'color': 'black'})

    ax.set(aspect="equal")
    plt.title('Question Generation: Strategy Distribution',
              pad=20, fontsize=16, fontweight='bold', color='#1e293b')

    plt.tight_layout()
    plt.show()


def plot_evidence_distribution(data_samples):
    """
    Plots a histogram of the number of rows (evidence) in 'Relevant table data'.
    """
    # Extract lengths
    counts = [len(s.get('Relevant table data', [])) for s in data_samples]

    plt.figure(figsize=(10, 6), facecolor='#f8fafc')
    sns.set_theme(style="whitegrid")

    # Create histogram
    ax = sns.histplot(counts, bins=min(len(set(counts)), 20),
                      kde=True, color='#0ea5e9', edgecolor='#0284c7', alpha=0.7)

    plt.title('Distribution of Evidence (Table Rows per Question)',
              fontsize=15, fontweight='bold', pad=20)
    plt.xlabel('Number of Evidence Rows', fontsize=12)
    plt.ylabel('Frequency (Number of Questions)', fontsize=12)

    # Add a vertical line for the median
    median_val = sum(counts) / len(counts) if counts else 0
    plt.axvline(median_val, color='#ef4444', linestyle='--', label=f'Mean: {median_val:.1f}')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_subject_distribution(data_samples, entity_groups=ENTITY_GROUPS):
    """
    Plots a horizontal bar chart of subjects with their defining keywords in brackets.

    :param data_samples: List of dicts containing 'Wiki link' or 'Wiki url'
    :param entity_groups: Dict { 'Subject': ['keyword1', 'keyword2', ...] }
    """
    subjects_found = []

    for s in data_samples:
        url = s.get('Wiki link', '') or s.get('Wiki url', '')
        if not url:
            continue

        # 1. Extract and Clean Page Title
        title = unquote(url.split('/')[-1].split('#')[0]).replace('_', ' ').lower()
        clean_title = re.sub(r'^list of ', '', title)

        # 2. Keyword Matching (Score based on overlap)
        best_subject = "Other/General"
        max_matches = 0

        for subject, keywords in entity_groups.items():
            matches = sum(1 for kw in keywords if kw.lower() in clean_title)
            if matches > max_matches:
                max_matches = matches
                best_subject = subject

        subjects_found.append(best_subject)

    # 3. Tally and Sort
    counts = Counter(subjects_found)
    # Sort subjects by frequency (descending)
    sorted_data = counts.most_common()

    # 4. Prepare Labels with Keywords
    final_labels = []
    final_counts = []

    for subj, count in reversed(sorted_data):  # Reverse for barh (top-down)
        # Get keywords from the original dict, truncate to top 3
        kws = entity_groups.get(subj, [])
        kw_string = ", ".join(kws[:3]) + ("..." if len(kws) > 3 else "")

        # Format: Subject Name (kw1, kw2, kw3...)
        label = f"{subj}\n[{kw_string}]"
        final_labels.append(label)
        final_counts.append(count)

    # 5. Plotting
    plt.figure(figsize=(12, len(final_labels) * 0.9), facecolor='#f8fafc')

    # Professional Blue-to-Emerald palette
    colors = plt.cm.viridis([i / len(final_labels) for i in range(len(final_labels))])

    bars = plt.barh(final_labels, final_counts, color=colors, edgecolor='#1e293b', alpha=0.8)

    # Add count labels on the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{int(width)}', va='center', fontsize=10, fontweight='bold', color='#1e293b')

    plt.title('Dataset Composition: Subject Clusters', fontsize=16, fontweight='bold', pad=30)
    plt.xlabel('Number of Questions', fontsize=12, labelpad=15)

    # Aesthetics
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_samples(dataset, num_samples=10):
    """
    Ultra-stable, left-aligned, thin card viewer.
    Uses legacy HTML attributes for 100% alignment reliability.
    """
    if not dataset:
        return print("Dataset is empty.")

    samples = random.sample(dataset, min(num_samples, len(dataset)))

    # 1. Start a left-aligned container
    html_out = '<div align="left" style="text-align: left; background-color: #ffffff; font-family: sans-serif;">'

    for i, s in enumerate(samples):
        strat = str(s.get('Selected strategy', 'N/A'))
        group = str(s.get('Group name', 'N/A'))
        question = str(s.get('Question', 'No Question'))
        table_data = s.get('Relevant table data', [])

        # 2. Each card is a table with fixed width 550, explicitly aligned left
        html_out += f'''
        <table align="left" width="550" cellpadding="0" cellspacing="0" style="margin-bottom: 10px; border: 1px solid #e2e8f0; table-layout: fixed; text-align: left;">
            <tr>
                <td align="left" style="padding: 10px; background-color: #f8fafc; border-bottom: 1px solid #e2e8f0; text-align: left;">
                    <div align="left" style="color: #64748b; font-size: 10px; font-weight: bold; text-align: left;">{strat}</div>
                    <div align="left" style="color: #94a3b8; font-size: 10px; text-align: left;">{group}</div>
                </td>
            </tr>
            <tr>
                <td align="left" style="padding: 15px; color: #1e293b; font-size: 14px; font-weight: 600; line-height: 1.4; text-align: left;">
                    {question}
                </td>
            </tr>
            <tr>
                <td align="left" style="padding: 0 15px 15px 15px; text-align: left;">
                    <table align="left" width="100%" border="0" cellpadding="4" cellspacing="0" style="border-collapse: collapse; text-align: left;">
        '''

        # 3. Evidence Table (Left Aligned)
        if table_data and isinstance(table_data, list):
            headers = list(table_data[0].keys())
            html_out += '<tr>'
            for h in headers:
                html_out += f'<th align="left" style="font-size: 10px; color: #64748b; border-bottom: 1px solid #cbd5e1; text-align: left;">{h}</th>'
            html_out += '</tr>'

            for row in table_data[:3]:
                html_out += '<tr>'
                for val in row.values():
                    html_out += f'<td align="left" style="font-size: 10px; color: #475569; border-bottom: 1px solid #f1f5f9; text-align: left;">{val}</td>'
                html_out += '</tr>'

        html_out += '</table></td></tr></table>'

        # 4. Clear the float and add a separation line
        html_out += '<div style="clear: both; width: 550px; border-top: 2px solid #334155; margin: 20px 0;"></div>'

    html_out += '</div>'
    display(HTML(html_out))
