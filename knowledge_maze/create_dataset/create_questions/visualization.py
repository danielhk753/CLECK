import re
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from urllib.parse import unquote

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