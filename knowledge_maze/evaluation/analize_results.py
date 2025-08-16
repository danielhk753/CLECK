import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.gridspec as gridspec
from typing import List, Dict, Optional

# --- Helper Plotting Functions - Memory Test ---

def _plot_mean_distribution_pie(ax, averages, plot_labels, plot_colors):
    """Plots the pie chart for mean probability distribution."""
    pie_values = [averages[label] for label in plot_labels]
    wedges, _, _ = ax.pie(
        pie_values,
        autopct='%1.1f%%',
        startangle=90,
        colors=plot_colors,
        wedgeprops=dict(edgecolor='w')
    )
    ax.axis('equal')
    ax.set_title("Mean Probability Distribution", fontsize=14)
    # Add a legend that is shared between the two pie charts
    ax.legend(wedges, plot_labels, title="Answer Categories", loc="center left", bbox_to_anchor=(1.05, 0.5))

def _plot_winner_distribution_pie(ax, winner_counts, color_map):
    """Plots the pie chart for the distribution of 'winning' answers."""
    winner_labels = sorted(winner_counts.keys(), key=lambda k: winner_counts[k], reverse=True)
    winner_values = [winner_counts[label] for label in winner_labels]
    winner_colors = [color_map.get(label, color_map.get('Other', 'lightgray')) for label in winner_labels]

    if not winner_values:
        ax.text(0.5, 0.5, "No 'winning' answers found.", ha='center', va='center')
        ax.set_title("Distribution of 'Winning' Answers", fontsize=14)
        return

    ax.pie(
        winner_values,
        autopct='%1.1f%%',
        startangle=90,
        colors=winner_colors,
        wedgeprops=dict(edgecolor='w')
    )
    ax.axis('equal')
    ax.set_title("Distribution of 'Winning' Answers", fontsize=14)

def _plot_score_distribution_bar(ax, distributions, plot_labels, plot_colors):
    """Plots the grouped, intra-category normalized bar chart."""
    num_bins = 10
    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    normalized_binned_counts = {}
    for label in plot_labels:
        counts, _ = np.histogram(distributions[label], bins=bins)
        total_occurrences = counts.sum()
        if total_occurrences > 0:
            normalized_binned_counts[label] = (counts / total_occurrences) * 100
        else:
            normalized_binned_counts[label] = np.zeros(num_bins)

    num_categories = len(plot_labels)
    bar_width = (bins[1] - bins[0]) * 0.8 / num_categories

    for i, label in enumerate(plot_labels):
        offset = (i - (num_categories - 1) / 2.0) * bar_width
        bar_positions = bin_centers + offset
        heights = normalized_binned_counts[label]
        ax.bar(bar_positions, heights, width=bar_width, label=label, color=plot_colors[i])

    ax.set_title("Intra-Category Score Distribution", fontsize=16, pad=20)
    ax.set_xlabel("Probability Score Bins", fontsize=12)
    ax.set_ylabel("Percent of Category's Occurrences (%)", fontsize=12)
    ax.legend(title="Answer Categories")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xticks(bin_centers)
    xtick_labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(num_bins)]
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")


# --- Main Orchestrator Functions ---

def plot_memory_test_summary(
    data: List[Dict[str, float]],
    title: str = "Overall Answer Distribution Analysis",
    output_path: Optional[str] = None
) -> None:
    """
    Aggregates results and plots summary graphs in a 2-row layout:
    Row 1: Two pie charts (mean probability and 'winning' answer distribution).
    Row 2: A grouped bar chart for intra-category score distribution.

    Args:
        data: A list of score dictionaries.
        title: The main title for the entire figure.
        output_path: Optional file path to save the generated figure.
    """
    if not data:
        print("Data is empty. Nothing to plot.")
        return

    # --- Step 1: Aggregate and Restructure the Data ---
    sums = defaultdict(float)
    distributions = defaultdict(list)
    winner_counts = defaultdict(int)
    num_samples = len(data)

    for result_dict in data:
        if result_dict:
            winner_key = max(result_dict, key=result_dict.get)
            winner_counts[winner_key] += 1
        for key, value in result_dict.items():
            sums[key] += value
            distributions[key].append(value)

    averages = {key: total_sum / num_samples for key, total_sum in sums.items()}
    plot_labels = sorted([k for k, v in averages.items() if v > 0], key=lambda k: averages[k], reverse=True)

    if not plot_labels:
        print("All average scores are zero. Nothing to plot.")
        return

    # --- Step 2: Define a consistent color map ---
    color_map = {
        'Memory answer': 'dodgerblue',
        'Counter answer 1': 'lightgreen',
        'Counter answer 2': 'gold',
        'Distractor answer': 'salmon',
        'Invalid': 'silver',
        'No Answer': 'orange',
        'Other Answer': 'mediumpurple',
        'Other': 'lightgray'
    }
    plot_colors = [color_map.get(label, color_map['Other']) for label in plot_labels]

    # --- Step 3: Create the Figure with a Custom Grid Layout ---
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2])

    ax_pie1 = fig.add_subplot(gs[0, 0])
    ax_pie2 = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :]) # Bar chart spans the entire second row

    fig.suptitle(title, fontsize=20, weight='bold')

    # --- Step 4: Call Helper Functions to Plot on Each Subplot ---
    _plot_mean_distribution_pie(ax_pie1, averages, plot_labels, plot_colors)
    _plot_winner_distribution_pie(ax_pie2, winner_counts, color_map)
    _plot_score_distribution_bar(ax_bar, distributions, plot_labels, plot_colors)

    # --- Step 5: Finalize and Show/Save ---
    plt.tight_layout(rect=[0, 0, 0.95, 0.95]) # Adjust rect to make space for suptitle and legend

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {output_path}")

    plt.show()


def plot_context_test_summary(
    data: List[Dict[str, float]],
    title: str = "Overall Answer Distribution Analysis",
    output_path: Optional[str] = None
) -> None:
    """
    Aggregates results and plots summary graphs in a 2-row layout:
    Row 1: Two pie charts (mean probability and 'winning' answer distribution).
    Row 2: A grouped bar chart for intra-category score distribution.

    Args:
        data: A list of score dictionaries.
        title: The main title for the entire figure.
        output_path: Optional file path to save the generated figure.
    """
    if not data:
        print("Data is empty. Nothing to plot.")
        return

    # --- Step 1: Aggregate and Restructure the Data ---
    sums = defaultdict(float)
    distributions = defaultdict(list)
    winner_counts = defaultdict(int)
    num_samples = len(data)

    for result_dict in data:
        if result_dict:
            winner_key = max(result_dict, key=result_dict.get)
            winner_counts[winner_key] += 1
        for key, value in result_dict.items():
            sums[key] += value
            distributions[key].append(value)

    averages = {key: total_sum / num_samples for key, total_sum in sums.items()}
    plot_labels = sorted([k for k, v in averages.items() if v > 0], key=lambda k: averages[k], reverse=True)

    if not plot_labels:
        print("All average scores are zero. Nothing to plot.")
        return

    # --- Step 2: Define a consistent color map ---
    color_map = {
        'Memory answer': 'dodgerblue',
        'Counter answer': 'lightgreen',
        'Distractor answer 1': 'gold',
        'Distractor answer 2': 'salmon',
        'Invalid': 'silver',
        'No Answer': 'orange',
        'Other Answer': 'mediumpurple',
        'Other': 'lightgray'
    }
    plot_colors = [color_map.get(label, color_map['Other']) for label in plot_labels]

    # --- Step 3: Create the Figure with a Custom Grid Layout ---
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2])

    ax_pie1 = fig.add_subplot(gs[0, 0])
    ax_pie2 = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :]) # Bar chart spans the entire second row

    fig.suptitle(title, fontsize=20, weight='bold')

    # --- Step 4: Call Helper Functions to Plot on Each Subplot ---
    _plot_mean_distribution_pie(ax_pie1, averages, plot_labels, plot_colors)
    _plot_winner_distribution_pie(ax_pie2, winner_counts, color_map)
    _plot_score_distribution_bar(ax_bar, distributions, plot_labels, plot_colors)

    # --- Step 5: Finalize and Show/Save ---
    plt.tight_layout(rect=[0, 0, 0.95, 0.95]) # Adjust rect to make space for suptitle and legend

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {output_path}")

    plt.show()