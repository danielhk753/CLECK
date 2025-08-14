import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional


def plot_aggregated_summary(
        data: List[Dict[str, float]],
        title: str = "Overall Answer Distribution Analysis",
        output_path: Optional[str] = None
) -> None:
    """
    Aggregates results and plots two summary graphs side-by-side:
    1. A pie chart showing the mean probability for each answer category.
    2. A grouped bar chart showing the intra-category score distribution
       (each category's distribution sums to 100%).

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
    num_samples = len(data)

    for result_dict in data:
        for key, value in result_dict.items():
            sums[key] += value
            distributions[key].append(value)

    averages = {key: total_sum / num_samples for key, total_sum in sums.items()}

    plot_labels = sorted([key for key, value in averages.items() if value > 0], key=lambda k: averages[k], reverse=True)
    if not plot_labels:
        print("All average scores are zero. Nothing to plot.")
        return

    pie_values = [averages[label] for label in plot_labels]

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

    # --- Step 3: Create the Figure with Two Subplots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 1.5]})
    fig.suptitle(title, fontsize=18, weight='bold')

    # --- Step 4: Plot 1 - The Pie Chart (on ax1) ---
    wedges, _, _ = ax1.pie(
        pie_values,
        autopct='%1.1f%%',
        startangle=90,
        colors=plot_colors,
        wedgeprops=dict(edgecolor='w')
    )
    ax1.axis('equal')
    ax1.set_title("Mean Probability Distribution", fontsize=14)
    ax1.legend(wedges, plot_labels, title="Answer Categories", loc="center left", bbox_to_anchor=(0.9, 0, 0.5, 1))

    # === MODIFICATION IS HERE ===
    # --- Step 5: Plot 2 - Grouped, Intra-Category Normalized Bar Chart ---

    # Define the bins and bin centers
    num_bins = 10
    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Manually bin the data for each category and normalize it
    normalized_binned_counts = {}
    for label in plot_labels:
        counts, _ = np.histogram(distributions[label], bins=bins)
        total_occurrences = counts.sum()
        if total_occurrences > 0:
            normalized_binned_counts[label] = (counts / total_occurrences) * 100
        else:
            normalized_binned_counts[label] = np.zeros(num_bins)

    # Calculate bar positions for the grouped chart
    num_categories = len(plot_labels)
    bar_width = (bins[1] - bins[0]) * 0.8 / num_categories  # Make bars narrower to fit in groups

    for i, label in enumerate(plot_labels):
        # Calculate the offset for this category's bars from the bin center
        offset = (i - (num_categories - 1) / 2.0) * bar_width
        bar_positions = bin_centers + offset

        # Get the heights for this category's bars
        heights = normalized_binned_counts[label]

        ax2.bar(
            bar_positions,
            heights,
            width=bar_width,
            label=label,
            color=plot_colors[i]
        )

    ax2.set_title("Intra-Category Score Distribution", fontsize=14)
    ax2.set_xlabel("Probability Score Bins", fontsize=12)
    ax2.set_ylabel("Percent of Category's Occurrences (%)", fontsize=12)
    ax2.legend(title="Answer Categories")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Set clear x-axis labels for the bins
    ax2.set_xticks(bin_centers)
    xtick_labels = [f'{bins[i]:.1f}-{bins[i + 1]:.1f}' for i in range(num_bins)]
    ax2.set_xticklabels(xtick_labels, rotation=45, ha="right")

    # --- Step 6: Finalize and Show/Save ---
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {output_path}")

    plt.show()