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
    2. A 100% stacked histogram showing the composition of scores within each bin.

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 1.2]})
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
    # --- Step 5: Plot 2 - 100% Stacked Histogram (on ax2) ---

    # Define the bins for the histogram
    bins = np.linspace(0, 1, 21)  # 20 bins from 0.0 to 1.0

    # Manually bin the data for each category
    binned_counts = {label: np.histogram(distributions[label], bins=bins)[0] for label in plot_labels}

    # Calculate the total count in each bin across all categories
    bin_totals = np.sum(list(binned_counts.values()), axis=0)

    # Prepare for plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    bottoms = np.zeros(len(bin_totals))  # Initialize the bottom of the bars at y=0

    for i, label in enumerate(plot_labels):
        # Calculate the percentage height of each bar for the current category
        # Using np.divide is safe against division by zero for empty bins
        percentages = np.divide(binned_counts[label], bin_totals,
                                out=np.zeros_like(bin_totals, dtype=float),
                                where=bin_totals != 0) * 100

        # Plot the bar for the current category on top of the previous one
        ax2.bar(
            bin_centers,
            percentages,
            label=label,
            color=plot_colors[i],
            width=bin_width,
            bottom=bottoms
        )
        # Update the bottoms for the next category
        bottoms += percentages

    ax2.set_title("Composition of Scores within each Bin", fontsize=14)
    ax2.set_xlabel("Probability Score", fontsize=12)
    ax2.set_ylabel("Percentage of Samples in Bin (%)", fontsize=12)
    ax2.set_ylim(0, 100)  # Ensure Y-axis is fixed from 0 to 100
    ax2.legend(title="Answer Categories")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.set_xticks(bins)

    # --- Step 6: Finalize and Show/Save ---
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {output_path}")

    plt.show()