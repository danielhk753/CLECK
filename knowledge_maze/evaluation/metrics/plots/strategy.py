import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv.main import load_dotenv
from collections import defaultdict
import matplotlib.patches as mpatches

from knowledge_maze.tools.utils import load_json

load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')

# =============================================================================
# CONFIGURATION & TAXONOMY
# =============================================================================

TAXONOMY_MAP = {
    # Group 1: Extremes & Boundaries
    "EXTREME_NUMERICAL": "Boundary Selection (Extremes)",
    "TOP_K_RANKING": "Boundary Selection (Extremes)",
    "EXTREME_TEMPORAL": "Boundary Selection (Extremes)",

    # Group 2: Global Distributions
    "FREQUENCY_MODE": "Global Distributions",
    "UNIQUE_DIVERSITY": "Global Distributions",

    # Group 3: Predicate Filters
    "FILTER_COUNT_NUMERICAL": "Predicate Filters",
    "FILTER_COUNT_TEMPORAL": "Predicate Filters",
    "CATEGORICAL_FILTER_LOOKUP": "Predicate Filters",

    # Ungrouped Baselines & Math Operations
    "COMPARATIVE_DIFFERENCE": "Ungrouped Baselines",
    "AGGREGATION": "Ungrouped Baselines",
    "COUNT_TOTAL": "Ungrouped Baselines"
}

COLOR_PALETTE = {
    "Boundary Selection (Extremes)": "#2ecc71",  # Vibrant Green
    "Global Distributions": "#9b59b6",  # Amethyst Purple
    "Predicate Filters": "#e74c3c",  # Coral Red
    "Ungrouped Baselines": "#78909c"  # Neutral Slate Gray
}

MARKER_PALETTE = {
    "Boundary Selection (Extremes)": "s",  # Square
    "Global Distributions": "D",  # Diamond
    "Predicate Filters": "o",  # Circle
    "Ungrouped Baselines": "^"  # Triangle
}

LEGEND_ORDER = [
    "Boundary Selection (Extremes)",
    "Global Distributions",
    "Predicate Filters",
    "Ungrouped Baselines"
]


# =============================================================================
# DATA PROCESSING
# =============================================================================

def build_strategy_dataframe(models_dict):
    """
    Builds per-strategy metrics dataframe from the parsed evaluation results.
    """
    rows = []

    for model_name, data in models_dict.items():
        strategy_stats = defaultdict(lambda: {
            "ack_count": 0,
            "total_iters": 0,
            "evidence_ack": 0,
            "claim_ack": 0
        })

        for sample in data.values():
            strategy = sample["strategy"]
            responses = sample.get("responses", [])

            for r in responses:
                if "judge_results" not in r:
                    continue

                jr = r["judge_results"]
                is_ack = jr.get("dual_awareness", False)
                classification = jr.get("classification", "other")

                # Track Awareness
                strategy_stats[strategy]["total_iters"] += 1
                if is_ack:
                    strategy_stats[strategy]["ack_count"] += 1

                    # Track Preference | Ack
                    if classification == "evidence":
                        strategy_stats[strategy]["evidence_ack"] += 1
                    elif classification == "claim":
                        strategy_stats[strategy]["claim_ack"] += 1

        # Final aggregation
        for strategy, stats in strategy_stats.items():
            total_iters = stats["total_iters"]
            ack_count = stats["ack_count"]

            ack_percent = (100 * ack_count / total_iters) if total_iters else 0
            total_ack_pref = stats["evidence_ack"] + stats["claim_ack"]
            evidence_given_ack = (100 * stats["evidence_ack"] / total_ack_pref) if total_ack_pref else 0

            rows.append({
                "Strategy": strategy,
                "Model": model_name,
                "Ack": ack_percent,
                "Evidence|Ack": evidence_given_ack
            })

    return pd.DataFrame(rows)


# =============================================================================
# VISUALIZATION SUITE
# =============================================================================

def apply_modern_spines(ax):
    """Helper to strip top/right spines for print-ready styling."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')


def plot_preference_bar_detailed(df, strategy_col="Strategy", evidence_col="Evidence|Ack", figsize=(10, 7)):
    """Plots the Evidence Preference horizontal bar chart."""
    grouped = df.groupby(strategy_col).agg({evidence_col: ["mean", "std"]}).reset_index()
    grouped.columns = [strategy_col, "ev_mean", "ev_std"]

    grouped["Taxonomy"] = grouped[strategy_col].map(TAXONOMY_MAP)
    grouped["Color"] = grouped["Taxonomy"].map(COLOR_PALETTE)
    grouped = grouped.sort_values("ev_mean", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(len(grouped))

    ax.barh(
        y_pos, grouped["ev_mean"], xerr=grouped["ev_std"],
        color=grouped["Color"], alpha=0.85, edgecolor="#2c3e50",
        linewidth=0.5, height=0.6,
        error_kw={'ecolor': '#2c3e50', 'elinewidth': 1.2, 'capsize': 3}
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(grouped[strategy_col], fontsize=10, fontweight="medium", color="#2c3e50")

    ax.set_xlabel("← Claim Preference         |         Evidence Preference →\nP(Evidence | Ack) [%]",
                  fontsize=11, fontweight="semibold", labelpad=12, color="#2c3e50")
    ax.set_title("Context Adherence Across Fine-Grained Operational Groups", fontsize=13, fontweight="bold", pad=15)

    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    apply_modern_spines(ax)

    legend_patches = [mpatches.Patch(color=COLOR_PALETTE[g], label=g, alpha=0.85) for g in LEGEND_ORDER]
    ax.legend(handles=legend_patches, loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0",
              fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_awareness_bar_detailed(df, strategy_col="Strategy", awareness_col="Ack", figsize=(10, 7)):
    """Plots the Conflict Awareness horizontal bar chart."""
    grouped = df.groupby(strategy_col).agg({awareness_col: ["mean", "std"]}).reset_index()
    grouped.columns = [strategy_col, "ack_mean", "ack_std"]

    grouped["Taxonomy"] = grouped[strategy_col].map(TAXONOMY_MAP)
    grouped["Color"] = grouped["Taxonomy"].map(COLOR_PALETTE)
    grouped = grouped.sort_values("ack_mean", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(len(grouped))

    ax.barh(
        y_pos, grouped["ack_mean"], xerr=grouped["ack_std"],
        color=grouped["Color"], alpha=0.85, edgecolor="#2c3e50",
        linewidth=0.5, height=0.6,
        error_kw={'ecolor': '#2c3e50', 'elinewidth': 1.2, 'capsize': 3}
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(grouped[strategy_col], fontsize=10, fontweight="medium", color="#2c3e50")

    ax.set_xlabel(
        "← Explicit Conflict Blindness         |         Explicit Conflict Recognition →\nConflict Awareness Probability: P(Ack) [%]",
        fontsize=11, fontweight="semibold", labelpad=12, color="#2c3e50")
    ax.set_title("Conflict Awareness Across Fine-Grained Operational Groups", fontsize=13, fontweight="bold", pad=15)

    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    apply_modern_spines(ax)

    legend_patches = [mpatches.Patch(color=COLOR_PALETTE[g], label=g, alpha=0.85) for g in LEGEND_ORDER]
    ax.legend(handles=legend_patches, loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0",
              fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_strategy_quadrant_detailed(df, strategy_col="Strategy", awareness_col="Ack", evidence_col="Evidence|Ack",
                                    figsize=(9, 8)):
    """Plots the 2D Behavioral Landscape Quadrant."""
    grouped = df.groupby(strategy_col).agg(
        {awareness_col: ["mean", "std"], evidence_col: ["mean", "std"]}).reset_index()
    grouped.columns = [strategy_col, "x_mean", "x_std", "y_mean", "y_std"]
    grouped["Taxonomy"] = grouped[strategy_col].map(TAXONOMY_MAP)

    fig, ax = plt.subplots(figsize=figsize)

    ax.axvline(x=grouped["x_mean"].mean(), color="#7f8c8d", linestyle="--", alpha=0.4, zorder=1)
    ax.axhline(y=grouped["y_mean"].mean(), color="#7f8c8d", linestyle="--", alpha=0.4, zorder=1)

    seen = set()
    for _, row in grouped.iterrows():
        tax = row["Taxonomy"]
        label = tax if tax not in seen else "__nolegend__"
        seen.add(tax)

        ax.scatter(
            row["x_mean"], row["y_mean"],
            color=COLOR_PALETTE[tax], marker=MARKER_PALETTE[tax],
            s=120, edgecolors="#2c3e50", linewidths=1, label=label, zorder=3
        )

        ax.text(
            row["x_mean"] + 0.8, row["y_mean"] + 0.5, row[strategy_col],
            fontsize=9, color="#2c3e50", fontweight="semibold", alpha=0.85, zorder=4
        )

    ax.set_xlabel("Conflict Awareness Probability: P(Ack) [%]", fontsize=11, labelpad=10)
    ax.set_ylabel("Evidence Preference Given Acknowledgement: P(Evidence | Ack) [%]", fontsize=11, labelpad=10)
    ax.set_title("Knowledge Conflict Behavioral Landscape Across Fine-Grained Groups", fontsize=12, fontweight="bold",
                 pad=15)

    ax.set_xlim(0, 60)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle=":", alpha=0.3, zorder=0)
    apply_modern_spines(ax)

    ax.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=10)

    # Quadrant Labels
    ax.text(2, 80, "Context Dominant\n(Low Awareness / High Deference)", fontsize=9, color="#7f8c8d", style="italic")
    ax.text(37, 94, "Strategic Alignment\n(High Awareness / High Deference)", fontsize=9, color="#7f8c8d",
            style="italic")
    ax.text(2, 3, "Parametric Stronghold\n(Low Awareness / Low Deference)", fontsize=9, color="#7f8c8d", style="italic")
    ax.text(37, 3, "Discerning Evaluator\n(High Awareness / Low Deference)", fontsize=9, color="#7f8c8d",
            style="italic")

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

def generate_report():
    """Loads model data, builds the dataframe, and generates all visual reports."""
    print(f"Loading evaluation results from: {REPO_PATH}...")

    models = {
        "DeepSeek V4 Flash": load_json(os.path.join(REPO_PATH, "results/baseline/deepseek_deepseek-v4-flash.json")),
        "Gemini 3 Flash": load_json(os.path.join(REPO_PATH, "results/baseline/google_gemini-3-flash-preview.json")),
        "Llama 4 Maverick": load_json(os.path.join(REPO_PATH, "results/baseline/meta-llama_llama-4-maverick.json")),
        "GPT 5.4 Mini": load_json(os.path.join(REPO_PATH, "results/baseline/openai_gpt-5.4-mini.json")),
        "Mistral 3.1 Medium": load_json(os.path.join(REPO_PATH, "results/baseline/mistralai_mistral-medium-3.1.json"))
    }

    print("Building strategy dataframe...")
    df_strategy = build_strategy_dataframe(models)

    print("Generating Visualizations...")
    plot_preference_bar_detailed(df_strategy)
    plot_awareness_bar_detailed(df_strategy)
    plot_strategy_quadrant_detailed(df_strategy)
    print("Report generation complete.")
