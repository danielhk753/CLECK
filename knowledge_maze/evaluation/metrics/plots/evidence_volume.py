"""
KnowledgeMaze Evidence Scaling Analysis & Visualization
========================================================

This module analyzes how Large Language Model (LLM) performance, specifically
conflict awareness and resolution preferences, scales as the volume of evidence
increases within the context.

Core Methodology:
-----------------
1. **Evidence Load Mapping**: Intersects evaluation results with the foundational
   `knowledge_maze_questions.json` dataset to extract the count of evidence items
   (`Num_Evidences`) provided per sample.
2. **Strategy Filtering**: Explicitly skips instances using the
   `COMPARATIVE_DIFFERENCE` strategy across both data mapping and model iterations
   to maintain an unconfounded analysis baseline.
3. **Metric Aggregation**:
   - `Ack`: The spontaneous macro conflict acknowledgment rate ($P(\text{Aware})$).
   - `Evidence|Ack`: The conditional resolution preference ($P(\text{Evidence} \mid \text{Aware})$)
     measuring model adherence to logical evidence given awareness of a conflict.

Visualizations Generated:
-------------------------
- `evidence_volume_awareness.pdf`: Monotonic decay tracking awareness rates across model architectures.
- `evidence_volume_preference.pdf`: Resolution tracking showing conditional preference patterns
  rendered with a dual-row ($2 \times 3$ grid style) legend layout.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from collections import Counter
import matplotlib.pyplot as plt
from collections import defaultdict
from knowledge_maze.tools.utils import load_json

# =============================================================================
# CONFIGURATION
# =============================================================================
load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')

MODEL_PALETTE = {
    "DeepSeek V4 Flash": "#e74c3c",
    "Gemini 3 Flash": "#3498db",
    "Llama 4 Maverick": "#f39c12",
    "GPT 5.4 Mini": "#2ecc71",
    "Mistral 3.1 Medium": "#9b59b6"
}


def apply_modern_spines(ax):
    """Consistent styling helper."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')


# =============================================================================
# DATA LOADING & PROCESSING
# =============================================================================

def load_question_map():
    """
    Loads the reference mapping {ID: num_evidences}.
    Includes an exception rule where 'COMPARATIVE_DIFFERENCE' is always set to 2.
    """
    path = os.path.join(REPO_PATH, "data_files/knowledge_maze_questions.json")
    data = load_json(path)

    mapping = {}
    for item in data:
        # Check for the specific strategy exception
        if item.get("Selected strategy") == "COMPARATIVE_DIFFERENCE":
            continue
        else:
            # Default to the length of 'Fake data', safely handling missing keys
            count = len(item.get("Fake data", []))

        mapping[item['ID']] = count

    return mapping


def load_all_models():
    """Loads specific baseline models using the established environment."""
    model_paths = {
        "DeepSeek V4 Flash": "results/baseline/deepseek_deepseek-v4-flash.json",
        "Gemini 3 Flash": "results/baseline/google_gemini-3-flash-preview.json",
        "Llama 4 Maverick": "results/baseline/meta-llama_llama-4-maverick.json",
        "GPT 5.4 Mini": "results/baseline/openai_gpt-5.4-mini.json",
        "Mistral 3.1 Medium": "results/baseline/mistralai_mistral-medium-3.1.json"
    }
    return {
        name: load_json(os.path.join(REPO_PATH, path))
        for name, path in model_paths.items()
    }


def build_evidence_count_dataframe(models_dict, question_map):
    """Extracts metrics grouped by evidence count via ID cross-reference."""
    rows = []
    for model_name, data in models_dict.items():
        stats = defaultdict(lambda: {"ack_count": 0, "total_iters": 0, "evidence_ack": 0, "claim_ack": 0})

        for sample_id, sample in data.items():
            # SKIP if strategy is COMPARATIVE_DIFFERENCE
            if sample.get("strategy") == "COMPARATIVE_DIFFERENCE":
                continue

            num_ev = question_map.get(sample_id, 1)  # Default to 1
            responses = sample.get("responses", [])

            for r in responses:
                if "judge_results" not in r: continue
                jr = r["judge_results"]
                is_ack = jr.get("dual_awareness", False)
                classif = jr.get("classification", "other")

                stats[num_ev]["total_iters"] += 1
                if is_ack:
                    stats[num_ev]["ack_count"] += 1
                    if classif == "evidence":
                        stats[num_ev]["evidence_ack"] += 1
                    elif classif == "claim":
                        stats[num_ev]["claim_ack"] += 1

        for num_ev, s in stats.items():
            ack_pct = (100 * s["ack_count"] / s["total_iters"]) if s["total_iters"] else 0
            total_pref = s["evidence_ack"] + s["claim_ack"]
            pref_pct = (100 * s["evidence_ack"] / total_pref) if total_pref else 0

            rows.append({
                "Num_Evidences": int(num_ev),
                "Model": model_name,
                "Ack": ack_pct,
                "Evidence|Ack": pref_pct
            })
    return pd.DataFrame(rows)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_scaling_line(df, metric_col, title, ylabel, sample_counts):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Global Mean Trend
    mean_data = df.groupby("Num_Evidences")[metric_col].mean().sort_index()
    ax.plot(
        mean_data.index, mean_data.values,
        color="#5a6465", linewidth=10, alpha=0.4, zorder=1,
        solid_capstyle='round', label="Global Mean Trend"
    )

    # Individual Model Traces
    for model in df["Model"].unique():
        m_data = df[df["Model"] == model].sort_values("Num_Evidences")
        ax.plot(
            m_data["Num_Evidences"], m_data[metric_col],
            marker='o', markersize=6, linewidth=2, alpha=0.85,
            color=MODEL_PALETTE.get(model, "#bdc3c7"),
            label=model, zorder=3
        )

    # ----------------------------------------------

    ax.set_xlabel("Number of Evidences Provided", fontsize=20, fontweight="medium", labelpad=10)
    ax.set_ylabel(ylabel, fontsize=20, fontweight="medium", labelpad=10)
    # ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

    ax.set_ylim(0, 100)
    ax.grid(True, linestyle=":", alpha=0.3, zorder=0)
    apply_modern_spines(ax)

    if title == "evidence_volume_awareness.pdf":
        ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=20)
    else:
        ax.legend(
            loc="lower right",
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#e2e8f0",
            fontsize=15,
            ncol=2  # Spreads entries across 3 columns, resulting in 2 rows for 6 items
        )
    plt.tight_layout()
    plt.savefig(title, bbox_inches='tight')
    plt.show()


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def run_scaling_analysis():
    print("Loading datasets...")
    question_map = load_question_map()

    # Calculate sample counts once
    sample_counts = Counter(question_map.values())

    models = load_all_models()

    print("Building scaling dataframe...")
    df = build_evidence_count_dataframe(models, question_map)

    print("Generating scaling plots...")
    # Pass sample_counts to the functions
    plot_scaling_line(df, "Evidence|Ack", "evidence_volume_preference.pdf", r'$\mathrm{PREF}(evid \mid \mathtt{aw}) \ [\%]$',
                      sample_counts)
    plot_scaling_line(df, "Ack", "evidence_volume_awareness.pdf", "AWARE [%]", sample_counts)
