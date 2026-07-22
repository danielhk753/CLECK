"""
Awareness Pivot Analysis and Visualization Framework.

This module quantifies and visualizes the operational behavioral shifts in Large
Language Models (LLMs) when confronting internal knowledge conflicts. Specifically,
it parses granular evaluation logs to calculate how a model's preference vectors
over authoritative claims versus evidence transition when conditioned on latent
epistemic awareness states.

Formally, the module aggregates and evaluates the preference distributions:
    $$\mathrm{PREF}(x \mid \mathtt{aw})$$
Where $x \in \{\text{claim}, \text{evidence}\}$, conditioned on whether the model exhibits
explicit dual awareness ($\mathtt{aw} = \text{True}$) versus unconditioned or unaware
baselines ($\mathtt{aw} = \text{False}$).

Key Features
------------
* Micro-metric Extraction: Parses multi-turn or iterative JSON evaluation structures
  to compute normalized class preferences cleanly isolated by awareness status.
* Publication-Ready Plotting: Generates clustered, stacked percentage bar charts
  optimized with custom geometry, strict y-axis formatting, and non-overlapping
  staggered labels to prevent layout crowding across multiple models.
* Unified Execution Pipeline: Exposes a standardized reporter interface to evaluate
  arbitrary model variants against localized repository data paths.
"""

import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from IPython.display import display

from knowledge_maze.tools.utils import load_json

load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')

class AwarenessPivotReporter:
    def __init__(self):
        self.repo_path = REPO_PATH

    def calculate_micro_metrics(self, data):
        ack_counter = Counter()
        noack_counter = Counter()

        for _, sample in data.items():
            judged = [r for r in sample.get("responses", []) if "judge_results" in r]
            for r in judged:
                cls = r["judge_results"].get("classification", "other")
                if cls not in {"claim", "evidence"}: continue

                if r["judge_results"].get("dual_awareness", False):
                    ack_counter[cls] += 1
                else:
                    noack_counter[cls] += 1

        def normalize(c):
            t = c["claim"] + c["evidence"]
            return {"claim": round(c["claim"] / t * 100, 2) if t > 0 else 0,
                    "evidence": round(c["evidence"] / t * 100, 2) if t > 0 else 0}

        ack = normalize(ack_counter)
        noack = normalize(noack_counter)
        return {"claim_ack": ack["claim"], "evidence_ack": ack["evidence"],
                "claim_noack": noack["claim"], "evidence_noack": noack["evidence"]}

    def plot_micro_stacked_bars(self, micro_results, figsize=(10, 8)):
        """
        Generates a grouped-stacked bar chart comparing Aware vs. Not Aware preferences.
        """
        col_names = {'Aware': 'True', 'Not Aware': 'False'}
        # 1. Transform dict to plotting DataFrame
        data = []
        for model, metrics in micro_results.items():
            # Aware Group (Conditioned)
            data.append({'Model': model, 'Group': 'Aware',
                         'Claim': metrics['claim_ack'], 'Evidence': metrics['evidence_ack']})
            # Not Aware Group (Unconditioned)
            data.append({'Model': model, 'Group': 'Not Aware',
                         'Claim': metrics['claim_noack'], 'Evidence': metrics['evidence_noack']})

        df = pd.DataFrame(data)

        # 2. Setup Plot
        models = df['Model'].unique()
        x = np.arange(len(models))
        width = 0.4  # Width of each bar

        fig, ax = plt.subplots(figsize=figsize)

        # Muted but saturated: A soft Navy/Charcoal and a warm, vibrant Ocher
        c_claim = '#708090'  # Slate Grey (Professional, grounded)
        c_evid = '#d4a373'  # Warm Sand/Ocher (Vibrant, warm, attention-grabbing)

        # 3. Plotting
        # We iterate through the groups to create the side-by-side clusters
        for i, group in enumerate(['Aware', 'Not Aware']):
            subset = df[df['Group'] == group]

            # Offset for the two bars (Aware vs Not Aware)
            offset = (i - 0.5) * width

            # Stacked Segments
            # Bottom: Claim
            ax.bar(x + offset, subset['Claim'], width, label=f'{group} - Claim' if i == 0 else "",
                   color=c_claim, alpha=0.9, edgecolor='white')

            # Top: Evidence
            ax.bar(x + offset, subset['Evidence'], width, bottom=subset['Claim'],
                   label=f'{group} - Evidence' if i == 0 else "",
                   color=c_evid, alpha=0.9, edgecolor='white')

            # Labels: Place at the junction (Top of Claim, Bottom of Evidence)
            for idx in range(len(x)):
                # Claim Label (Bottom part) - place just below the junction
                ax.text(x[idx] + offset, subset['Claim'].iloc[idx] - 2,
                        f"{subset['Claim'].iloc[idx]:.0f}%",
                        ha='center', va='top', color='white', fontweight='bold', fontsize=18)

                # Evidence Label (Top part) - place just above the junction
                ax.text(x[idx] + offset, subset['Claim'].iloc[idx] + 2,
                        f"{subset['Evidence'].iloc[idx]:.0f}%",
                        ha='center', va='bottom', color='white', fontweight='bold', fontsize=18)

                # Add "Aware" / "Not Aware" text below each bar
                # This iterates through every bar index to position the label correctly
                for idx in range(len(x)):
                    ax.text(x[idx] + offset, 104, col_names[group],
                            ha='center', va='top', fontsize=16,
                            fontweight='light')

        # 4. Styling

        # 1. Set the ticks
        ax.set_xticks(x)
        ax.set_xticklabels([])  # We will add custom labels below

        # 2. Add staggered labels
        for i, model in enumerate(models):
            # If the index is odd, push it further down by an extra 15 units
            y_offset = -15 if i % 2 != 0 else -5
            ax.text(x[i], y_offset, model, ha='center', va='top',
                    fontsize=20, fontweight='semibold', color='#1e293b')
        ax.set_ylabel(r'$\mathrm{PREF}(x \mid \mathtt{aw}) \ [\%]$', fontsize=20, fontweight='semibold')
        # ax.set_title("Operational Strategy: Preference Shift (Aware vs. Not Aware)",
        #              fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Custom Legend to clarify the stack and groups
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=c_claim, lw=0, marker='s', markersize=15, label=r'$x = \mathrm{claim}$'),
            Line2D([0], [0], color=c_evid, lw=0, marker='s', markersize=15, label=r'$x = \mathrm{evidence}$')
        ]

        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False, fontsize=20)

        plt.tight_layout()
        plt.savefig("model_preference_shift.pdf", bbox_inches='tight')
        plt.show()


def pivot_report():

    aware_pivot = AwarenessPivotReporter()

    model_files = {
        "DeepSeek V4 Flash": "results/baseline/deepseek_deepseek-v4-flash.json",
        "Gemini 3 Flash": "results/baseline/google_gemini-3-flash-preview.json",
        "Llama 4 Maverick": "results/baseline/meta-llama_llama-4-maverick.json",
        "GPT 5.4 Mini": "results/baseline/openai_gpt-5.4-mini.json",
        "Mistral 3.1 Medium": "results/baseline/mistralai_mistral-medium-3.1.json",
    }

    micro_results = {}

    for name, path in model_files.items():
        full_path = os.path.join(REPO_PATH, path)
        data = load_json(full_path)
        micro_results[name] = aware_pivot.calculate_micro_metrics(data)

    # Micro Table
    df_micro = pd.DataFrame([{
        "Model": k,
        "Claim (Ack|NoAck)": f'{v["claim_ack"]:.1f} | {v["claim_noack"]:.1f}',
        "Evidence (Ack|NoAck)": f'{v["evidence_ack"]:.1f} | {v["evidence_noack"]:.1f}'
    } for k, v in micro_results.items()])

    display(df_micro)
    aware_pivot.plot_micro_stacked_bars(micro_results)