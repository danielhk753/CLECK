"""
Model Evaluation Reporting Module
=================================
This module provides the 'display_report' function and 'ReportGenerator' class
to visualize model performance across strategies, taxonomy categories,
and metrics (conditioned evidence adherence, awareness, and stable preferences).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv.main import load_dotenv
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from knowledge_maze.tools.utils import load_json

load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')

class ReportGenerator:
    """
    A utility to parse model evaluation results and generate comparative analytical plots.

    This module processes raw model baseline test results (JSON format) and computes
    key performance metrics including conditioned preference, awareness rates, and
    stable unconditioned preferences. It provides a unified visual architecture
    across multiple plots using faceted horizontal bar charts.

    Attributes:
        test_results_baseline (dict): Raw nested dictionary containing model responses.
        df (pd.DataFrame): The parsed and computed dataset used for visualization.

    Methods:
        display_strategy_report(output_dir): Entry point to generate and save three primary analytical plots to PDF.
    """
    def __init__(self, test_results_baseline):
        self.test_results_baseline = test_results_baseline
        self.df = self._parse_data()

        self.strategy_display_names = {
            'Extreme Numerical': 'Extreme Num.',
            'Top K Ranking': 'Top K',
            'Extreme Temporal': 'Extreme Temp.',
            'Frequency Mode': 'Freq. Mode',
            'Unique Diversity': 'Unique Div.',
            'Filter Count Numerical': 'Filter Num.',
            'Filter Count Temporal': 'Filter Temp.',
            'Categorical Filter Lookup': 'Filter Cat.',
            'Comparative Difference': 'Comp. Diff.',
            'Aggregation': 'Aggregation',
            'Count Total': 'Count Total'
        }

        self.taxonomy_map = {
            'Extreme Numerical': 'Extreme Ops',
            'Top K Ranking': 'Extreme Ops',
            'Extreme Temporal': 'Extreme Ops',
            'Frequency Mode': 'Distributions',
            'Unique Diversity': 'Distributions',
            'Filter Count Numerical': 'Filters Ops',
            'Filter Count Temporal': 'Filters Ops',
            'Categorical Filter Lookup': 'Filters Ops',
            'Comparative Difference': 'Ungrouped',
            'Aggregation': 'Ungrouped',
            'Count Total': 'Ungrouped',
        }

        self.taxonomy_colors = {
            'Extreme Ops': '#2ecc71',
            'Distributions': '#9b59b6',
            'Filters Ops': '#e74c3c',
            'Ungrouped': '#95a5a6',
        }

        self.taxonomy_order = ['Extreme Ops', 'Distributions', 'Filters Ops', 'Ungrouped']
        self.models = ["Gemini 3 Flash", "Mistral 3.1 Medium", "Llama 4 Maverick", "GPT 5.4 Mini", "DeepSeek V4 Flash"]

        self.short_names = {
            "Gemini 3 Flash": "Gemini 3\nFlash",
            "Mistral 3.1 Medium": "Mistral 3.1\nMedium",
            "Llama 4 Maverick": "Llama 4\nMaverick",
            "GPT 5.4 Mini": "GPT 5.4\nMini",
            "DeepSeek V4 Flash": "DeepSeek V4\nFlash",
        }

        # Determine plotting order (grouped by taxonomy, bottom to top)
        self.df['Taxonomy'] = self.df['Strategy'].map(self.taxonomy_map)
        ordered_strategies = []
        for tax in self.taxonomy_order:
            members = [s for s, t in self.taxonomy_map.items() if t == tax]
            ordered_strategies.extend(members)

        self.plot_order = list(reversed(ordered_strategies))
        self.n_strats = len(self.plot_order)

        # Calculate boundaries for drawing separator lines
        self.boundaries = []
        pos = 0
        for tax in reversed(self.taxonomy_order):
            members = [s for s, t in self.taxonomy_map.items() if t == tax]
            pos += len(members)
            self.boundaries.append(pos)
        self.boundaries = self.boundaries[:-1]

        # Font standards
        self.FONTS = {'title': 22, 'label': 22, 'tick': 22, 'ytick': 22, 'yytick': 18}

    def _parse_data(self):
        """Parses the raw JSON results into a unified DataFrame with all required metrics."""
        records = []
        for model_name, questions_dict in self.test_results_baseline.items():
            for q_id, q_data in questions_dict.items():
                strategy = q_data.get('strategy', 'UNKNOWN')
                strategy_clean = strategy.replace('_', ' ').title()
                responses = q_data.get('responses', [])

                valid_resps, aware_count, ev_aware_count = 0, 0, 0
                ev_count, claim_count = 0, 0

                for r in responses:
                    judge = r.get('judge_results', {})
                    if not judge: continue
                    valid_resps += 1

                    cls = judge.get('classification', '').lower()
                    is_aware = judge.get('dual_awareness', False) or cls in ['mix', 'mixed', 'both']

                    if is_aware:
                        aware_count += 1
                        if cls == 'evidence':
                            ev_aware_count += 1

                    if cls == 'evidence': ev_count += 1
                    if cls == 'claim': claim_count += 1

                if valid_resps == 0: continue

                # Metric 1: Conditioned Preference (Evidence Adherence)
                ev_given_aware = (ev_aware_count / aware_count * 100) if aware_count > 0 else np.nan

                # Metric 2: Awareness Rate
                p_aware = (aware_count / valid_resps) * 100

                # Metric 3: Stable Unconditioned Preference (>= 4/5 logic equates to >= 0.8 proportion)
                if ev_count / valid_resps >= 0.8:
                    stable = 'Evidence'
                elif claim_count / valid_resps >= 0.8:
                    stable = 'Claim'
                else:
                    stable = 'Mixed'

                records.append({
                    'Model': model_name,
                    'Q_ID': q_id,
                    'Strategy': strategy_clean,
                    'Ev_Given_Aware': ev_given_aware,
                    'P_Aware': p_aware,
                    'Stable': stable
                })
        return pd.DataFrame(records)

    def _style_panel(self, ax, title, show_yticks, is_mean):
        """Applies the common visual architecture (backgrounds, lines, taxonomy sidebar)."""
        if is_mean:
            ax.set_facecolor('#f8fafc')
            for i in range(self.n_strats):
                color = '#e2e8f0' if i % 2 == 0 else '#f8fafc'
                ax.axhspan(i - 0.5, i + 0.5, color=color, zorder=0)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#64748b')
                spine.set_linewidth(2)
        else:
            ax.set_facecolor('white')
            for i in range(self.n_strats):
                color = '#f8fafc' if i % 2 == 0 else 'white'
                ax.axhspan(i - 0.5, i + 0.5, color=color, zorder=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cbd5e1')
            ax.spines['bottom'].set_color('#cbd5e1')

        # Dividers & 50% line
        for b in self.boundaries:
            ax.axhline(y=b - 0.5, color='#aaaaaa', linewidth=0.8, linestyle='-', zorder=2)
        ax.axvline(x=50, color='#94a3b8', linestyle='--', linewidth=1.2, alpha=0.8, zorder=1)

        # X-Axis Layout
        ax.set_xlim(-3, 105)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(['0', '50', '100'], fontsize=self.FONTS['tick'], color='#334155')
        ax.grid(axis='x', linestyle=':', alpha=0.35, zorder=0)

        # Y-Axis Layout
        ax.set_ylim(-0.5, self.n_strats - 0.5)
        ax.tick_params(axis='y', length=0)

        if show_yticks:
            ax.set_yticks(range(self.n_strats))
            display_labels = [self.strategy_display_names.get(s, s) for s in self.plot_order]
            ax.set_yticklabels(display_labels, fontsize=self.FONTS['ytick'], color='#1e293b', fontweight='semibold')
            ax.tick_params(axis='y', pad=10)

            # Sidebar logic
            start_y = -0.5
            trans = ax.get_yaxis_transform()
            for tax in reversed(self.taxonomy_order):
                members = [s for s, t in self.taxonomy_map.items() if t == tax]
                count = len(members)
                end_y = start_y + count
                mid_y = (start_y + end_y) / 2

                ax.plot([-0.90, -0.90], [start_y + 0.1, end_y - 0.1], transform=trans,
                        color=self.taxonomy_colors[tax], lw=6, clip_on=False)

                ax.text(-0.95, mid_y, tax, transform=trans,
                        rotation=90, ha='right', va='center',
                        fontsize=self.FONTS['yytick'], fontweight='bold', color=self.taxonomy_colors[tax])
                start_y = end_y
        else:
            ax.set_yticks(range(self.n_strats))
            ax.set_yticklabels([])

        ax.set_title(title, fontsize=self.FONTS['title'], fontweight='bold', pad=12, color='#0f172a')

    def _plot_standard(self, metric_col, x_label, output_path):
        """Generates the standard non-stacked faceted bar charts (Plots 1 & 2)"""
        n_models = len(self.models)
        fig = plt.figure(figsize=(26, 9), facecolor='white')
        gs = GridSpec(1, n_models + 1, figure=fig, width_ratios=[1] * n_models + [1.15], wspace=0.08)

        bar_colors = [self.taxonomy_colors[self.taxonomy_map[s]] for s in self.plot_order]
        y_pos = range(self.n_strats)

        for idx, model in enumerate(self.models):
            ax = fig.add_subplot(gs[0, idx])
            self._style_panel(ax, self.short_names[model], show_yticks=(idx == 0), is_mean=False)

            model_df = self.df[self.df['Model'] == model]
            means = [model_df[model_df['Strategy'] == s][metric_col].mean() for s in self.plot_order]
            means = [v if not np.isnan(v) else 0 for v in means]

            ax.barh(y_pos, means, height=0.6, color=bar_colors, edgecolor='#2c3e50', linewidth=0.7, zorder=3,
                    alpha=0.92)

        # Mean Panel
        ax_mean = fig.add_subplot(gs[0, n_models])
        self._style_panel(ax_mean, "Mean ± SD\n(All Models)", show_yticks=False, is_mean=True)

        means, stds = [], []
        for s in self.plot_order:
            per_model = [self.df[(self.df['Strategy'] == s) & (self.df['Model'] == m)][metric_col].mean() for m in
                         self.models]
            per_model = [v for v in per_model if not np.isnan(v)]
            means.append(np.mean(per_model) if per_model else 0)
            stds.append(np.std(per_model) if len(per_model) > 1 else 0)

        ax_mean.barh(y_pos, means, height=0.6, color=bar_colors, edgecolor='#2c3e50', linewidth=0.7, zorder=3,
                     alpha=0.92)
        ax_mean.errorbar(means, y_pos, xerr=stds, fmt='none', color='#2c3e50', capsize=3, linewidth=1.2, zorder=4)

        # Labels & Saving
        fig.text(0.55, 0.01, x_label, ha='center', va='center', fontsize=self.FONTS['label'], color='#1e293b',
                 fontweight='bold')
        fig.subplots_adjust(left=0.25, right=0.98, bottom=0.10, top=0.92)

        if output_path:
            plt.savefig(output_path, bbox_inches="tight")
            print(f"Saved {x_label} -> {output_path}")

        plt.show()

    def _plot_stacked(self, output_path):
        """Generates the stacked bar chart for Stable Preference (Plot 3)"""
        n_models = len(self.models)
        fig = plt.figure(figsize=(26, 9), facecolor='white')
        gs = GridSpec(1, n_models + 1, figure=fig, width_ratios=[1] * n_models + [1.15], wspace=0.08)

        # Stacked categories and colors
        stack_order = ['Evidence', 'Mixed', 'Claim']
        stack_colors = {
            'Evidence': '#94a3b8',  # slate-400
            'Mixed': '#64748b',  # slate-500
            'Claim': '#cbd5e1'  # slate-300
        }
        y_pos = range(self.n_strats)

        # Helper to calculate proportions for stacked bars
        def get_props(sub_df):
            total = len(sub_df)
            if total == 0: return {c: 0 for c in stack_order}
            counts = sub_df['Stable'].value_counts()
            return {c: (counts.get(c, 0) / total * 100) for c in stack_order}

        all_models_props = {s: {m: {} for m in self.models} for s in self.plot_order}

        for idx, model in enumerate(self.models):
            ax = fig.add_subplot(gs[0, idx])
            self._style_panel(ax, self.short_names[model], show_yticks=(idx == 0), is_mean=False)

            model_df = self.df[self.df['Model'] == model]

            left_arr = np.zeros(self.n_strats)
            for cat in stack_order:
                cat_vals = []
                for s_idx, s in enumerate(self.plot_order):
                    strat_df = model_df[model_df['Strategy'] == s]
                    props = get_props(strat_df)
                    val = props[cat]
                    all_models_props[s][model][cat] = val
                    cat_vals.append(val)

                ax.barh(y_pos, cat_vals, left=left_arr, height=0.6, color=stack_colors[cat], edgecolor='white',
                        linewidth=0.5, zorder=3, alpha=0.95)
                left_arr += np.array(cat_vals)

        # Mean Panel (Average of the proportions across models)
        ax_mean = fig.add_subplot(gs[0, n_models])
        self._style_panel(ax_mean, "Mean Proportions\n(All Models)", show_yticks=False, is_mean=True)

        left_arr = np.zeros(self.n_strats)
        for cat in stack_order:
            mean_cat_vals = []
            for s in self.plot_order:
                # Average this category's percentage across all models for strategy 's'
                model_vals = [all_models_props[s][m][cat] for m in self.models]
                mean_cat_vals.append(np.mean(model_vals))

            ax_mean.barh(y_pos, mean_cat_vals, left=left_arr, height=0.6, color=stack_colors[cat], edgecolor='white',
                         linewidth=0.5, zorder=3, alpha=0.95)
            left_arr += np.array(mean_cat_vals)

        # Labels & Custom Stacked Legend
        fig.text(0.55, 0.05, r"$P(c),\; c \in \{\mathrm{Evidence},\mathrm{Mixed},\mathrm{Claim}\}$", ha='center',
                 va='center', fontsize=self.FONTS['label'], color='#1e293b', fontweight='bold')

        # Add legend mapping for the Stacked bar colors
        legend_elements = [mpatches.Patch(facecolor=stack_colors[cat], edgecolor='white', label=f"Stable {cat}") for cat
                           in stack_order]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=16, frameon=False,
                   bbox_to_anchor=(0.55, -0.02))

        fig.subplots_adjust(left=0.25, right=0.98, bottom=0.14, top=0.92)  # Extra bottom margin for legend

        if output_path:
            plt.savefig(output_path, bbox_inches="tight")
            print(f"Saved Stacked Preference -> {output_path}")

        plt.show()


def display_strategy_report():
    """
    Parses baseline results and generates three faceted comparative reports.
    """

    print(f"Loading evaluation results from: {REPO_PATH}...")

    test_results_baseline = {
        "DeepSeek V4 Flash": load_json(os.path.join(REPO_PATH, "results/baseline/deepseek_deepseek-v4-flash.json")),
        "Gemini 3 Flash": load_json(os.path.join(REPO_PATH, "results/baseline/google_gemini-3-flash-preview.json")),
        "Llama 4 Maverick": load_json(os.path.join(REPO_PATH, "results/baseline/meta-llama_llama-4-maverick.json")),
        "GPT 5.4 Mini": load_json(os.path.join(REPO_PATH, "results/baseline/openai_gpt-5.4-mini.json")),
        "Mistral 3.1 Medium": load_json(os.path.join(REPO_PATH, "results/baseline/mistralai_mistral-medium-3.1.json"))
    }

    # Initialize the generator to parse the data once
    report = ReportGenerator(test_results_baseline)

    # Plot 1: P(Evidence | Awareness)
    report._plot_standard(
        metric_col='Ev_Given_Aware',
        x_label='P(Evidence | Awareness) [%]',
        output_path="strategy_conditioned_preference.pdf"
    )

    # Plot 2: Awareness Rate
    report._plot_standard(
        metric_col='P_Aware',
        x_label='Awareness Rate [%]',
        output_path="strategy_awareness_rate.pdf"
    )

    # Plot 3: Stable Unconditioned Preference (Stacked)
    report._plot_stacked(
        output_path="strategy_stable_preference.pdf"
    )
