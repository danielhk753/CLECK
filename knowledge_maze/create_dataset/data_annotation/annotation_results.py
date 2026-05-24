"""
KnowledgeMaze Dataset Audit & Quality Manager
=============================================

This module serves as the final gateway for data quality control within the
KnowledgeMaze pipeline. It specializes in multi-source data merging and
high-level diagnostic visualization.

The module implements a strict priority-based hierarchy to resolve conflicts
between multiple human annotators and expert reviewers, ensuring the final
dataset represents the highest possible level of ground-truth accuracy.

Key Features:
-------------
1. **Hierarchical Merging Logic**:
   Implements a three-tier "Source of Truth" protocol to resolve ID collisions:
   - **Tier 1 (Resolution)**: Overrides from the expert_resolution file.
   - **Tier 2 (Expert)**: Direct audits from the expert annotator.
   - **Tier 3 (Consensus)**: Primary annotator data (fallback).

2. **Rule-Based Failure Analysis**:
   Maps internal validation codes (Rule 1-5) to human-readable dimensions
   of data quality, such as "Logical Framing," "Semantic Conflict," and
   "Reasoning Latency."

3. **Diagnostic Dashboarding**:
   Renders an integrated HTML and Matplotlib report containing:
   - **Metric Cards**: Real-time Pass/Fail rates and sample counts.
   - **Audit Donut**: A proportional breakdown of failure types to identify
     systemic weaknesses in the generation pipeline.
   - **Strategy Bar Chart**: Success rates segmented by prompting strategy
     (e.g., Frequency Mode, Top-K), identifying which strategies require
     further refinement.

Usage:
------
Initialize the `DatasetAuditManager` with paths to your annotation JSONs
and the resolution file. Call `plot_results()` to generate a visual audit
suitable for inclusion in technical reports or thesis appendices.
"""

import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML


class DatasetAuditManager:
    def __init__(self, annotator_paths, resolution_path):
        """
        Initializes the manager and merges the data.
        :param annotator_paths: List of paths to annotator/expert JSON files.
        :param resolution_path: Path to the expert_resolution.json file.
        """
        self.annotator_paths = annotator_paths
        self.resolution_path = resolution_path
        self.results = []
        self._load_and_merge()

    def _load_json(self, path):
        if not os.path.exists(path):
            print(f"⚠️ Warning: File not found {path}")
            return []
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_and_merge(self):
        # 1. Load the Resolution file (Highest Priority)
        res_data = {item['ID']: item for item in self._load_json(self.resolution_path)}

        # 2. Load the Expert file (Second Priority)
        expert_path = next((p for p in self.annotator_paths if 'expert.json' in p), None)
        expert_data = {item['ID']: item for item in self._load_json(expert_path)} if expert_path else {}

        # 3. Load all other annotators
        all_ann_data = {}
        for path in self.annotator_paths:
            if 'expert.json' not in path:
                data = self._load_json(path)
                for item in data:
                    item_id = item['ID']
                    if item_id not in all_ann_data:
                        all_ann_data[item_id] = []
                    all_ann_data[item_id].append(item)

        # Get the unique set of all IDs across all files
        all_ids = set(res_data.keys()) | set(expert_data.keys()) | set(all_ann_data.keys())

        merged_results = []
        for uid in all_ids:
            # HIERARCHY: Resolution > Expert > Consensus/First Annotator
            if uid in res_data:
                merged_results.append(res_data[uid])
            elif uid in expert_data:
                merged_results.append(expert_data[uid])
            elif uid in all_ann_data:
                # Fallback to the first available annotator if no expert/resolution exists
                merged_results.append(all_ann_data[uid][0])

        self.results = merged_results
        print(f"✅ Merged {len(self.results)} unique audited samples.")

    def plot_results(self):
        if not self.results:
            print("⚠️ No data tagged yet.")
            return

        # --- 1. Data Preparation ---
        df_tagged = pd.DataFrame(self.results)
        total_samples = len(df_tagged)
        total_passed = len(df_tagged[df_tagged['status'] == 'success'])
        total_failed = total_samples - total_passed
        pass_rate = (total_passed / total_samples * 100) if total_samples > 0 else 0

        # --- 2. HTML Metric Cards ---
        pass_color = "#27ae60" if pass_rate > 80 else "#f39c12" if pass_rate > 50 else "#e74c3c"
        html_cards = f"""
        <div style="display: flex; gap: 20px; font-family: 'Segoe UI', sans-serif; margin-bottom: 30px;">
            <div style="flex: 1; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid #3498db;">
                <div style="color: #7f8c8d; font-size: 0.9em; font-weight: bold; text-transform: uppercase;">Total Audited</div>
                <div style="font-size: 2.2em; font-weight: bold; color: #2c3e50;">{total_samples}</div>
            </div>
            <div style="flex: 1; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid {pass_color};">
                <div style="color: #7f8c8d; font-size: 0.9em; font-weight: bold; text-transform: uppercase;">Audit Pass Rate</div>
                <div style="font-size: 2.2em; font-weight: bold; color: {pass_color};">{pass_rate:.1f}%</div>
            </div>
            <div style="flex: 1; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid #e74c3c;">
                <div style="color: #7f8c8d; font-size: 0.9em; font-weight: bold; text-transform: uppercase;">Audit Failures</div>
                <div style="font-size: 2.2em; font-weight: bold; color: #c0392b;">{total_failed}</div>
            </div>
        </div>
        """
        display(HTML(html_cards))

        # --- 3. Graphs Configuration ---
        RULE_NAME_MAP = {
            "rule_1_logic": "Logical Framing",
            "rule_2_strategy": "Strategic Fidelity",
            "rule_3_conflict": "Semantic Conflict",
            "rule_4_purity": "Claim Purity",
            "rule_5_latency": "Reasoning Latency"
        }

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 15))
        sns.set_theme(style="whitegrid")

        # --- Chart 1: Donut Plot of Audit Failures (Disagreements) ---
        df_failed = df_tagged[df_tagged['status'] != 'success']

        if df_failed.empty:
            ax1.text(0.5, 0.5, "No Failures!\n100% Audit Pass Rate", ha='center', va='center', fontsize=16,
                     fontweight='bold', color='#27ae60')
            ax1.axis('off')
            ax1.set_title("Human Audit: Distribution of Failures", loc='left', fontsize=14, fontweight='bold', pad=15)
        else:
            failure_counts = df_failed['status'].value_counts()
            # Map raw status keys to readable names
            labels = [RULE_NAME_MAP.get(k, k) for k in failure_counts.index]
            sizes = failure_counts.values

            # Create Donut plot
            colors = sns.color_palette("flare", len(sizes))
            # We capture 'texts' (labels) and 'autotexts' (percentages)
            # Function to calculate and format: "COUNT (PERCENT%)"
            def make_autopct(values):
                def my_autopct(pct):
                    total = sum(values)
                    val = int(round(pct * total / 100.0))
                    return f'{val}\n({pct:.1f}%)'

                return my_autopct

            # Capture 'texts' (labels) and 'autotexts' (the numbers/percents)
            patches, texts, autotexts = ax1.pie(
                sizes, labels=labels,
                autopct=make_autopct(sizes),  # Uses our custom function
                startangle=140,
                colors=colors,
                wedgeprops=dict(width=0.4, edgecolor='w'),
                pctdistance=0.8,  # Adjusted slightly for the two-line text
                textprops={'fontsize': 12, 'fontweight': 'bold'}
            )

            # Set the inner text (Count + %) to white
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_horizontalalignment('center')

            # Set the category labels to black
            for text in texts:
                text.set_color('black')

            ax1.set_title(f"Human Audit: Distribution of Failures (Total: {len(df_failed)})",
                          loc='left', fontsize=14, fontweight='bold', pad=15)

        # Save only the donut plot in a pdf:
        if not df_failed.empty:
            # Create a separate, isolated figure context for the PDF export
            fig_donut, ax_donut = plt.subplots(figsize=(8, 8))

            # Recreate the pie chart components on the isolated axis
            patches_d, texts_d, autotexts_d = ax_donut.pie(
                sizes, labels=labels,
                autopct=make_autopct(sizes),
                startangle=140,
                colors=colors,
                wedgeprops=dict(width=0.4, edgecolor='w'),
                pctdistance=0.8,
                textprops={'fontsize': 12, 'fontweight': 'bold'}
            )

            for autotext in autotexts_d:
                autotext.set_color('white')
                autotext.set_horizontalalignment('center')
            for text in texts_d:
                text.set_color('black')

            # Save strictly this isolated figure layout to PDF
            fig_donut.savefig("audit_failures_donut.pdf", bbox_inches='tight', dpi=300)
            plt.close(fig_donut)  # Clean up the figure cache immediately

        # Chart 2: Success Rate by Strategy
        strat_stats = []
        unique_strats = df_tagged['strategy'].unique()
        for strat in unique_strats:
            df_s = df_tagged[df_tagged['strategy'] == strat]
            t_strat = len(df_s)
            p_strat = len(df_s[df_s['status'] == "success"])
            s_rate = (p_strat / t_strat * 100) if t_strat > 0 else 0
            strat_stats.append({"Strategy": strat, "Success Rate (%)": s_rate, "Passed": p_strat, "Total": t_strat})

        df_strat = pd.DataFrame(strat_stats)
        sns.barplot(data=df_strat, x="Strategy", y="Success Rate (%)", hue="Strategy", palette="viridis", ax=ax2,
                    legend=False)
        ax2.axhline(90, color='red', linestyle='--', alpha=0.3, label="Target (90%)")
        ax2.set_title("Human Audit: Success Rate per Strategy", loc='left', fontsize=14, fontweight='bold', pad=15)
        ax2.set_ylim(0, 115)

        for i, p in enumerate(ax2.patches):
            row = df_strat.iloc[i]
            # Changed format to X/Y (Z%)
            label = f"{int(row['Passed'])}/{int(row['Total'])}\n({row['Success Rate (%)']:.1f}%)"
            ax2.annotate(label, (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', fontsize=10, fontweight='bold', xytext=(0, 12),
                         textcoords='offset points')

        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout(pad=6.0)
        plt.show()
