# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
#
# class KnowledgeMazeVisualizer:
#     def __init__(self, statistics_list):
#         """
#         Expects a list of dicts:
#         [{'Answer': 'Claim', 'Certainty': 'High', 'Strategy': 'Task', 'Num_evidence': 2}, ...]
#         """
#         self.df = pd.DataFrame(statistics_list)
#         # Standardize colors for the whole thesis
#         self.palette = {
#             'Claim': '#2980b9',  # Blue
#             'Evidence': '#c0392b',  # Red
#             'Memory': '#7f8c8d',  # Grey
#             'Other': '#f39c12'  # Orange
#         }
#         sns.set_theme(style="white")
#
#     def plot_task_vulnerability(self):
#         """Visualizes Claim vs Evidence reliance per reasoning task (Excludes Memory/Other)."""
#         plot_df = self.df[self.df['Answer'].isin(['Claim', 'Evidence'])]
#         counts = pd.crosstab(plot_df['Strategy'], plot_df['Answer'])
#
#         # Ensure columns exist
#         for col in ['Claim', 'Evidence']:
#             if col not in counts.columns: counts[col] = 0
#
#         totals = counts.sum(axis=1)
#         pct_df = counts.div(totals, axis=0) * 100
#         pct_df = pct_df.sort_values(by='Claim', ascending=True)
#
#         fig, ax = plt.subplots(figsize=(12, 8))
#         pct_df[['Claim', 'Evidence']].plot(kind='barh', stacked=True, ax=ax,
#                                            color=[self.palette['Claim'], self.palette['Evidence']], width=0.8)
#
#         for i, (name, row) in enumerate(pct_df.iterrows()):
#             ax.text(row['Claim'] - 1.5, i, f"{row['Claim']:.1f}%", va='center', ha='right', color='white',
#                     fontweight='bold')
#             ax.text(row['Claim'] + 1.5, i, f"{row['Evidence']:.1f}%", va='center', ha='left', color='white',
#                     fontweight='bold')
#             ax.text(102, i, f"n={int(totals.loc[name])}", va='center', color='#95a5a6', fontweight='bold')
#
#         plt.title('Reasoning Vulnerability Spectrum', fontsize=15, fontweight='bold', pad=25)
#         plt.xlim(0, 100)
#         ax.legend(['Claim (Framing)', 'Evidence (Fact)'], loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2,
#                   frameon=False)
#         sns.despine(left=True, bottom=True)
#         plt.show()
#
#     def plot_confidence_divergence(self):
#         """
#         Diverging bar chart based on Confidence levels.
#         - Claim (Framing) on the LEFT (Negative axis)
#         - Evidence (Fact) on the RIGHT (Positive axis)
#         """
#         # 1. Filter and Pivot
#         plot_df = self.df[self.df['Answer'].isin(['Claim', 'Evidence'])]
#         counts = pd.crosstab(plot_df['Certainty'], plot_df['Answer'])
#
#         # 2. Normalize and Sort (High at top for barh means it should be last in the index)
#         totals = counts.sum(axis=1)
#         pct_df = counts.div(totals, axis=0) * 100
#         # Reindexing so 'High' is the last index (top of the plot)
#         pct_df = pct_df.reindex(['Low', 'Medium', 'High']).dropna()
#
#         fig, ax = plt.subplots(figsize=(12, 6))
#         y_pos = np.arange(len(pct_df))
#
#         # 3. Plotting - Claim as negative (Left), Evidence as positive (Right)
#         ax.barh(y_pos, -pct_df['Claim'], color=self.palette['Claim'], height=0.6, label='Claim (Framing)')
#         ax.barh(y_pos, pct_df['Evidence'], color=self.palette['Evidence'], height=0.6, label='Evidence (Fact)')
#
#         # 4. Center Spine and Axis Formatting
#         plt.axvline(0, color='black', linewidth=2, zorder=3)
#         plt.xlim(-110, 110)
#
#         plt.yticks(y_pos, pct_df.index, fontweight='bold', fontsize=11)
#         ticks = np.arange(-100, 101, 25)
#         plt.xticks(ticks, [f"{abs(t)}%" for t in ticks], color='#7f8c8d')
#
#         # 5. Percentage Labels (Pinned to the divider line)
#         for i, (idx, row) in enumerate(pct_df.iterrows()):
#             # Claim Label (placed just left of 0)
#             if row['Claim'] > 0:
#                 ax.text(-2, i, f"{row['Claim']:.1f}%", va='center', ha='right',
#                         color='white', fontweight='bold', fontsize=10)
#             # Evidence Label (placed just right of 0)
#             if row['Evidence'] > 0:
#                 ax.text(2, i, f"{row['Evidence']:.1f}%", va='center', ha='left',
#                         color='white', fontweight='bold', fontsize=10)
#
#         # 6. Status Labels
#         plt.text(-50, len(pct_df) - 0.4, 'CLAIM DRIVEN', ha='center', fontweight='bold',
#                  color=self.palette['Claim'],
#                  bbox=dict(facecolor='white', edgecolor=self.palette['Claim'], boxstyle='round,pad=0.3'))
#         plt.text(50, len(pct_df) - 0.4, 'EVIDENCE DRIVEN', ha='center', fontweight='bold',
#                  color=self.palette['Evidence'],
#                  bbox=dict(facecolor='white', edgecolor=self.palette['Evidence'], boxstyle='round,pad=0.3'))
#
#         # 7. Legend and Titles
#         # Legend order now matches the visual: [Left Side, Right Side]
#         plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
#                    ncol=2, frameon=False, fontsize=11)
#
#         plt.title('Normalized Epistemic Shift by Confidence Level', fontsize=16, fontweight='bold', pad=40)
#         sns.despine(left=True, bottom=True)
#         plt.tight_layout()
#         plt.show()
#
#     def plot_evidence_count_impact(self):
#         """
#         Single-line trend showing Truth-Alignment.
#         100% = Pure Evidence Reliance | 0% = Pure Claim Reliance.
#         """
#         plot_df = self.df[self.df['Answer'].isin(['Claim', 'Evidence'])]
#         counts = pd.crosstab(plot_df['Num_evidence'], plot_df['Answer'])
#         pct_df = (counts.div(counts.sum(axis=1), axis=0) * 100)
#
#         plt.figure(figsize=(11, 6))
#
#         # Plot only the Evidence percentage as the "Truth-Alignment" line
#         plt.plot(pct_df.index, pct_df['Evidence'], marker='o', color=self.palette['Evidence'],
#                  linewidth=4, markersize=10, label='Evidence Reliance (Fact)')
#
#         # Fill the area below to show the "Claim" territory
#         plt.fill_between(pct_df.index, pct_df['Evidence'], color=self.palette['Claim'], alpha=0.1,
#                          label='Claim Territory')
#
#         # Add the 50% "Tipping Point" line
#         plt.axhline(50, color='#7f8c8d', linestyle='--', alpha=0.6, linewidth=1.5)
#         plt.text(pct_df.index.max(), 52, 'Logic Tipping Point (50%)', color='#7f8c8d', fontsize=9, ha='right')
#
#         plt.title('Evidence Volume vs. Model Grounding', fontsize=15, fontweight='bold', pad=20)
#         plt.xlabel('Number of Evidence Pieces Provided', fontweight='bold')
#         plt.ylabel('Evidence Reliance (%)', fontweight='bold')
#         plt.ylim(-5, 105)
#
#         # Customize Y-axis to show the 0-100 spectrum clearly
#         plt.yticks([0, 25, 50, 75, 100], ['0% (Pure Claim)', '25%', '50%', '75%', '100% (Pure Evidence)'])
#
#         plt.grid(axis='y', linestyle=':', alpha=0.5)
#         sns.despine()
#         plt.tight_layout()
#         plt.show()
#
#     def plot_overall_distribution(self):
#         """Pie chart showing the total distribution of all outcomes."""
#         data = self.df['Answer'].value_counts()
#         colors = [self.palette.get(x, '#bdc3c7') for x in data.index]
#
#         plt.figure(figsize=(8, 8))
#         plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140,
#                 colors=colors, wedgeprops={'edgecolor': 'white', 'linewidth': 2},
#                 textprops={'fontweight': 'bold'})
#
#         plt.title('Total Decision Distribution (Full Dataset)', fontsize=15, fontweight='bold')
#         plt.show()
import math
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec


class KnowledgeMazeVisualizer:
    def __init__(self, model_statistics_dict):
        """
        Expects a dict of model results:
        {
            'GPT-4': [{'Answer': 'Claim', 'Certainty': 'High', ...}, ...],
            'google/gemini-3-flash-preview': [{'Answer': 'Evidence', ...}, ...]
        }
        """
        # 1. Compile into a single DataFrame
        df_list = []
        for model_name, stats in model_statistics_dict.items():
            temp_df = pd.DataFrame(stats)
            temp_df['Model'] = model_name
            df_list.append(temp_df)

        self.df = pd.concat(df_list, ignore_index=True)
        self.models = list(model_statistics_dict.keys())

        # Helper: Wrap long model names so they don't overlap on plots
        # We replace '/' with '/ ' temporarily so textwrap knows it can break there
        self.wrapped_models = {
            m: textwrap.fill(m.replace('/', '/ '), width=16).replace('/ ', '/')
            for m in self.models
        }

        # 2. Setup Palettes
        self.ans_palette = {
            'Claim': '#2980b9',  # Blue
            'Evidence': '#c0392b',  # Red
            'Memory': '#7f8c8d',  # Grey
            'Other': '#f39c12'  # Orange
        }
        # Model palette for line charts
        model_colors = sns.color_palette("deep", len(self.models))
        self.model_palette = dict(zip(self.models, model_colors))

        sns.set_theme(style="white")

    def plot_overall_distribution(self):
        """100% Stacked Bar Chart comparing total distributions across models."""
        counts = pd.crosstab(self.df['Model'], self.df['Answer'], normalize='index') * 100

        # Map to wrapped names for the X-axis
        counts.index = [self.wrapped_models[m] for m in counts.index]

        order = ['Claim', 'Evidence', 'Memory', 'Other']
        cols = [c for c in order if c in counts.columns]
        counts = counts[cols]

        fig, ax = plt.subplots(figsize=(12, 7))
        colors = [self.ans_palette[c] for c in cols]

        counts.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.6)

        for c in ax.containers:
            labels = [f"{v.get_height():.1f}%" if v.get_height() > 2 else "" for v in c]
            ax.bar_label(c, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=10)

        plt.title('Overall Decision Distribution by Model', fontsize=15, fontweight='bold', pad=20)
        plt.xlabel('')
        plt.ylabel('Percentage of Answers (%)', fontweight='bold')
        plt.xticks(rotation=0, fontweight='bold', fontsize=10)
        plt.legend(title='', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)

        sns.despine(left=True)
        plt.tight_layout()
        plt.show()

    def plot_task_vulnerability(self):
        """Generates a separate reasoning vulnerability chart sequentially for each model."""
        for model in self.models:
            model_df = self.df[(self.df['Model'] == model) & (self.df['Answer'].isin(['Claim', 'Evidence']))]
            if model_df.empty: continue

            counts = pd.crosstab(model_df['Strategy'], model_df['Answer'])
            for col in ['Claim', 'Evidence']:
                if col not in counts.columns: counts[col] = 0

            totals = counts.sum(axis=1)
            pct_df = (counts.div(totals, axis=0) * 100).sort_values(by='Claim', ascending=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            pct_df[['Claim', 'Evidence']].plot(kind='barh', stacked=True, ax=ax,
                                               color=[self.ans_palette['Claim'], self.ans_palette['Evidence']],
                                               width=0.7)

            for i, (name, row) in enumerate(pct_df.iterrows()):
                if row['Claim'] > 0:
                    ax.text(row['Claim'] - 1.5, i, f"{row['Claim']:.1f}%", va='center', ha='right', color='white',
                            fontweight='bold')
                if row['Evidence'] > 0:
                    ax.text(row['Claim'] + 1.5, i, f"{row['Evidence']:.1f}%", va='center', ha='left', color='white',
                            fontweight='bold')

            plt.title(f'Reasoning Vulnerability: {model}', fontsize=14, fontweight='bold', pad=20)
            plt.xlim(0, 100)
            ax.legend(['Claim (Framing)', 'Evidence (Fact)'], loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=2,
                      frameon=False)
            sns.despine(left=True, bottom=True)
            plt.tight_layout()
            plt.show()

    def plot_merged_vulnerability_pro(self):
        """
        Paper-Level Layout:
        - Row 1: Aggregate Benchmark (Full width).
        - Rows 2+: 4 models per row.
        - Shared Legend: Only the leftmost plot in each row shows Strategy names.
        """
        # 1. Prepare Data
        df_clean = self.df[self.df['Answer'].isin(['Claim', 'Evidence'])].copy()

        overall_df = df_clean.copy()
        overall_df['Model'] = 'AGGREGATED BENCHMARK (All Models)'

        # Determine consistent strategy ordering (High Capture to Low Capture)
        overall_counts = pd.crosstab(overall_df['Strategy'], overall_df['Answer'])
        strategy_order = (overall_counts['Claim'] / overall_counts.sum(axis=1)).sort_values().index.tolist()

        # 2. Setup Grid
        n_models = len(self.models)
        n_cols = 4
        n_rows = 1 + math.ceil(n_models / n_cols)

        fig = plt.figure(figsize=(20, n_rows * 4.5))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)

        # 3. Top Row: Aggregated Benchmark
        ax_top = fig.add_subplot(gs[0, :])
        # Always show labels for the top benchmark
        self._draw_vulnerability_subplot(ax_top, overall_df, strategy_order, show_labels=True)
        ax_top.set_title("OVERALL SYSTEMIC VULNERABILITY (BENCHMARK)", fontsize=16, fontweight='bold', pad=20)

        # 4. Individual Model Rows
        for i, model in enumerate(self.models):
            row = (i // n_cols) + 1
            col = i % n_cols
            ax = fig.add_subplot(gs[row, col])

            model_df = df_clean[df_clean['Model'] == model]

            # CRITICAL: Only show strategy labels if it is the first column in the row
            show_labels = (col == 0)

            self._draw_vulnerability_subplot(ax, model_df, strategy_order, show_labels=show_labels)

            wrapped_name = self.wrapped_models.get(model, model)
            ax.set_title(wrapped_name, fontsize=12, fontweight='bold', color='#34495e', pad=10)

        # 5. Legend and Final Layout
        handles = [plt.Rectangle((0, 0), 1, 1, color=self.ans_palette['Claim']),
                   plt.Rectangle((0, 0), 1, 1, color=self.ans_palette['Evidence'])]
        fig.legend(handles, ['Claim (Framing Capture)', 'Evidence (Fact Grounding)'],
                   loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    def _draw_vulnerability_subplot(self, ax, data, strategy_order, show_labels=True):
        """Helper to draw bars and manage label visibility."""
        counts = pd.crosstab(data['Strategy'], data['Answer'])
        for col in ['Claim', 'Evidence']:
            if col not in counts.columns: counts[col] = 0

        counts = counts.reindex(strategy_order).fillna(0)
        totals = counts.sum(axis=1)
        pct_df = (counts.div(totals, axis=0) * 100)

        # Plot
        pct_df[['Claim', 'Evidence']].plot(
            kind='barh', stacked=True, ax=ax,
            color=[self.ans_palette['Claim'], self.ans_palette['Evidence']],
            width=0.7, legend=False
        )

        # Percentage Labels
        for j, (name, row) in enumerate(pct_df.iterrows()):
            # Only draw text if the bar segment is large enough
            if row['Claim'] > 8:
                ax.text(row['Claim'] - 2, j, f"{row['Claim']:.0f}%",
                        va='center', ha='right', color='white', fontweight='bold', fontsize=8)
            if row['Evidence'] > 8:
                ax.text(row['Claim'] + 2, j, f"{row['Evidence']:.0f}%",
                        va='center', ha='left', color='white', fontweight='bold', fontsize=8)

            # n-count always visible on the right edge
            ax.text(102, j, f"n={int(totals.iloc[j])}", va='center', color='#95a5a6', fontsize=7)

        ax.set_xlim(0, 100)
        ax.set_xlabel('')

        # This part handles the "legend on the side" logic
        if not show_labels:
            ax.set_yticklabels([])
            ax.set_ylabel('')
        else:
            ax.set_ylabel('Reasoning Strategy', fontweight='bold')
            plt.setp(ax.get_yticklabels(), fontweight='bold', fontsize=9)

        sns.despine(ax=ax, left=True, bottom=True)
        ax.set_xticks([])  # Keep the x-axis clean as percentages are inside

    def plot_confidence_heatmap(self, steps=20):
        # 1. Data Prep
        plot_df = self.df[self.df['Answer'].isin(['Claim', 'Evidence'])]

        # Calculate per-model percentages
        ct = pd.crosstab([plot_df['Model'], plot_df['Certainty']], plot_df['Answer'])
        pct_df = ct.div(ct.sum(axis=1), axis=0) * 100
        heatmap_data = pct_df['Claim'].unstack(level='Certainty').reindex(columns=['Low', 'Medium', 'High']).fillna(0)

        # Calculate SYSTEM TOTAL (weighted average)
        total_ct = pd.crosstab(plot_df['Certainty'], plot_df['Answer'])
        total_pct = (total_ct.div(total_ct.sum(axis=1), axis=0) * 100)['Claim']
        total_row = pd.DataFrame(total_pct).transpose()
        total_row.index = ['SYSTEM TOTAL']

        # Combine and wrap names
        heatmap_data.index = [self.wrapped_models[m] for m in heatmap_data.index]
        heatmap_data = pd.concat([heatmap_data, total_row])

        # 2. Colors & Bins
        base_cmap = mcolors.LinearSegmentedColormap.from_list("EpistemicShift", [
            self.ans_palette['Evidence'], "#f8f9f9", self.ans_palette['Claim']
        ])
        colors = base_cmap(np.linspace(0, 1, steps))
        cmap = mcolors.ListedColormap(colors)
        boundaries = np.linspace(0, 100, steps + 1)
        norm = mcolors.BoundaryNorm(boundaries, cmap.N)

        # 3. Plotting
        plt.figure(figsize=(12, len(heatmap_data) * 1.1 + 2))
        ax = sns.heatmap(
            heatmap_data, annot=True, fmt=".1f", cmap=cmap, norm=norm,
            linewidths=1, linecolor='#ecf0f1', cbar_kws={"shrink": .8},
            annot_kws={"weight": "bold", "size": 10}
        )

        # Highlight the Summary Row visually
        ax.add_patch(
            plt.Rectangle((0, len(heatmap_data) - 1), 3, 1, fill=False, edgecolor='black', lw=3, clip_on=False))

        # 4. Legend & Formatting
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([5, 95])
        cbar.set_ticklabels(['Evidence Reliance', 'Claim Reliance'], fontweight='bold')

        plt.title('Quantized Vulnerability Map with Systemic Aggregate', fontsize=16, fontweight='bold', pad=30)
        plt.yticks(rotation=0, fontweight='bold')
        plt.show()

    def plot_evidence_count_impact(self):
        plot_df = self.df[self.df['Answer'].isin(['Claim', 'Evidence'])]

        # 1. Data Prep
        counts = pd.crosstab([plot_df['Num_evidence'], plot_df['Model']], plot_df['Answer'])
        pct_df = (counts.div(counts.sum(axis=1), axis=0) * 100)
        evid_pct = pct_df['Evidence'].unstack(level='Model')

        plt.figure(figsize=(12, 7))

        # 2. Add THE AGGREGATE LINE (Modified for subtle visibility)
        total_counts = pd.crosstab(plot_df['Num_evidence'], plot_df['Answer'])
        total_evid_pct = (total_counts.div(total_counts.sum(axis=1), axis=0) * 100)['Evidence']

        # We use a thicker but lighter grey line with a higher transparency (alpha)
        # Putting it at zorder=1 keeps it behind the model-specific lines
        plt.plot(total_evid_pct.index, total_evid_pct.values, color='#2c3e50',
                 linewidth=6, alpha=0.15, label='SYSTEM AGGREGATE', zorder=1)

        # Also add a thin dashed core to the aggregate to keep it sharp
        plt.plot(total_evid_pct.index, total_evid_pct.values, color='#2c3e50',
                 linewidth=1, linestyle='--', alpha=0.4, zorder=2)

        # 3. Individual Model Lines (Higher alpha and zorder)
        for model in evid_pct.columns:
            plt.plot(evid_pct.index, evid_pct[model], marker='o', linewidth=2.5,
                     alpha=0.85, markersize=7, label=self.wrapped_models[model],
                     color=self.model_palette[model], zorder=5)

        # 4. Formatting
        plt.axhline(50, color='#bdc3c7', linestyle='-', alpha=0.5, linewidth=1, zorder=0)

        plt.title('Evidence Impact: Model Variance vs. Systemic Trend', fontsize=15, fontweight='bold', pad=20)
        plt.ylabel('Evidence Reliance (%)', fontweight='bold')
        plt.xlabel('Number of Evidence Pieces', fontweight='bold')
        plt.ylim(-5, 105)
        plt.xticks(evid_pct.index)  # Ensure x-axis only shows actual integer evidence counts

        # Custom Legend Handling
        handles, labels = plt.gca().get_legend_handles_labels()
        # Merge the two aggregate components into one legend handle
        order = [labels.index('SYSTEM AGGREGATE')] + [i for i, l in enumerate(labels) if l != 'SYSTEM AGGREGATE']
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                   bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)

        sns.despine()
        plt.tight_layout()
        plt.show()


class KnowledgeMazeTechnicVisualizer:
    def __init__(self, model_technique_dict):

        df_list = []
        for model_name, techniques in model_technique_dict.items():
            for tech_name, stats in techniques.items():
                temp_df = pd.DataFrame(stats)
                temp_df['Model'] = model_name
                temp_df['Prompt_Technique'] = tech_name
                df_list.append(temp_df)

        self.df = pd.concat(df_list, ignore_index=True)
        self.models = self.df['Model'].unique()
        self.techniques = self.df['Prompt_Technique'].unique()

        self.ans_palette = {
            'Claim': '#2980b9',
            'Evidence': '#c0392b',
            'Memory': '#7f8c8d',
            'Other': '#f39c12'
        }

        sns.set_theme(style="white")

    def plot_grouped_overall_distribution(self):
        """
        100% Stacked Bar Chart with custom spacing:
        Tight spacing between techniques, large spacing between model clusters.
        """
        # 1. Prepare Data
        counts = self.df.groupby(['Model', 'Prompt_Technique', 'Answer']).size().unstack(fill_value=0)
        order = ['Claim', 'Evidence', 'Memory', 'Other']
        for col in order:
            if col not in counts.columns: counts[col] = 0
        pct_df = (counts.div(counts.sum(axis=1), axis=0) * 100)[order]

        # 2. Define Spacing Logic & Exact Order
        models = self.df['Model'].unique()
        techs = self.df['Prompt_Technique'].unique()

        # --- CRITICAL FIX: Align DataFrame order to match the loop order ---
        # This prevents alphabetical sorting mismatches and fills in missing model-technique combos
        expected_index = pd.MultiIndex.from_product([models, techs], names=['Model', 'Prompt_Technique'])
        pct_df = pct_df.reindex(expected_index).fillna(0)
        # -------------------------------------------------------------------

        bar_width = 0.25
        intra_gap = 0.05  # Distance between bars in a cluster
        inter_gap = 0.2  # Distance between clusters

        # 3. Calculate Bar Positions
        bar_positions = []
        current_pos = 0
        cluster_centers = []

        for model in models:
            cluster_start = current_pos
            for tech in techs:
                bar_positions.append(current_pos)
                current_pos += bar_width + intra_gap

            # Calculate center for the Model label
            cluster_end = current_pos - (bar_width + intra_gap)
            cluster_centers.append((cluster_start + cluster_end) / 2)

            # Add the big gap before the next model
            current_pos += inter_gap

        # 4. Plotting
        fig, ax = plt.subplots(figsize=(16, 9))
        colors = [self.ans_palette[c] for c in order]

        # We plot each category layer by layer to handle manual x-positions
        bottoms = np.zeros(len(bar_positions))
        for i, col_name in enumerate(order):
            # Now pct_df[col_name].values perfectly matches the bar_positions list
            values = pct_df[col_name].values
            container = ax.bar(bar_positions, values, bottom=bottoms,
                               width=bar_width, color=colors[i], label=col_name)

            # Add percentage labels (might want to change v > 0 to v > 5 to avoid clutter on tiny bars)
            labels = [f'{v:.0f}%' if v > 0 else '' for v in values]
            ax.bar_label(container, labels=labels, label_type='center',
                         color='white', fontweight='bold', fontsize=9)
            bottoms += values

        # 5. Two-Tier X-Axis Labels
        ax.set_xticks(bar_positions)
        ax.set_xticklabels([t for m in models for t in techs], fontsize=10, fontweight='bold')

        for i, model in enumerate(models):
            ax.annotate(model,
                        xy=(cluster_centers[i], 0),
                        xycoords=('data', 'axes fraction'),
                        xytext=(0, -40),
                        textcoords='offset points',
                        va='top', ha='center',
                        fontsize=14, fontweight='bold', color='#2c3e50')

        # 6. Styling
        plt.title('Knowledge Conflict Distribution by Prompting Strategy',
                  fontsize=20, fontweight='bold', pad=40)
        plt.ylabel('Percentage of Total Answers (%)', fontweight='bold', fontsize=12)
        plt.legend(title='Answer Type', bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)

        plt.ylim(0, 100)
        sns.despine(left=True, bottom=False)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.show()

    def plot_strategy_by_prompt_type(self):
        """One row of 3 grids: Strategies on Y, Claim/Evidence on X. Mean of all models."""
        # 1. Filter data
        df_clean = self.df[self.df['Answer'].isin(['Claim', 'Evidence'])].copy()

        if df_clean.empty:
            print("Warning: No 'Claim' or 'Evidence' data found to plot.")
            return

        # 2. Determine Strategy Order (Mean Claim Reliance across all models/techs)
        overall_counts = pd.crosstab(df_clean['Strategy'], df_clean['Answer'])
        for col in ['Claim', 'Evidence']:
            if col not in overall_counts.columns: overall_counts[col] = 0

        # Sort strategies so the most 'vulnerable' (highest Claim %) are at the top
        strategy_order = (overall_counts['Claim'] / overall_counts.sum(axis=1)).sort_values().index.tolist()

        # Wrap long strategy names so they don't overlap
        wrapped_strategies = [textwrap.fill(s, width=25) for s in strategy_order]

        # 3. Setup Figure
        n_techs = len(self.techniques)
        fig, axes = plt.subplots(1, n_techs, figsize=(22, 8), sharey=True)
        if n_techs == 1: axes = [axes]

        for i, tech in enumerate(self.techniques):
            ax = axes[i]
            tech_df = df_clean[df_clean['Prompt_Technique'] == tech]

            # Calculate percentages for THIS technique
            counts = pd.crosstab(tech_df['Strategy'], tech_df['Answer'])
            for col in ['Claim', 'Evidence']:
                if col not in counts.columns: counts[col] = 0

            # Reindex to ensure all grids use the same strategy order
            pct_df = (counts.div(counts.sum(axis=1).replace(0, 1), axis=0) * 100).reindex(strategy_order).fillna(0)

            # Plot
            pct_df[['Claim', 'Evidence']].plot(
                kind='barh', stacked=True, ax=ax,
                color=[self.ans_palette['Claim'], self.ans_palette['Evidence']],
                width=0.75, legend=False
            )

            # 4. INTERNAL PERCENTAGE LABELS
            for j, (name, row) in enumerate(pct_df.iterrows()):
                if row['Claim'] > 6:
                    ax.text(row['Claim'] - 2, j, f"{row['Claim']:.0f}%",
                            va='center', ha='right', color='white', fontweight='bold', fontsize=9)
                if row['Evidence'] > 6:
                    ax.text(row['Claim'] + 2, j, f"{row['Evidence']:.0f}%",
                            va='center', ha='left', color='white', fontweight='bold', fontsize=9)

            # 5. STRATEGY NAMES (The "Left Legend")
            # We only show these on the very first subplot (index 0)
            if i == 0:
                ax.set_yticks(range(len(strategy_order)))
                ax.set_yticklabels(wrapped_strategies, fontsize=11, fontweight='bold', color='#2c3e50')
                ax.set_ylabel('Reasoning Strategy', fontweight='bold', fontsize=14, labelpad=20)
            else:
                # Explicitly clear ticks for shared-axis subplots to avoid ghosting
                ax.yaxis.set_tick_params(labelleft=False)

            # Formatting
            ax.set_title(f"TECHNIQUE: {tech.upper()}", fontsize=15, fontweight='bold', pad=15)
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 50, 100])
            ax.set_xticklabels(['0%', '50%', '100%'], fontsize=10)
            sns.despine(ax=ax, left=True, bottom=True)

        # 6. SHARED LEGEND (Claim vs Evidence)
        handles = [plt.Rectangle((0, 0), 1, 1, color=self.ans_palette['Claim']),
                   plt.Rectangle((0, 0), 1, 1, color=self.ans_palette['Evidence'])]
        fig.legend(handles, ['Claim (Framing Capture)', 'Evidence (Fact Grounding)'],
                   loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=13)

        plt.suptitle("STRATEGY VULNERABILITY: SYSTEMIC BIAS PER PROMPT TECHNIQUE",
                     fontsize=20, fontweight='bold', y=1.05)

        # CRITICAL: Adjust the left margin so the strategy names aren't cut off
        plt.subplots_adjust(left=0.18, bottom=0.15)
        plt.show()

    def _draw_vulnerability_subplot(self, ax, data, strategy_order, show_labels=True):
        """Internal helper for horizontal stacked bars with y-labels and percentages."""
        # Calculate percentages for this specific technique
        counts = pd.crosstab(data['Strategy'], data['Answer'])
        for col in ['Claim', 'Evidence']:
            if col not in counts.columns: counts[col] = 0

        counts = counts.reindex(strategy_order).fillna(0)
        pct_df = (counts.div(counts.sum(axis=1).replace(0, 1), axis=0) * 100)

        # Draw the stacked bars
        pct_df[['Claim', 'Evidence']].plot(
            kind='barh', stacked=True, ax=ax,
            color=[self.ans_palette['Claim'], self.ans_palette['Evidence']],
            width=0.75, legend=False
        )

        # 4. Percentage Text Annotations
        for j, (name, row) in enumerate(pct_df.iterrows()):
            # Label for Claim (on the left side of the stack)
            if row['Claim'] > 5:
                ax.text(row['Claim'] - 1.5, j, f"{row['Claim']:.0f}%",
                        va='center', ha='right', color='white', fontweight='bold', fontsize=10)
            # Label for Evidence (on the right side of the stack)
            if row['Evidence'] > 5:
                ax.text(row['Claim'] + 1.5, j, f"{row['Evidence']:.0f}%",
                        va='center', ha='left', color='white', fontweight='bold', fontsize=10)

        # 5. Y-Axis Strategy Labels (The "Left Legend")
        if show_labels:
            ax.set_yticks(range(len(strategy_order)))
            ax.set_yticklabels(strategy_order, fontsize=11, fontweight='bold', color='#2c3e50')
            ax.set_ylabel('Reasoning Strategy', fontweight='bold', fontsize=12, labelpad=10)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel('')

        # Formatting
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(['0%', '50%', '100%'], fontsize=10)
        sns.despine(ax=ax, left=True, bottom=True)

    def plot_confidence_heatmap(self, steps=20):
        """
        Quantized Heatmap:
        Y-axis = Prompting Techniques (Aggregated mean across all models).
        X-axis = Model Certainty levels.
        """
        # 1. Data Prep: Filter only for binary conflict answers
        plot_df = self.df[self.df['Answer'].isin(['Claim', 'Evidence'])]

        # Calculate percentages per Technique and Certainty level
        # This automatically averages across all models because 'Model' isn't in the grouping
        ct = pd.crosstab([plot_df['Prompt_Technique'], plot_df['Certainty']], plot_df['Answer'])
        pct_df = ct.div(ct.sum(axis=1), axis=0) * 100

        # Pivot to heatmap format and ensure logical order of certainty
        heatmap_data = pct_df['Claim'].unstack(level='Certainty').reindex(
            columns=['Low', 'Medium', 'High']
        ).fillna(0)

        # 2. Colors & Bins (Maintaining your original quantized look)
        base_cmap = mcolors.LinearSegmentedColormap.from_list("EpistemicShift", [
            self.ans_palette['Evidence'], "#f8f9f9", self.ans_palette['Claim']
        ])

        # Create the stepped/quantized effect
        colors = base_cmap(np.linspace(0, 1, steps))
        cmap = mcolors.ListedColormap(colors)
        boundaries = np.linspace(0, 100, steps + 1)
        norm = mcolors.BoundaryNorm(boundaries, cmap.N)

        # 3. Plotting
        plt.figure(figsize=(10, len(heatmap_data) * 1.2 + 2))
        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".1f",
            cmap=cmap,
            norm=norm,
            linewidths=1.5,
            linecolor='#ecf0f1',
            cbar_kws={"shrink": .8},
            annot_kws={"weight": "bold", "size": 11}
        )

        # 4. Legend & Formatting
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([5, 95])
        cbar.set_ticklabels(['Evidence Reliance', 'Claim Reliance'], fontweight='bold')

        plt.title('Vulnerability by Technique vs. Model Certainty',
                  fontsize=16, fontweight='bold', pad=30)
        plt.ylabel('Prompting Technique', fontweight='bold', fontsize=12)
        plt.xlabel('Reported Model Certainty', fontweight='bold', fontsize=12)
        plt.yticks(rotation=0, fontweight='bold')
        plt.xticks(fontweight='bold')

        plt.tight_layout()
        plt.show()
