import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


class KnowledgeMazeVisualizer:
    def __init__(self, statistics_list):
        """
        Expects a list of dicts:
        [{'Answer': 'Claim', 'Certainty': 'High', 'Strategy': 'Task', 'Num_evidence': 2}, ...]
        """
        self.df = pd.DataFrame(statistics_list)
        # Standardize colors for the whole thesis
        self.palette = {
            'Claim': '#2980b9',  # Blue
            'Evidence': '#c0392b',  # Red
            'Memory': '#7f8c8d',  # Grey
            'Other': '#f39c12'  # Orange
        }
        sns.set_theme(style="white")

    def plot_task_vulnerability(self):
        """Visualizes Claim vs Evidence reliance per reasoning task (Excludes Memory/Other)."""
        plot_df = self.df[self.df['Answer'].isin(['Claim', 'Evidence'])]
        counts = pd.crosstab(plot_df['Strategy'], plot_df['Answer'])

        # Ensure columns exist
        for col in ['Claim', 'Evidence']:
            if col not in counts.columns: counts[col] = 0

        totals = counts.sum(axis=1)
        pct_df = counts.div(totals, axis=0) * 100
        pct_df = pct_df.sort_values(by='Claim', ascending=True)

        fig, ax = plt.subplots(figsize=(12, 8))
        pct_df[['Claim', 'Evidence']].plot(kind='barh', stacked=True, ax=ax,
                                           color=[self.palette['Claim'], self.palette['Evidence']], width=0.8)

        for i, (name, row) in enumerate(pct_df.iterrows()):
            ax.text(row['Claim'] - 1.5, i, f"{row['Claim']:.1f}%", va='center', ha='right', color='white',
                    fontweight='bold')
            ax.text(row['Claim'] + 1.5, i, f"{row['Evidence']:.1f}%", va='center', ha='left', color='white',
                    fontweight='bold')
            ax.text(102, i, f"n={int(totals.loc[name])}", va='center', color='#95a5a6', fontweight='bold')

        plt.title('Reasoning Vulnerability Spectrum', fontsize=15, fontweight='bold', pad=25)
        plt.xlim(0, 100)
        ax.legend(['Claim (Framing)', 'Evidence (Fact)'], loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2,
                  frameon=False)
        sns.despine(left=True, bottom=True)
        plt.show()

    def plot_confidence_divergence(self):
        """
        Diverging bar chart based on Confidence levels.
        - Claim (Framing) on the LEFT (Negative axis)
        - Evidence (Fact) on the RIGHT (Positive axis)
        """
        # 1. Filter and Pivot
        plot_df = self.df[self.df['Answer'].isin(['Claim', 'Evidence'])]
        counts = pd.crosstab(plot_df['Certainty'], plot_df['Answer'])

        # 2. Normalize and Sort (High at top for barh means it should be last in the index)
        totals = counts.sum(axis=1)
        pct_df = counts.div(totals, axis=0) * 100
        # Reindexing so 'High' is the last index (top of the plot)
        pct_df = pct_df.reindex(['Low', 'Medium', 'High']).dropna()

        fig, ax = plt.subplots(figsize=(12, 6))
        y_pos = np.arange(len(pct_df))

        # 3. Plotting - Claim as negative (Left), Evidence as positive (Right)
        ax.barh(y_pos, -pct_df['Claim'], color=self.palette['Claim'], height=0.6, label='Claim (Framing)')
        ax.barh(y_pos, pct_df['Evidence'], color=self.palette['Evidence'], height=0.6, label='Evidence (Fact)')

        # 4. Center Spine and Axis Formatting
        plt.axvline(0, color='black', linewidth=2, zorder=3)
        plt.xlim(-110, 110)

        plt.yticks(y_pos, pct_df.index, fontweight='bold', fontsize=11)
        ticks = np.arange(-100, 101, 25)
        plt.xticks(ticks, [f"{abs(t)}%" for t in ticks], color='#7f8c8d')

        # 5. Percentage Labels (Pinned to the divider line)
        for i, (idx, row) in enumerate(pct_df.iterrows()):
            # Claim Label (placed just left of 0)
            if row['Claim'] > 0:
                ax.text(-2, i, f"{row['Claim']:.1f}%", va='center', ha='right',
                        color='white', fontweight='bold', fontsize=10)
            # Evidence Label (placed just right of 0)
            if row['Evidence'] > 0:
                ax.text(2, i, f"{row['Evidence']:.1f}%", va='center', ha='left',
                        color='white', fontweight='bold', fontsize=10)

        # 6. Status Labels
        plt.text(-50, len(pct_df) - 0.4, 'CLAIM DRIVEN', ha='center', fontweight='bold',
                 color=self.palette['Claim'],
                 bbox=dict(facecolor='white', edgecolor=self.palette['Claim'], boxstyle='round,pad=0.3'))
        plt.text(50, len(pct_df) - 0.4, 'EVIDENCE DRIVEN', ha='center', fontweight='bold',
                 color=self.palette['Evidence'],
                 bbox=dict(facecolor='white', edgecolor=self.palette['Evidence'], boxstyle='round,pad=0.3'))

        # 7. Legend and Titles
        # Legend order now matches the visual: [Left Side, Right Side]
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                   ncol=2, frameon=False, fontsize=11)

        plt.title('Normalized Epistemic Shift by Confidence Level', fontsize=16, fontweight='bold', pad=40)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()

    def plot_evidence_count_impact(self):
        """Shows how the number of evidences influences model choice."""
        # Filter for the core conflict
        plot_df = self.df[self.df['Answer'].isin(['Claim', 'Evidence'])]
        counts = pd.crosstab(plot_df['Num_evidence'], plot_df['Answer'])
        pct_df = counts.div(counts.sum(axis=1), axis=0) * 100

        plt.figure(figsize=(10, 6))
        plt.plot(pct_df.index, pct_df['Evidence'], marker='o', color=self.palette['Evidence'], linewidth=3,
                 label='Evidence Reliance')
        plt.plot(pct_df.index, pct_df['Claim'], marker='s', color=self.palette['Claim'], linewidth=3,
                 label='Claim Reliance')

        plt.title('Impact of Evidence Volume on Reliance', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Provided Evidence Pieces', fontweight='bold')
        plt.ylabel('Percentage of Answers (%)', fontweight='bold')
        plt.ylim(0, 105)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(frameon=False)
        sns.despine()
        plt.show()

    def plot_overall_distribution(self):
        """Pie chart showing the total distribution of all outcomes."""
        data = self.df['Answer'].value_counts()
        colors = [self.palette.get(x, '#bdc3c7') for x in data.index]

        plt.figure(figsize=(8, 8))
        plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140,
                colors=colors, wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                textprops={'fontweight': 'bold'})

        plt.title('Total Decision Distribution (Full Dataset)', fontsize=15, fontweight='bold')
        plt.show()
