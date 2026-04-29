"""
KnowledgeMaze Agreement Analysis Module
=======================================

This module provides tools for assessing inter-annotator reliability between
automated model evaluations (e.g., Gemini 3.1 Pro) and human expert audits
within the KnowledgeMaze "Knowledge Conflict" framework.

The primary function, `calculate_auditor_agreement`, quantifies how closely
the model's automated validation of Rule 1-5 aligns with human judgment.

Methodology & Metrics:
---------------------
1. Agreement Rate:
   The raw percentage of samples where the model and human auditor provided
   the same final verdict (PASS/FAIL).

2. Cohen's Kappa (κ):
   A statistical measure that accounts for agreement occurring by chance.
   - κ > 0.60: Substantial agreement.
   - 0.40 < κ < 0.60: Moderate agreement.
   - κ < 0.40: Fair to poor agreement.

3. Disagreement Distribution:
   Visualizes which reasoning strategies (e.g., FREQUENCY_MODE, TOP_K_RANKING)
   are most prone to conflicting audit results, highlighting potential
   weaknesses in the automated evaluation pipeline.

Data Requirements:
-----------------
- model_raw (list of dict): Objects containing 'passed' (bool) and
  'original_data' with strategy and ID metadata.
- human_raw (list of dict): Objects containing 'status' ('success' or
  specific Rule IDs) and ID metadata.

The module utilizes a vertical dashboard layout with HTML-rendered metric
cards followed by a confusion matrix and a strategy-based disagreement bar chart.
"""

import json
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from sklearn.metrics import confusion_matrix, cohen_kappa_score


def calculate_auditor_agreement(model_raw, human_raw):
    # --- 1. Data Processing & Filtering ---
    # Filter out JSON_PARSE_ERROR before processing
    valid_model_samples = [
        m for m in model_raw
        if m.get('annotation').get('reason_for_rejection') != "JSON_PARSE_ERROR"
    ]

    model_data = [{'ID': m['original_data'].get('ID'),
                   'model_verdict': 1 if m['passed'] else 0,
                   'strategy': m['original_data'].get('Selected strategy')} for m in valid_model_samples]
    df_m = pd.DataFrame(model_data)

    human_data = [{'ID': h.get('ID'),
                   'human_verdict': 1 if h['status'] == 'success' else 0} for h in human_raw]
    df_h = pd.DataFrame(human_data)

    # Join logic
    if df_h['ID'].isnull().all():
        df_merged = pd.concat([df_m.reset_index(drop=True), df_h.reset_index(drop=True)], axis=1)
        df_merged = df_merged.dropna(subset=['model_verdict', 'human_verdict'])
    else:
        df_merged = pd.merge(df_m, df_h, on='ID', how='inner')

    # --- 2. Calculate Metrics ---
    y_model = df_merged['model_verdict'].astype(int)
    y_human = df_merged['human_verdict'].astype(int)
    total = len(df_merged)

    # Raw Agreement
    agreement_rate = (y_model == y_human).mean()

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_model, y_human)

    # Gwet's AC1 Calculation (Robust to Prevalence Paradox)
    p_o = agreement_rate
    # Pi is the average probability that a classification is "PASS"
    pi_pass = ((y_model == 1).sum() + (y_human == 1).sum()) / (2 * total)
    p_e_ac1 = 2 * pi_pass * (1 - pi_pass)
    gwet_ac1 = (p_o - p_e_ac1) / (1 - p_e_ac1) if p_e_ac1 < 1 else 1.0

    disagreements = total - (y_model == y_human).sum()

    # --- 3. Render Dashboard Cards ---
    kappa_color = "#27ae60" if kappa > 0.6 else "#f39c12" if kappa > 0.4 else "#e74c3c"
    ac1_color = "#8e44ad"  # Purple for the adjusted metric

    html_dashboard = f"""
    <div style="display: flex; gap: 15px; font-family: 'Segoe UI', sans-serif; margin-bottom: 30px;">
        <div style="flex: 1; padding: 15px; background: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-top: 5px solid #3498db;">
            <div style="color: #7f8c8d; font-size: 0.75em; font-weight: bold; text-transform: uppercase;">Agreement Rate</div>
            <div style="font-size: 1.8em; font-weight: bold; color: #2c3e50;">{agreement_rate * 100:.1f}%</div>
            <div style="font-size: 0.85em; color: #95a5a6;">{total - disagreements} / {total} samples</div>
        </div>
        <div style="flex: 1; padding: 15px; background: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-top: 5px solid {kappa_color};">
            <div style="color: #7f8c8d; font-size: 0.75em; font-weight: bold; text-transform: uppercase;">Cohen's Kappa</div>
            <div style="font-size: 1.8em; font-weight: bold; color: {kappa_color};">{kappa:.2f}</div>
            <div style="font-size: 0.85em; color: #95a5a6;">Likely biased by prevalence</div>
        </div>
        <div style="flex: 1; padding: 15px; background: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-top: 5px solid {ac1_color};">
            <div style="color: #7f8c8d; font-size: 0.75em; font-weight: bold; text-transform: uppercase;">Gwet's AC1</div>
            <div style="font-size: 1.8em; font-weight: bold; color: {ac1_color};">{gwet_ac1:.2f}</div>
            <div style="font-size: 0.85em; color: #95a5a6;">Prevalence-adjusted</div>
        </div>
        <div style="flex: 1; padding: 15px; background: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-top: 5px solid #e74c3c;">
            <div style="color: #7f8c8d; font-size: 0.75em; font-weight: bold; text-transform: uppercase;">Disagreements</div>
            <div style="font-size: 1.8em; font-weight: bold; color: #c0392b;">{disagreements}</div>
            <div style="font-size: 0.85em; color: #95a5a6;">Audit conflicts</div>
        </div>
    </div>
    """
    display(HTML(html_dashboard))

    # --- 4. Plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1.5]})
    sns.set_theme(style="whitegrid")

    # Confusion Matrix
    cm = confusion_matrix(y_human, y_model)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False,
                xticklabels=['FAIL', 'PASS'], yticklabels=['FAIL', 'PASS'])
    ax1.set_title("Verdict Confusion Matrix", fontweight='bold')

    # Disagreement by Strategy
    df_dis = df_merged[df_merged['model_verdict'] != df_merged['human_verdict']]
    if not df_dis.empty:
        sns.countplot(data=df_dis, y='strategy', hue='strategy', palette='magma', ax=ax2, legend=False)
        ax2.set_title("Disagreement Distribution by Strategy", fontweight='bold')
    else:
        ax2.text(0.5, 0.5, "100% Agreement", ha='center', va='center')

    plt.tight_layout()
    plt.show()

    return df_merged


def gwet_ac1_binary(y1, y2):
    """Calculates Gwet's AC1 for binary classification (0 and 1)."""
    y1, y2 = np.asarray(y1).astype(int), np.asarray(y2).astype(int)
    if len(y1) == 0: return np.nan
    p_a = np.mean(y1 == y2)
    p_1 = (np.mean(y1 == 1) + np.mean(y2 == 1)) / 2
    p_e = 2 * p_1 * (1 - p_1)
    return (p_a - p_e) / (1 - p_e) if p_e < 1 else 1.0


def calculate_human_agreement(json_file_paths, strategy_mapping=None):
    """
    Args:
        json_file_paths: List of paths to annotator JSON files.
        strategy_mapping: Optional dict {ID: strategy_name} to enable the strategy barplot.
    """
    # --- 1. Data Processing ---
    dfs = []
    for i, path in enumerate(json_file_paths):
        with open(path, 'r') as f:
            raw_data = json.load(f)
        ann_name = f"Annotator_{i + 1}"
        parsed = [{'ID': h.get('ID'), ann_name: 1 if h.get('status') == 'success' else 0}
                  for h in raw_data if h.get('ID')]
        dfs.append(pd.DataFrame(parsed).set_index('ID'))

    df_merged = pd.concat(dfs, axis=1)
    ann_cols = [c for c in df_merged.columns if c.startswith("Annotator_")]

    # --- 2. Consensus Logic ---
    row_counts = df_merged[ann_cols].count(axis=1)
    row_sums = df_merged[ann_cols].sum(axis=1)

    def get_consensus(idx):
        s, c = row_sums.loc[idx], row_counts.loc[idx]
        if s == c: return "Unanimous Success"
        if s == 0: return "Unanimous Failure"
        return "Partial Agreement"

    df_merged['consensus'] = [get_consensus(idx) for idx in df_merged.index]
    df_merged['eyes_count'] = row_counts
    if strategy_mapping:
        df_merged['Strategy'] = df_merged.index.map(strategy_mapping)

    # --- 3. Pairwise Metrics (AC1, Kappa, Agreement) ---
    ac1_mat = pd.DataFrame(index=ann_cols, columns=ann_cols, dtype=float)
    kappa_mat = pd.DataFrame(index=ann_cols, columns=ann_cols, dtype=float)
    agree_mat = pd.DataFrame(index=ann_cols, columns=ann_cols, dtype=float)
    shared_mat = pd.DataFrame(index=ann_cols, columns=ann_cols, dtype=int)

    all_ac1s, all_kappas = [], []
    for a1, a2 in itertools.combinations(ann_cols, 2):
        shared = df_merged[[a1, a2]].dropna()
        if not shared.empty:
            ac1 = gwet_ac1_binary(shared[a1], shared[a2])
            kappa = cohen_kappa_score(shared[a1], shared[a2]) if len(np.unique(shared.values)) > 1 else 1.0
            agree = np.mean(shared[a1] == shared[a2])

            for mat, val in zip([ac1_mat, kappa_mat, agree_mat], [ac1, kappa, agree]):
                mat.loc[a1, a2] = mat.loc[a2, a1] = val
            shared_mat.loc[a1, a2] = shared_mat.loc[a2, a1] = len(shared)
            all_ac1s.append(ac1);
            all_kappas.append(kappa)

    # --- 4. Render Dashboard Cards ---
    total_samples = len(df_merged)
    full_pass_rate = (df_merged['consensus'] == "Unanimous Success").mean() * 100
    avg_ac1 = np.nanmean(all_ac1s) if all_ac1s else 0

    pass_color = "#27ae60" if full_pass_rate > 85 else "#f39c12" if full_pass_rate > 60 else "#e74c3c"

    html_dashboard = f"""
    <div style="display: flex; gap: 15px; font-family: 'Segoe UI', sans-serif; margin-bottom: 25px;">
        <div style="flex: 1; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid #3498db;">
            <div style="color: #7f8c8d; font-size: 0.9em; font-weight: bold; text-transform: uppercase;">Total Samples</div>
            <div style="font-size: 2.2em; font-weight: bold; color: #2c3e50;">{total_samples}</div>
        </div>
        <div style="flex: 1; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid {pass_color};">
            <div style="color: #7f8c8d; font-size: 0.9em; font-weight: bold; text-transform: uppercase;">Unanimous Success Rate</div>
            <div style="font-size: 2.2em; font-weight: bold; color: {pass_color};">{full_pass_rate:.1f}%</div>
        </div>
        <div style="flex: 1; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid #e67e22;">
            <div style="color: #7f8c8d; font-size: 0.9em; font-weight: bold; text-transform: uppercase;">Avg Gwet's AC1</div>
            <div style="font-size: 2.2em; font-weight: bold; color: #e67e22;">{avg_ac1:.2f}</div>
        </div>
    </div>
    """
    display(HTML(html_dashboard))

    # --- 5. Visualizations ---
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(18, 24))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.2])

    # Mask for unique pairs only
    mask = np.triu(np.ones_like(ac1_mat, dtype=bool))

    # Helper for Heatmaps
    def draw_heatmap(ax, data, title, fmt):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='RdYlGn', vmin=0, vmax=1,
                    mask=mask, ax=ax, cbar=False, annot_kws={"weight": "bold"})
        ax.set_title(title, fontweight='bold', fontsize=14, pad=12)
        for i, j in zip(*np.where(~mask)):
            if i > j:
                ax.text(j + 0.5, i + 0.8, f"n={shared_mat.iloc[i, j]}",
                        ha='center', va='center', fontsize=8, color='black', alpha=0.7)

    # Row 1: Consensus & Coverage
    ax1 = fig.add_subplot(gs[0, 0])
    c_counts = df_merged['consensus'].value_counts()
    c_map = {"Unanimous Success": "#27ae60", "Partial Agreement": "#f39c12", "Unanimous Failure": "#e74c3c"}
    ax1.pie([c_counts.get(k, 0) for k in c_map.keys()], labels=c_map.keys(), autopct='%1.1f%%',
            startangle=140, colors=c_map.values(), wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax1.set_title("Overall Data Consensus (Accuracy)", fontweight='bold', fontsize=15)

    ax2 = fig.add_subplot(gs[0, 1])
    cov_counts = df_merged['eyes_count'].value_counts().sort_index()
    ax2.pie(cov_counts.values, labels=[f"{int(k)} Annotators" for k in cov_counts.index],
            autopct='%1.1f%%', colors=sns.color_palette("viridis_r", len(cov_counts)),
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax2.set_title("Annotator Coverage Distribution", fontweight='bold', fontsize=15)

    # Row 2 & 3: Reliability Matrices
    draw_heatmap(fig.add_subplot(gs[1, 0]), ac1_mat, "Pairwise Gwet's AC1 (Unique Pairs)", '.2f')
    draw_heatmap(fig.add_subplot(gs[1, 1]), kappa_mat, "Pairwise Cohen's Kappa (Unique Pairs)", '.2f')
    draw_heatmap(fig.add_subplot(gs[2, 0]), agree_mat, "Pairwise Raw Agreement %", '.0%')

    # Row 4: Strategy Success Barplot
    if strategy_mapping:
        ax_strat = fig.add_subplot(gs[3, :])
        strat_stats = []
        for strat, group in df_merged.groupby('Strategy'):
            total = len(group)
            passed = (group['consensus'] == "Unanimous Success").sum()
            rate = (passed / total * 100)
            strat_stats.append({"Strategy": strat, "Success Rate (%)": rate, "Passed": passed, "Total": total})

        df_strat = pd.DataFrame(strat_stats).sort_values("Success Rate (%)", ascending=False)
        sns.barplot(data=df_strat, x="Strategy", y="Success Rate (%)", palette="flare", ax=ax_strat)

        ax_strat.axhline(90, color='red', linestyle='--', alpha=0.4, label="Target (90%)")
        ax_strat.set_title("Unanimous Success Rate per Reasoning Strategy", loc='left', fontsize=16, fontweight='bold',
                           pad=20)
        ax_strat.set_ylim(0, 115)

        for i, p in enumerate(ax_strat.patches):
            row = df_strat.iloc[i]
            label = f"{row['Success Rate (%)']:.1f}%\n({int(row['Passed'])}/{int(row['Total'])})"
            ax_strat.annotate(label, (p.get_x() + p.get_width() / 2., p.get_height()),
                              ha='center', va='bottom', fontsize=10, fontweight='bold',
                              color='#2c3e50', xytext=(0, 8), textcoords='offset points')

    plt.tight_layout(pad=4.0)
    plt.show()

    disagreement_ids = df_merged[df_merged['consensus'] == "Partial Agreement"].index.tolist()
    print(f"✅ Analysis Complete. Disputed IDs found: {len(disagreement_ids)}")
    return df_merged, disagreement_ids
