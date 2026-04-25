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
    # --- 1. Data Processing ---
    model_data = [{'ID': m['original_data'].get('ID'),
                   'model_verdict': 1 if m['passed'] else 0,
                   'strategy': m['original_data'].get('Selected strategy')} for m in model_raw]
    df_m = pd.DataFrame(model_data)

    human_data = [{'ID': h.get('ID'),
                   'human_verdict': 1 if h['status'] == 'success' else 0} for h in human_raw]
    df_h = pd.DataFrame(human_data)

    # Fallback join logic for demonstration
    if df_h['ID'].isnull().all():
        df_merged = pd.concat([df_m.reset_index(drop=True), df_h.reset_index(drop=True)], axis=1)
        df_merged = df_merged.dropna(subset=['model_verdict', 'human_verdict'])
    else:
        df_merged = pd.merge(df_m, df_h, on='ID', how='inner')

    # --- 2. Calculate Metrics ---
    y_model = df_merged['model_verdict']
    y_human = df_merged['human_verdict']

    total = len(df_merged)
    agreement_rate = (y_model == y_human).mean() * 100
    kappa = cohen_kappa_score(y_model, y_human)
    disagreements = total - (y_model == y_human).sum()

    # --- 3. Render Dashboard Cards ---
    kappa_color = "#27ae60" if kappa > 0.6 else "#f39c12" if kappa > 0.4 else "#e74c3c"

    html_dashboard = f"""
    <div style="display: flex; gap: 20px; font-family: 'Segoe UI', sans-serif; margin-bottom: 30px;">
        <div style="flex: 1; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid #3498db;">
            <div style="color: #7f8c8d; font-size: 0.85em; font-weight: bold; text-transform: uppercase;">Agreement Rate</div>
            <div style="font-size: 2.2em; font-weight: bold; color: #2c3e50;">{agreement_rate:.1f}%</div>
            <div style="font-size: 0.9em; color: #95a5a6;">{total - disagreements} / {total} samples matched</div>
        </div>
        <div style="flex: 1; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid {kappa_color};">
            <div style="color: #7f8c8d; font-size: 0.85em; font-weight: bold; text-transform: uppercase;">Cohen's Kappa</div>
            <div style="font-size: 2.2em; font-weight: bold; color: {kappa_color};">{kappa:.2f}</div>
            <div style="font-size: 0.9em; color: #95a5a6;">Inter-annotator reliability</div>
        </div>
        <div style="flex: 1; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid #e74c3c;">
            <div style="color: #7f8c8d; font-size: 0.85em; font-weight: bold; text-transform: uppercase;">Total Disagreements</div>
            <div style="font-size: 2.2em; font-weight: bold; color: #c0392b;">{disagreements}</div>
            <div style="font-size: 0.9em; color: #95a5a6;">Conflict in audit verdicts</div>
        </div>
    </div>
    """
    display(HTML(html_dashboard))

    # --- 4. Plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 1.5]})
    sns.set_theme(style="whitegrid")

    # Confusion Matrix
    cm = confusion_matrix(y_human, y_model)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False,
                xticklabels=['FAIL', 'PASS'], yticklabels=['FAIL', 'PASS'])
    ax1.set_title("Verdict Confusion Matrix", fontweight='bold', pad=15)
    ax1.set_xlabel("Model Verdict")
    ax1.set_ylabel("Human Verdict")

    # Disagreement by Strategy
    df_dis = df_merged[df_merged['model_verdict'] != df_merged['human_verdict']]
    if not df_dis.empty:
        sns.countplot(data=df_dis, y='strategy', hue='strategy', palette='magma', ax=ax2, legend=False)
        ax2.set_title("Disagreement Distribution by Strategy", fontweight='bold', pad=15)
        ax2.set_xlabel("Number of Disputed Samples")
    else:
        ax2.text(0.5, 0.5, "100% Agreement Reached", ha='center', va='center', fontsize=12)

    plt.tight_layout(pad=4.0)
    plt.show()

    return df_merged

def calculate_human_agreement(json_file_paths):
    # --- 1. Data Processing & Outer Join ---
    dfs = []
    for i, path in enumerate(json_file_paths):
        with open(path, 'r') as f:
            raw_data = json.load(f)

        # Parse verdicts: assume 'status' == 'success' means 1, else 0
        annotator_name = f"Annotator_{i + 1}"
        parsed = [{'ID': h.get('ID'),
                   annotator_name: 1 if h.get('status') == 'success' else 0}
                  for h in raw_data if h.get('ID')]  # Skip missing IDs

        df_a = pd.DataFrame(parsed).set_index('ID')
        dfs.append(df_a)

    # Concat along columns creates an outer join based on the 'ID' index
    df_merged = pd.concat(dfs, axis=1)
    annotator_cols = df_merged.columns.tolist()

    # --- 2. Calculate Coverage ("Eyes per sample") ---
    df_merged['eyes_count'] = df_merged[annotator_cols].count(axis=1)

    # --- 3. Calculate Pairwise Metrics ---
    # Create empty matrices for Kappa and Shared Sample counts
    kappa_matrix = pd.DataFrame(index=annotator_cols, columns=annotator_cols, dtype=float)
    shared_matrix = pd.DataFrame(index=annotator_cols, columns=annotator_cols, dtype=int)

    kappas = []

    for a1, a2 in itertools.combinations(annotator_cols, 2):
        # Drop rows where either annotator is missing a verdict
        shared = df_merged[[a1, a2]].dropna()
        shared_count = len(shared)

        shared_matrix.loc[a1, a2] = shared_count
        shared_matrix.loc[a2, a1] = shared_count

        if shared_count > 0:
            # Check if there is more than 1 class present, otherwise Kappa is undefined/0
            if len(np.unique(shared)) > 1:
                k = cohen_kappa_score(shared[a1], shared[a2])
            else:
                k = 1.0  # 100% agreement on a single class

            kappa_matrix.loc[a1, a2] = k
            kappa_matrix.loc[a2, a1] = k
            kappas.append(k)
        else:
            kappa_matrix.loc[a1, a2] = np.nan
            kappa_matrix.loc[a2, a1] = np.nan

    # Fill diagonals
    np.fill_diagonal(kappa_matrix.values, 1.0)
    for a in annotator_cols:
        shared_matrix.loc[a, a] = df_merged[a].count()

    # Aggregate Metrics
    avg_kappa = np.nanmean(kappas) if kappas else 0
    total_samples = len(df_merged)
    avg_eyes = df_merged['eyes_count'].mean()

    # --- 4. Render Dashboard Cards ---
    kappa_color = "#27ae60" if avg_kappa > 0.6 else "#f39c12" if avg_kappa > 0.4 else "#e74c3c"

    html_dashboard = f"""
    <div style="display: flex; gap: 20px; font-family: 'Segoe UI', sans-serif; margin-bottom: 30px;">
        <div style="flex: 1; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid #8e44ad;">
            <div style="color: #7f8c8d; font-size: 0.85em; font-weight: bold; text-transform: uppercase;">Total Unique Samples</div>
            <div style="font-size: 2.2em; font-weight: bold; color: #2c3e50;">{total_samples}</div>
            <div style="font-size: 0.9em; color: #95a5a6;">Across {len(annotator_cols)} annotators</div>
        </div>
        <div style="flex: 1; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid {kappa_color};">
            <div style="color: #7f8c8d; font-size: 0.85em; font-weight: bold; text-transform: uppercase;">Average Pairwise Kappa</div>
            <div style="font-size: 2.2em; font-weight: bold; color: {kappa_color};">{avg_kappa:.2f}</div>
            <div style="font-size: 0.9em; color: #95a5a6;">Mean inter-annotator reliability</div>
        </div>
        <div style="flex: 1; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 5px solid #3498db;">
            <div style="color: #7f8c8d; font-size: 0.85em; font-weight: bold; text-transform: uppercase;">Average Coverage</div>
            <div style="font-size: 2.2em; font-weight: bold; color: #2980b9;">{avg_eyes:.2f}</div>
            <div style="font-size: 0.9em; color: #95a5a6;">Eyes per sample</div>
        </div>
    </div>
    """
    display(HTML(html_dashboard))

    # --- 5. Plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1.2, 1]})
    sns.set_theme(style="whitegrid")

    # Plot 1: Pairwise Kappa Heatmap
    mask = np.triu(np.ones_like(kappa_matrix, dtype=bool))  # Hide upper triangle for cleaner look
    sns.heatmap(kappa_matrix.astype(float), annot=True, fmt='.2f', cmap='YlGnBu',
                mask=mask, vmin=0, vmax=1, ax=ax1, cbar_kws={'label': "Cohen's Kappa"})
    ax1.set_title("Pairwise Inter-Annotator Agreement (Kappa)", fontweight='bold', pad=15)

    # Overlay shared counts as text below the Kappa scores
    for i in range(len(annotator_cols)):
        for j in range(len(annotator_cols)):
            if i > j:  # Lower triangle only
                ax1.text(j + 0.5, i + 0.7, f"n={shared_matrix.iloc[i, j]}",
                         ha='center', va='center', fontsize=9, color='gray')

    # Plot 2: Coverage Distribution (Eyes per Sample)
    sns.countplot(data=df_merged, x='eyes_count', hue='eyes_count', palette='viridis', ax=ax2, legend=False)
    ax2.set_title("Sample Coverage Distribution", fontweight='bold', pad=15)
    ax2.set_xlabel("Number of Annotators (Eyes)")
    ax2.set_ylabel("Number of Samples")

    # Add percentage labels on top of bars
    total = len(df_merged)
    for p in ax2.patches:
        height = p.get_height()
        if height > 0:
            ax2.annotate(f'{height / total * 100:.1f}%',
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=10, xytext=(0, 5),
                         textcoords='offset points')

    plt.tight_layout(pad=4.0)
    plt.show()

    return df_merged, kappa_matrix, shared_matrix