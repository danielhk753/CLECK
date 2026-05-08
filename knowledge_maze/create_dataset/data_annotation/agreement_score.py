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

import re
import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path


from IPython.display import display, HTML


# =========================
# Agreement Metric
# =========================
def gwet_ac1_binary(y1, y2):
    """
    Gwet's AC1 for binary labels {0,1}.
    More stable than Cohen's Kappa under class imbalance.
    """
    y1 = np.asarray(y1).astype(int)
    y2 = np.asarray(y2).astype(int)

    if len(y1) == 0:
        return np.nan

    p_a = np.mean(y1 == y2)
    p1 = (np.mean(y1 == 1) + np.mean(y2 == 1)) / 2
    p_e = 2 * p1 * (1 - p1)

    if p_e >= 1:
        return 1.0

    return (p_a - p_e) / (1 - p_e)


# =========================
# Loading
# =========================
def load_annotation_files(json_paths):
    from collections import defaultdict

    all_data = defaultdict(dict)
    annotator_names = []

    for path in json_paths:
        path = Path(path)
        stem = path.stem.lower()

        # --- name parsing ---
        if "expert" in stem:
            annotator_name = "Expert"
        else:
            match = re.findall(r"\d+", stem)
            annotator_name = f"Annotator{match[0]}" if match else stem

        annotator_names.append(annotator_name)

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for item in raw:
            sid = item.get("ID")
            if sid is None:
                continue

            # overwrite-safe aggregation
            all_data[sid][annotator_name] = 1 if item.get("status") == "success" else 0

    # --- build unified dataframe ---
    df = pd.DataFrame.from_dict(all_data, orient="index")

    # ensure all annotator columns exist
    for a in annotator_names:
        if a not in df.columns:
            df[a] = np.nan

    return df


# =========================
# Consensus
# =========================
def add_consensus_columns(df):
    df = df.copy()
    ann_cols = [c for c in df.columns if c.startswith("Annotator") or c == "Expert"]
    df["n_annotations"] = df[ann_cols].notna().sum(axis=1)
    df["sum_valid"] = df[ann_cols].sum(axis=1)

    def consensus(row):
        s = row["sum_valid"]
        n = row["n_annotations"]

        if s == n:
            return "Unanimous Valid"
        elif s == 0:
            return "Unanimous Invalid"
        else:
            return "Disagreement"

    df["consensus"] = df.apply(consensus, axis=1)
    return df


# =========================
# Summary Table
# =========================
def compute_summary(df):
    total_samples = len(df)

    shared_df = df[df["n_annotations"] > 1]
    shared_samples = len(shared_df)

    shared_ratio = shared_samples / total_samples

    unanimous_valid_shared = (
        (shared_df["consensus"] == "Unanimous Valid").mean()
        if shared_samples > 0 else np.nan
    )

    summary = pd.DataFrame([
        ("Total annotated samples", total_samples),
        ("Double-annotated samples", f"{shared_samples} ({shared_ratio:.1%})"),
        ("Shared unanimous validity", f"{unanimous_valid_shared:.1%}")
    ], columns=["Metric", "Value"])

    return summary


# =========================
# Pairwise Agreement
# =========================
def compute_pairwise_metrics(df):
    ann_cols = [c for c in df.columns if c.startswith("Annotator") or c == "Expert"]
    rows = []

    agreements = []
    ac1s = []
    ns = []

    for a1, a2 in itertools.combinations(ann_cols, 2):
        shared = df[[a1, a2]].dropna()

        if len(shared) == 0:
            continue

        agreement = np.mean(shared[a1] == shared[a2])
        ac1 = gwet_ac1_binary(shared[a1], shared[a2])

        agreements.append(agreement)
        ac1s.append(ac1)
        ns.append(len(shared))

        rows.append({
            "Pair": f"{a1} ↔ {a2}",
            "Shared n": len(shared),
            "Agreement": f"{agreement:.1%}",
            "Gwet AC1": round(ac1, 3)
        })

    # weighted pooled overall
    if len(rows) > 0:
        weighted_agreement = np.average(
            [float(a.strip('%')) / 100 for a in [r["Agreement"] for r in rows]],
            weights=ns
        )
        weighted_ac1 = np.average(ac1s, weights=ns)

        rows.append({
            "Pair": "Overall",
            "Shared n": sum(ns),
            "Agreement": f"{weighted_agreement:.1%}",
            "Gwet AC1": round(weighted_ac1, 3)
        })

    return pd.DataFrame(rows)


# =========================
# Display Tables Side-by-Side
# =========================
def display_tables_side_by_side(df1, df2):
    html = f"""
    <div style="display:flex; gap:40px; align-items:flex-start;">
        <div style="flex:1">
            <h3>Summary</h3>
            {df1.to_html(index=False)}
        </div>
        <div style="flex:1">
            <h3>Pairwise Agreement</h3>
            {df2.to_html(index=False)}
        </div>
    </div>
    """
    display(HTML(html))

# =========================
# Main
# =========================
def run_annotation_report(json_paths):
    df = load_annotation_files(json_paths)
    df = add_consensus_columns(df)

    summary_df = compute_summary(df)
    pairwise_df = compute_pairwise_metrics(df)

    display_tables_side_by_side(summary_df, pairwise_df)

    return df, summary_df, pairwise_df