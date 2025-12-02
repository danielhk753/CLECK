# eval_visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


CATEGORIES = ["memory", "claim", "evidence", "no answer", "other"]


def normalize_distribution(dist):
    total = sum(dist.values())
    return {k: dist[k] / total for k in CATEGORIES}


def prepare_dataframe(results):
    rows = []
    for idx, sample in enumerate(results):
        norm = normalize_distribution(sample)
        for cat in CATEGORIES:
            rows.append({
                "sample": idx,
                "category": cat,
                "value": norm[cat]
            })
    return pd.DataFrame(rows)


def plot_stacked_bars(df):
    pivot = df.pivot(index="sample", columns="category", values="value")
    pivot.plot(kind="bar", stacked=True, figsize=(14, 8))
    plt.title("Per-sample Answer Distribution (Stacked Bars)")
    plt.ylabel("Proportion")
    plt.xlabel("Sample Index")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_global_pie(df):
    global_dist = df.groupby("category")["value"].sum()
    plt.figure(figsize=(6, 6))
    plt.pie(global_dist, labels=global_dist.index, autopct="%1.1f%%", startangle=90)
    plt.title("Global Answer Category Distribution")
    plt.show()


def plot_evidence_leakage(df):
    leak_df = df[df["category"] == "evidence"]
    plt.figure(figsize=(12, 5))
    sns.barplot(x="sample", y="value", data=leak_df)
    plt.title("Evidence Leakage per Sample")
    plt.ylabel("Evidence Rate")
    plt.xlabel("Sample Index")
    plt.show()


def plot_claim_histogram(df):
    claim_df = df[df["category"] == "claim"]
    plt.figure(figsize=(7, 5))
    sns.histplot(claim_df["value"], bins=10, kde=False)
    plt.title("Histogram of Claim Rates Across Samples")
    plt.xlabel("Claim Proportion")
    plt.ylabel("Frequency")
    plt.show()


def plot_heatmap(df):
    pivot = df.pivot(index="sample", columns="category", values="value")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".2f")
    plt.title("Answer Distribution Heatmap")
    plt.show()


def plot_claim_vs_evidence(df):
    claim_df = df[df["category"] == "claim"]
    evidence_df = df[df["category"] == "evidence"]

    merged = pd.merge(claim_df, evidence_df, on="sample", suffixes=("_claim", "_evidence"))

    plt.figure(figsize=(7, 7))
    plt.scatter(merged["value_claim"], merged["value_evidence"])
    plt.xlabel("Claim %")
    plt.ylabel("Evidence %")
    plt.title("Claim vs Evidence per Sample")
    plt.grid(True)
    plt.show()


# ---------------------------
# MAIN DRIVER FUNCTION
# ---------------------------

def run_full_evaluation(results1, results2):
    """
    results = list of dicts like:
    [{'memory': 0, 'claim': 79, 'evidence': 17, 'no answer': 0, 'other': 4}, ...]
    """
    df1 = prepare_dataframe(results1)
    df2 = prepare_dataframe(results2)

    # plot_stacked_bars(df)
    plot_global_pie(df1)
    plot_global_pie(df2)
    # plot_evidence_leakage(df)
    # plot_claim_histogram(df)
    # plot_heatmap(df)
    # plot_claim_vs_evidence(df)

    # return df  # Return for further analysis if needed



import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def plot_claim_vs_evidence_two_sets(df_co, df_dist):
    def prep(df):
        claim_df = df[df["category"] == "claim"]
        evidence_df = df[df["category"] == "evidence"]
        merged = pd.merge(claim_df, evidence_df, on="sample", suffixes=("_claim", "_evidence"))
        merged = merged[["sample", "value_claim", "value_evidence"]]
        return merged

    # prepare both
    co = prep(df_co).assign(source="co-located")
    dist = prep(df_dist).assign(source="distributed")

    # combine
    all_df = pd.concat([co, dist], ignore_index=True)

    # Compute density to scale marker size
    coords = list(zip(all_df["value_claim"], all_df["value_evidence"]))
    counts = Counter(coords)
    sizes = np.array([counts[(x, y)] for x, y in coords]) * 40  # scale factor

    # plot
    plt.figure(figsize=(8, 8))

    # plot co-located
    mask_co = all_df["source"] == "co-located"
    plt.scatter(
        all_df[mask_co]["value_claim"],
        all_df[mask_co]["value_evidence"],
        s=sizes[mask_co],
        label="Co-located",
        alpha=0.7
    )

    # plot distributed
    mask_dist = all_df["source"] == "distributed"
    plt.scatter(
        all_df[mask_dist]["value_claim"],
        all_df[mask_dist]["value_evidence"],
        s=sizes[mask_dist],
        label="Distributed",
        alpha=0.7
    )

    # labels
    plt.xlabel("Claim %")
    plt.ylabel("Evidence %")
    plt.title("Claim vs. Evidence – Co-located vs Distributed")
    plt.legend()
    plt.grid(True)
    plt.show()
