"""
Cross-model agreement analysis for the CLECK benchmark.

Loads per-model baseline result files, computes instance-level cross-model
agreement on conflict awareness and resolution preference, and produces three
figures (each also saved as a PDF):

  1. Awareness  — pairwise Cohen's kappa across three binarization thresholds.
  2. Preference — pairwise agreement above marginal-prior baseline (heatmap).
  3. Preference — pairwise agreement above marginal-prior baseline (bar chart).

Entry point:
    run_agreement_analysis(model_files, out_dir="plots")
"""

import os
import json
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from dotenv.main import load_dotenv

load_dotenv()
REPO_PATH = os.getenv("REPO_PATH")

QA_CONFIGS = {
    "DeepSeek V4 Flash": "results/baseline/deepseek_deepseek-v4-flash.json",
    "Gemini 3 Flash": "results/baseline/google_gemini-3-flash-preview.json",
    "Llama 4 Maverick": "results/baseline/meta-llama_llama-4-maverick.json",
    "GPT 5.4 Mini": "results/baseline/openai_gpt-5.4-mini.json",
    "Mistral 3.1 Medium": "results/baseline/mistralai_mistral-medium-3.1.json",
}

# =============================================================================
# LOADING
# =============================================================================

def load_iterations(path):
    """
    Load one model's result file.

    Returns
    -------
    List[Tuple[str, List[Tuple[bool, str]]]]
        Each element: (sample_id, [(dual_awareness, classification), ...])
        Outer list in file order, inner list in iteration order.

    Responses missing a ``judge_results`` field are skipped (the rest of the
    sample is kept). A summary count is printed when any are dropped.
    """
    with open(path) as f:
        data = json.load(f)

    out = []
    n_dropped = 0
    for sid, s in data.items():
        iters = []
        for r in s.get("responses", []):
            jr = r.get("judge_results")
            if jr is None:
                n_dropped += 1
                continue
            iters.append((
                bool(jr.get("dual_awareness")),
                jr.get("classification"),
            ))
        out.append((sid, iters))

    if n_dropped > 0:
        print(f"[load_iterations] {path}: skipped {n_dropped} responses "
              f"missing 'judge_results'")
    return out


# =============================================================================
# PER-SAMPLE SCALARS & ALIGNMENT
# =============================================================================

def _per_model_scalars(iters):
    """For one sample of one model, return (a, p, cp). NaNs when iters is empty."""
    n = len(iters)
    if n == 0:
        return np.nan, np.nan, np.nan
    n_aw    = sum(1 for aw, _ in iters if aw)
    n_cl    = sum(1 for _,  c in iters if c == "claim")
    n_ev    = sum(1 for _,  c in iters if c == "evidence")
    n_cl_aw = sum(1 for aw, c in iters if aw and c == "claim")
    n_ev_aw = sum(1 for aw, c in iters if aw and c == "evidence")
    a  = n_aw / n
    p  = (n_cl - n_ev) / n
    cp = (n_cl_aw - n_ev_aw) / n_aw if n_aw > 0 else np.nan
    return a, p, cp


def _per_sample_pref_label(iters):
    """Majority preference label for one sample: claim / evidence / mixed."""
    counts = Counter(c if c in ("claim", "evidence") else "mixed" for _, c in iters)
    if not counts:
        return "mixed"
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "mixed"
    return top[0][0]


def _align(per_model_data):
    """Verify identical ID sets and return (canonical_ids, list_of_dicts)."""
    dicts = [dict(d) for d in per_model_data]
    canonical_ids = [sid for sid, _ in per_model_data[0]]
    canonical_set = set(canonical_ids)
    for i, d in enumerate(dicts[1:], start=1):
        other = set(d.keys())
        if other != canonical_set:
            missing = canonical_set - other
            extra = other - canonical_set
            raise ValueError(
                f"Model index {i} has a different sample ID set than model 0. "
                f"Missing {len(missing)} ids, extra {len(extra)} ids."
            )
    return canonical_ids, dicts


# =============================================================================
# AGREEMENT SCALARS (std-based) + SUMMARY
# =============================================================================

def compute_per_sample_agreement(per_model_data):
    """
    Compute three cross-model agreement scalars per sample (std-based).

    Returns
    -------
    List[Tuple[str, Tuple[float, float, float]]]
        (sample_id, (aware_agreement, pref_agreement, cond_pref_agreement)),
        each in [0, 1] where 1 = perfect cross-model agreement.
    """
    M = len(per_model_data)
    if M < 2:
        raise ValueError(f"Need at least 2 models, got {M}.")

    canonical_ids, dicts = _align(per_model_data)

    out = []
    for sid in canonical_ids:
        a_vals, p_vals, cp_vals = [], [], []
        for d in dicts:
            a, p, cp = _per_model_scalars(d[sid])
            a_vals.append(a)
            p_vals.append(p)
            cp_vals.append(cp)

        a_valid  = np.asarray(a_vals,  float)[~np.isnan(a_vals)]
        p_valid  = np.asarray(p_vals,  float)[~np.isnan(p_vals)]
        cp_valid = np.asarray(cp_vals, float)[~np.isnan(cp_vals)]

        a_agree  = 1.0 - 2.0 * a_valid.std(ddof=0)  if len(a_valid)  >= 2 else np.nan
        p_agree  = 1.0 - p_valid.std(ddof=0)        if len(p_valid)  >= 2 else np.nan
        cp_agree = 1.0 - cp_valid.std(ddof=0)       if len(cp_valid) >= 2 else np.nan

        out.append((sid, (a_agree, p_agree, cp_agree)))
    return out


def report_agreement(agreement):
    """Print a short summary of compute_per_sample_agreement output."""
    arr = np.array([scores for _, scores in agreement], dtype=float)
    labels = ["aware", "pref", "cond_pref"]
    n = len(arr)
    print(f"=== Per-sample cross-model agreement | {n} samples ===\n")

    print(f"{'axis':<10} {'mean':>6} {'median':>7} {'std':>6} "
          f"{'q25':>6} {'q75':>6} {'n_nan':>6}")
    for i, label in enumerate(labels):
        col = arr[:, i]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            print(f"{label:<10}  (no valid values)")
            continue
        print(f"{label:<10} {valid.mean():>6.3f} {np.median(valid):>7.3f} "
              f"{valid.std(ddof=0):>6.3f} {np.percentile(valid, 25):>6.3f} "
              f"{np.percentile(valid, 75):>6.3f} {(len(col) - len(valid)):>6d}")

    print("\nDistribution by agreement level:")
    print(f"{'axis':<10} {'≥0.9':>8} {'0.7–0.9':>9} {'0.5–0.7':>9} {'<0.5':>8}")
    for i, label in enumerate(labels):
        valid = arr[:, i][~np.isnan(arr[:, i])]
        if len(valid) == 0:
            continue
        t = len(valid)
        hi = (valid >= 0.9).sum()
        mh = ((valid >= 0.7) & (valid < 0.9)).sum()
        ml = ((valid >= 0.5) & (valid < 0.7)).sum()
        lo = (valid < 0.5).sum()
        print(f"{label:<10} {hi/t:>8.1%} {mh/t:>9.1%} {ml/t:>9.1%} {lo/t:>8.1%}")


# =============================================================================
# MATRIX BUILDERS
# =============================================================================

def cohens_kappa(a, b):
    """Cohen's kappa for two binary arrays of equal length."""
    p_o = np.mean(a == b)
    p_a, p_b = np.mean(a), np.mean(b)
    p_e = p_a * p_b + (1 - p_a) * (1 - p_b)
    return (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 1.0


def build_awareness_matrix(per_model_data):
    """(N, M) matrix of per-sample awareness rates, one column per model."""
    canonical_ids, dicts = _align(per_model_data)
    M, N = len(dicts), len(canonical_ids)
    a_mat = np.full((N, M), np.nan)
    for r, sid in enumerate(canonical_ids):
        for c, d in enumerate(dicts):
            a_mat[r, c], _, _ = _per_model_scalars(d[sid])
    return a_mat


def build_preference_excess(per_model_data):
    """
    (M, M) matrix of pairwise preference agreement above the marginal-prior
    baseline: observed agreement minus the agreement expected if each model's
    per-sample preference label were drawn independently from its own marginal
    distribution over {claim, evidence, mixed}.
    """
    canonical_ids, dicts = _align(per_model_data)
    M, N = len(dicts), len(canonical_ids)

    labels = np.empty((N, M), dtype=object)
    for r, sid in enumerate(canonical_ids):
        for c, d in enumerate(dicts):
            labels[r, c] = _per_sample_pref_label(d[sid])

    cats = ("claim", "evidence", "mixed")
    marg = np.zeros((M, len(cats)))
    for c in range(M):
        for k, cat in enumerate(cats):
            marg[c, k] = np.mean(labels[:, c] == cat)

    excess = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            observed = np.mean(labels[:, i] == labels[:, j])
            expected = float(np.dot(marg[i], marg[j]))
            excess[i, j] = observed - expected
    return excess


# =============================================================================
# PLOTS
# =============================================================================

def plot_awareness_kappa(a_mat, model_names, thresholds=(0.2, 0.5, 0.8),
                         out_path=None):
    M = len(model_names)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)
    mask = np.tril(np.ones((M, M), dtype=bool), k=-1)
    im = None

    for idx, thresh in enumerate(thresholds):
        ax = axes[idx]
        aware_bin = (a_mat >= thresh).astype(int)
        kappa_a = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                if mask[i, j]:
                    kappa_a[i, j] = cohens_kappa(aware_bin[:, i], aware_bin[:, j])
        disp = np.where(mask, kappa_a, np.nan)
        im = ax.imshow(disp, cmap="YlGn", vmin=0.0, vmax=0.30)
        for i in range(M):
            for j in range(M):
                if mask[i, j]:
                    v = kappa_a[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=9.5, color="white" if v > 0.18 else "black")
        ax.set_xticks(range(M))
        ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=9)
        ax.set_title(f"Threshold = {thresh}", fontweight="bold", fontsize=11, pad=10)
        if idx == 0:
            ax.set_yticks(range(M))
            ax.set_yticklabels(model_names, fontsize=9)
        else:
            ax.yaxis.set_ticks_position("none")

    fig.subplots_adjust(right=0.88, top=0.85)
    cbar_ax = fig.add_axes([0.90, 0.22, 0.015, 0.55])
    fig.colorbar(im, cax=cbar_ax, label="Cohen's κ")
    plt.suptitle("Awareness Profiling — Pairwise Cohen's κ Sensitivity Analysis",
                 fontweight="bold", fontsize=13, y=0.95)

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    return fig


def plot_preference_heatmap(excess, model_names, out_path=None):
    M = len(model_names)
    mat = excess.copy()
    np.fill_diagonal(mat, np.nan)
    mask = np.tril(np.ones_like(mat, dtype=bool), k=-1)
    disp = np.where(mask, mat, np.nan)

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(disp, cmap="YlGn", vmin=0.0, vmax=0.30)
    for i in range(M):
        for j in range(M):
            if mask[i, j]:
                v = mat[i, j]
                ax.text(j, i, f"+{v:.2f}", ha="center", va="center",
                        fontsize=10, color="white" if v > 0.20 else "black")
    ax.set_xticks(range(M)); ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(M)); ax.set_yticklabels(model_names, fontsize=9)
    ax.set_title("Preference — pairwise agreement above marginal-prior baseline",
                 fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Δ (observed − expected)")
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    return fig


def plot_preference_bars(excess, model_names, out_path=None):
    M = len(model_names)
    pairs = []
    for i in range(M):
        for j in range(i + 1, M):
            pairs.append((f"{model_names[i]} × {model_names[j]}", excess[i, j]))
    pairs.sort(key=lambda x: x[1], reverse=True)
    labels, values = zip(*pairs)

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    colors = plt.cm.YlGn(np.array(values) / max(values))
    ax.barh(range(len(values)), values, color=colors, edgecolor="white", linewidth=1)
    for i, v in enumerate(values):
        ax.text(v + 0.005, i, f"+{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Δ (observed − expected pairwise agreement)")
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_title("Preference — pairwise agreement above marginal-prior baseline",
                 fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    return fig


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_agreement_analysis(model_files=QA_CONFIGS, out_dir="plots"):
    """
    Load all models, build the awareness and preference matrices, and render +
    save the three agreement figures.

    Returns (a_mat, excess, (fig1, fig2, fig3)).
    """
    os.makedirs(out_dir, exist_ok=True)
    model_names = list(model_files.keys())
    per_model = [load_iterations(os.path.join(REPO_PATH, p))
                 for p in model_files.values()]

    a_mat = build_awareness_matrix(per_model)
    excess = build_preference_excess(per_model)

    fig1 = plot_awareness_kappa(
        a_mat, model_names,
        out_path=os.path.join(out_dir, "awareness_kappa_heatmap.pdf"))
    fig2 = plot_preference_heatmap(
        excess, model_names,
        out_path=os.path.join(out_dir, "preference_excess_heatmap.pdf"))
    fig3 = plot_preference_bars(
        excess, model_names,
        out_path=os.path.join(out_dir, "preference_pairwise_agreement.pdf"))

    return a_mat, excess, (fig1, fig2, fig3)