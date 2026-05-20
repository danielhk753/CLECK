import os
import pandas as pd
from collections import Counter
from dotenv import load_dotenv
from IPython.display import display
from knowledge_maze.tools.utils import load_json

load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')


# =============================================================================
# EXPANDED METRIC CALCULATION
# =============================================================================

def calculate_main_metrics(data):
    total_samples = 0
    total_iterations = 0
    raw_ack_count = 0
    stable_ack_count = 0

    # Track iteration-level outcomes for conditional probability
    ack_claim_count = 0
    ack_evidence_count = 0

    stable_pref_counts = Counter()

    for sample_id, sample in data.items():
        responses = sample.get("responses", [])
        judged = [r for r in responses if "judge_results" in r]

        if len(judged) < 5: continue

        total_samples += 1
        total_iterations += 5

        for r in judged:
            jr = r["judge_results"]
            is_ack = jr.get("dual_awareness", False)
            classif = jr.get("classification", "other")

            if is_ack:
                raw_ack_count += 1
                if classif == "claim":
                    ack_claim_count += 1
                elif classif == "evidence":
                    ack_evidence_count += 1

        # Stable Ack (Sample level)
        ack_flags = [r["judge_results"].get("dual_awareness", False) for r in judged]
        if sum(ack_flags) >= 4:
            stable_ack_count += 1

        # Stable Preference (Sample level)
        classes = [r["judge_results"].get("classification", "other") for r in judged]
        class_counts = Counter(classes)
        if class_counts["claim"] >= 4:
            stable_pref = "claim"
        elif class_counts["evidence"] >= 4:
            stable_pref = "evidence"
        else:
            stable_pref = "mixed"
        stable_pref_counts[stable_pref] += 1

    def pct(x, denom):
        return round((x / denom) * 100, 2) if denom > 0 else 0.0

    return {
        "samples_n": total_samples,
        "ack": pct(raw_ack_count, total_iterations),
        "stable_ack": pct(stable_ack_count, total_samples),
        "pref_claim": pct(stable_pref_counts["claim"], total_samples),
        "pref_mix": pct(stable_pref_counts["mixed"], total_samples),
        "pref_evid": pct(stable_pref_counts["evidence"], total_samples),
        # Conditional metrics (Denom: raw_ack_count)
        "cond_claim": pct(ack_claim_count, raw_ack_count),
        "cond_evid": pct(ack_evidence_count, raw_ack_count)
    }


# =============================================================================
# FORMATTING & STYLING
# =============================================================================

def format_and_style_table(df):
    output_rows = []
    # Identify metric columns to process for deltas
    metric_cols = ["Ack", "StableAck", "Claim", "Mix", "Evidence", "CondClaim", "CondEvid"]

    for i in range(0, len(df), 2):
        base = df.iloc[i]
        abl = df.iloc[i + 1]

        # Baseline Row
        row = {"Model": base["Model"], "Version": "Baseline"}
        for col in metric_cols:
            row[col] = f"{base[col]:.1f}"
        output_rows.append(row)

        # Ablation Row with Delta
        delta_row = {"Model": "", "Version": abl["Version"]}
        for col in metric_cols:
            diff = abl[col] - base[col]
            delta_row[col] = f"{abl[col]:.1f} ({diff:+.1f})"
        output_rows.append(delta_row)

    final_df = pd.DataFrame(output_rows)

    # Color function for deltas
    def color_delta(val):
        if '(' in str(val):
            try:
                num = float(val.split('(')[1].replace(')', ''))
                return 'color: green' if num > 0 else 'color: red' if num < 0 else 'color: black'
            except:
                return 'color: black'
        return 'color: black'

    # Apply style
    styled = final_df.style.map(color_delta, subset=metric_cols)
    display(styled)
    return final_df


# =============================================================================
# RUNNER
# =============================================================================

def run_ablation_comparison(ablation_name, model_configs):
    rows = []
    for cfg in model_configs:
        base_full = load_json(os.path.join(REPO_PATH, cfg['base_path']))
        abl_full = load_json(os.path.join(REPO_PATH, cfg['abl_path']))

        common_keys = set(base_full.keys()).intersection(set(abl_full.keys()))
        base_metrics = calculate_main_metrics({k: base_full[k] for k in common_keys})
        abl_metrics = calculate_main_metrics({k: abl_full[k] for k in common_keys})

        for label, m in [("Baseline", base_metrics), (ablation_name, abl_metrics)]:
            rows.append({
                "Model": cfg['name'],
                "Version": label,
                "Ack": m["ack"],
                "StableAck": m["stable_ack"],
                "Claim": m["pref_claim"],
                "Mix": m["pref_mix"],
                "Evidence": m["pref_evid"],
                "CondClaim": m["cond_claim"],
                "CondEvid": m["cond_evid"]
            })
    return format_and_style_table(pd.DataFrame(rows))


def run_scaling_comparison(model_configs):
    rows = []
    for cfg in model_configs:
        small_full = load_json(os.path.join(REPO_PATH, cfg['small_path']))
        large_full = load_json(os.path.join(REPO_PATH, cfg['large_path']))

        # Alignment
        common_keys = set(small_full.keys()).intersection(set(large_full.keys()))
        small_metrics = calculate_main_metrics({k: small_full[k] for k in common_keys})
        large_metrics = calculate_main_metrics({k: large_full[k] for k in common_keys})

        # We append them as 'Small' (Baseline) and 'Large' (Ablation/Scaling)
        for label, m in [("Small", small_metrics), ("Large", large_metrics)]:
            rows.append({
                "Model": cfg['name'],
                "Version": label,
                "Ack": m["ack"],
                "StableAck": m["stable_ack"],
                "Claim": m["pref_claim"],
                "Mix": m["pref_mix"],
                "Evidence": m["pref_evid"],
                "CondClaim": m["cond_claim"],
                "CondEvid": m["cond_evid"]
            })

    # Reuse your existing formatting function
    return format_and_style_table(pd.DataFrame(rows))

def run_all_ablations():
    # 1. Define your base models
    base_models = [
        {"name": "DeepSeek V4", "file": "deepseek_deepseek-v4-flash.json"},
        {"name": "Gemini 3", "file": "google_gemini-3-flash-preview.json"},
        {"name": "Llama 4", "file": "meta-llama_llama-4-maverick.json"}
    ]

    # 2. Build the configs for CoT
    cot_configs = [
        {
            "name": m["name"],
            "base_path": f"results/baseline/{m['file']}",
            "abl_path": f"results/cot/{m['file']}"
        }
        for m in base_models
    ]

    # 3. Build the configs for Hedging
    hedge_configs = [
        {
            "name": m["name"],
            "base_path": f"results/baseline/{m['file']}",
            "abl_path": f"results/hedged/{m['file']}"
        }
        for m in base_models
    ]

    # 4. Build the configs for Scaling
    scaling_configs = [
        {
            "name": "DeepSeek V4",
            "small_path": "results/baseline/deepseek_deepseek-v4-flash.json",
            "large_path": "results/scaling/deepseek_deepseek-v4-pro.json"
        },
        {
            "name": "GPT 5.4",
            "small_path": "results/baseline/openai_gpt-5.4-mini.json",
            "large_path": "results/scaling/openai_gpt-5.4.json"
        }
    ]


    # Run CoT Comparison
    print("--- CoT Ablation Report ---")
    run_ablation_comparison(ablation_name="CoT", model_configs=cot_configs)

    # Run Hedging Comparison
    print("\n--- Hedging Ablation Report ---")
    run_ablation_comparison(ablation_name="Hedge", model_configs=hedge_configs)
    print()

    # Run Scaling Comparison
    print("--- Scaling Comparison Report ---")
    run_scaling_comparison(scaling_configs)