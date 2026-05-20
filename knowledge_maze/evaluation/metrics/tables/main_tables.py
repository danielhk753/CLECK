import os
import pandas as pd
from collections import Counter
from IPython.display import display
from knowledge_maze.tools.utils import load_json


class MetricsReporter:
    def __init__(self, repo_path):
        self.repo_path = repo_path

    def calculate_main_metrics(self, data):
        total_samples, total_iterations = 0, 0
        raw_ack_count, stable_ack_count = 0, 0
        stable_pref_counts = Counter()

        for _, sample in data.items():
            responses = sample.get("responses", [])
            judged = [r for r in responses if "judge_results" in r]
            if len(judged) < 5: continue

            total_samples += 1
            total_iterations += 5

            ack_flags = [r["judge_results"].get("dual_awareness", False) for r in judged]
            raw_ack_count += sum(ack_flags)
            if sum(ack_flags) >= 4: stable_ack_count += 1

            classes = [r["judge_results"].get("classification", "other") for r in judged]
            counts = Counter(classes)
            if counts["claim"] >= 4:
                stable_pref = "claim"
            elif counts["evidence"] >= 4:
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
        }

    def calculate_micro_metrics(self, data):
        ack_counter = Counter()
        noack_counter = Counter()

        for _, sample in data.items():
            judged = [r for r in sample.get("responses", []) if "judge_results" in r]
            for r in judged:
                cls = r["judge_results"].get("classification", "other")
                if cls not in {"claim", "evidence"}: continue

                if r["judge_results"].get("dual_awareness", False):
                    ack_counter[cls] += 1
                else:
                    noack_counter[cls] += 1

        def normalize(c):
            t = c["claim"] + c["evidence"]
            return {"claim": round(c["claim"] / t * 100, 2) if t > 0 else 0,
                    "evidence": round(c["evidence"] / t * 100, 2) if t > 0 else 0}

        ack = normalize(ack_counter)
        noack = normalize(noack_counter)
        return {"claim_ack": ack["claim"], "evidence_ack": ack["evidence"],
                "claim_noack": noack["claim"], "evidence_noack": noack["evidence"]}

    def run_report(self, model_configs):
        """
        model_configs: dict {DisplayName: "path/to/file.json"}
        """
        main_results, micro_results = {}, {}

        for name, path in model_configs.items():
            full_path = os.path.join(self.repo_path, path)
            data = load_json(full_path)
            main_results[name] = self.calculate_main_metrics(data)
            micro_results[name] = self.calculate_micro_metrics(data)

        # Main Table
        # Inside your run_report method:
        df_main = pd.DataFrame([{"Model": k, **v} for k, v in main_results.items()])

        # Define formatting
        fmt = {c: "{:.2f}" for c in ['ack', 'stable_ack', 'pref_claim', 'pref_mix', 'pref_evid']}

        # Apply formatting AND the background gradient
        df_styled = df_main.style.format(fmt).background_gradient(subset=["stable_ack"], cmap="Greens")

        # Micro Table
        df_micro = pd.DataFrame([{
            "Model": k,
            "Claim (Ack|NoAck)": f'{v["claim_ack"]:.1f} | {v["claim_noack"]:.1f}',
            "Evidence (Ack|NoAck)": f'{v["evidence_ack"]:.1f} | {v["evidence_noack"]:.1f}'
        } for k, v in micro_results.items()])

        print("--- Main Benchmark Report ---")
        display(df_styled)
        print("\n--- Conditional Micro Report ---")
        display(df_micro)