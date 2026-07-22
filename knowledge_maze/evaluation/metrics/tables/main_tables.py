"""
CLECK Benchmark Evaluation and Reporting Pipeline.

This module acts as the primary analytical reporting suite for the CLECK (CLaim
Evidence Conflict of Knowledge) benchmark. It processes evaluation logs from
both baseline QA execution and extraction/retrieval phases to generate structured,
publication-ready dataframes mapping directly to thesis experimental results.

The pipeline focuses on evaluating three core dimensions of model behavior under
latent knowledge conflicts:
1. Dual Awareness (§5.1.1): Contrasting a model's information-retrieval limits
   against its explicit epistemic awareness during generation.
2. Recall Sufficiency (§5.1.1): Quantifying independent and joint target retrieval
   rates for underlying claims and evidence.
3. Stable Preference Profiles (§5.1.2): Tracking consensus preference vectors
   (Pref*) across strict iteration thresholds to isolate systematic biases.

Reporting Features
------------------
* Automated Styling: Highlights peak column-wise performance with programmatic bold
  formatting via Pandas Styler configurations, enforcing clean black-and-white layouts.
* Data Integrity Fail-safes: Skips degraded evaluation sessions dynamically (e.g.,
  requiring a minimum of 5 valid judgment iterations per prompt sequence).

Main Interface Endpoints
------------------------
* get_awareness_report(): Cross-evaluates dual awareness markers across QA and retrieval.
* get_preference_report(): Tabulates high-threshold stable preference distributions.
* analyze_classification_percentages(): Extracts raw, unconditioned macro-level
  judgment percentages across claim, evidence, and fallback types.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from collections import Counter
from IPython.display import display
from knowledge_maze.tools.utils import load_json
from knowledge_maze.evaluation.metrics.tables.exhaustive_recall import ExhaustiveRecallEvaluator

load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')

QA_CONFIGS = {
    "DeepSeek V4 Flash": "results/baseline/deepseek_deepseek-v4-flash.json",
    "Gemini 3 Flash": "results/baseline/google_gemini-3-flash-preview.json",
    "Llama 4 Maverick": "results/baseline/meta-llama_llama-4-maverick.json",
    "GPT 5.4 Mini": "results/baseline/openai_gpt-5.4-mini.json",
    "Mistral 3.1 Medium": "results/baseline/mistralai_mistral-medium-3.1.json",
}

RETRIEVAL_CONFIGS = {
    "DeepSeek V4 Flash": "results/extract/deepseek_deepseek-v4-flash.json",
    "Gemini 3 Flash": "results/extract/google_gemini-3-flash-preview.json",
    "Llama 4 Maverick": "results/extract/meta-llama_llama-4-maverick.json",
    "GPT 5.4 Mini": "results/extract/openai_gpt-5.4-mini.json",
    "Mistral 3.1 Medium": "results/extract/mistralai_mistral-medium-3.1.json"
}


class MetricsReporter:
    def __init__(self, repo_path=REPO_PATH):
        self.repo_path = repo_path
        self._recall_evaluator = ExhaustiveRecallEvaluator()

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _calculate_qa_metrics(self, data):
        """Awareness and preference metrics from baseline QA data."""
        total_samples, total_iterations = 0, 0
        raw_ack_count, stable_ack_count = 0, 0
        stable_pref_counts = Counter()

        for _, sample in data.items():
            responses = sample.get("responses", [])
            judged = [r for r in responses if "judge_results" in r]
            if len(judged) < 5:
                continue

            total_samples += 1
            total_iterations += 5

            ack_flags = [r["judge_results"].get("dual_awareness", False) for r in judged]
            raw_ack_count += sum(ack_flags)
            if sum(ack_flags) >= 4:
                stable_ack_count += 1

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
            "ack":        pct(raw_ack_count, total_iterations),
            "stable_ack": pct(stable_ack_count, total_samples),
            "pref_claim": pct(stable_pref_counts["claim"], total_samples),
            "pref_mix":   pct(stable_pref_counts["mixed"], total_samples),
            "pref_evid":  pct(stable_pref_counts["evidence"], total_samples),
        }

    def _calculate_recall_metrics(self, data):
        """Claim / evidence / joint recall from retrieval data."""
        _, metrics = self._recall_evaluator.evaluate_dataset(data)
        return metrics

    def _styled(self, df, cols, fmt):
        """Bold the max value per column; apply number format."""
        def bold_max(s):
            return ['font-weight: bold' if v else '' for v in (s == s.max())]
        return df.style.apply(bold_max, subset=cols).format(fmt)

    # --------------------------------------------------
    # Public report functions
    # --------------------------------------------------

    def run_awareness_report(self, qa_configs=QA_CONFIGS, retrieval_configs=RETRIEVAL_CONFIGS):
        """
        Produces two tables matching the thesis §5.1.1 layout.

        Table 1 — Dual Awareness:
            Retrieval Aware | QA Aware | QA Aware*
        Table 2 — Recall Breakdown:
            Claim Recall | Evidence Recall | Joint Recall

        Args:
            qa_configs:        {DisplayName: relative_path} for baseline QA results
            retrieval_configs: {DisplayName: relative_path} for retrieval results
        """
        qa_metrics     = {n: self._calculate_qa_metrics(load_json(os.path.join(self.repo_path, p)))
                          for n, p in qa_configs.items()}
        recall_metrics = {n: self._calculate_recall_metrics(load_json(os.path.join(self.repo_path, p)))
                          for n, p in retrieval_configs.items()}

        # Table 1 — Dual Awareness
        df_awareness = pd.DataFrame([{
            "Model":           name,
            "AWARE (Retrieval)": recall_metrics[name]["joint_recall"] * 100,
            "AWARE (QA)":        qa_metrics[name]["ack"],
            "AWARE* (QA)":       qa_metrics[name]["stable_ack"],
        } for name in qa_configs])

        awareness_cols = ["AWARE (Retrieval)", "AWARE (QA)", "AWARE* (QA)"]
        print("--- Dual Awareness ---")
        display(self._styled(df_awareness, awareness_cols,
                             {c: "{:.1f}" for c in awareness_cols}))

        # Table 2 — Recall Breakdown
        df_recall = pd.DataFrame([{
            "Model":           name,
            "Claim Recall":    recall_metrics[name]["claim_recall"] * 100,
            "Evidence Recall": recall_metrics[name]["evidence_recall"] * 100,
            "Joint Recall":    recall_metrics[name]["joint_recall"] * 100,
        } for name in retrieval_configs])

        recall_cols = ["Claim Recall", "Evidence Recall", "Joint Recall"]
        print("\n--- Recall Breakdown ---")
        display(self._styled(df_recall, recall_cols,
                             {c: "{:.1f}" for c in recall_cols}))

        return df_awareness, df_recall

    def run_preference_report(self, model_configs=QA_CONFIGS):
        """
        Produces one table matching the thesis §5.1.2 layout.

        Table — Stable Preference Distribution:
            Pref*(cl) | Pref*(mix) | Pref*(ev)

        Args:
            model_configs: {DisplayName: relative_path} for baseline QA results
        """
        df = pd.DataFrame([{
            "Model":    name,
            "PREF*(cl)": self._calculate_qa_metrics(
                             load_json(os.path.join(self.repo_path, path)))["pref_claim"],
            "PREF*(mix)": self._calculate_qa_metrics(
                             load_json(os.path.join(self.repo_path, path)))["pref_mix"],
            "PREF*(ev)":  self._calculate_qa_metrics(
                             load_json(os.path.join(self.repo_path, path)))["pref_evid"],
        } for name, path in model_configs.items()])

        pref_cols = ["PREF*(cl)", "PREF*(mix)", "PREF*(ev)"]
        print("--- Stable Preference Distribution ---")
        display(self._styled(df, pref_cols, {c: "{:.2f}" for c in pref_cols}))

        return df


def get_awareness_report():
    met_rep = MetricsReporter()
    met_rep.run_awareness_report()


def get_preference_report():
    met_rep = MetricsReporter()
    met_rep.run_preference_report()


def analyze_classification_percentages():
    """
    Parses baseline test results and outputs a DataFrame summarizing
    the percentage of 'evidence' vs 'claim' classifications per model.
    """
    records = []

    test_results_baseline = {n: (load_json(os.path.join(REPO_PATH, p)))
                  for n, p in QA_CONFIGS.items()}

    for model_name, questions_dict in test_results_baseline.items():
        total_responses = 0
        evidence_count = 0
        claim_count = 0
        mixed_other_count = 0

        for q_id, q_data in questions_dict.items():
            responses = q_data.get('responses', [])
            for r in responses:
                judge = r.get('judge_results', {})
                if not judge:
                    continue  # Skip missing/invalid judge results

                classification = judge.get('classification', '').lower().strip()
                total_responses += 1

                if classification == 'evidence':
                    evidence_count += 1
                elif classification == 'claim':
                    claim_count += 1
                else:
                    # Captures 'mix', 'mixed', 'both', or any fallback case
                    mixed_other_count += 1

        # Calculate percentages safely
        if total_responses > 0:
            evidence_pct = (evidence_count / total_responses) * 100
            claim_pct = (claim_count / total_responses) * 100
            mixed_pct = (mixed_other_count / total_responses) * 100
        else:
            evidence_pct = claim_pct = mixed_pct = 0.0

        records.append({
            'Model': model_name,
            'Total Responses': total_responses,
            'Evidence (%)': round(evidence_pct, 2),
            'Claim (%)': round(claim_pct, 2),
            'Other (%)': round(mixed_pct, 2)
        })

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Sort by Evidence percentage descending for better readability
    df = df.sort_values(by='Evidence (%)', ascending=False).reset_index(drop=True)
    display(df)