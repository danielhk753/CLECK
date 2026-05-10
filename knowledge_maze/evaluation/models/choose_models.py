"""
Unified Validation Sweep Module
===============================

This module provides a rigorous framework for benchmarking the baseline reliability
of Large Language Models (LLMs) before subjecting them to complex reasoning or
conflict-based experiments. It is specifically designed to identify a
"Grounded Cohort" of models by measuring their ability to extract factual
information consistently across multiple stochastic attempts.

Key Features:
-------------
* Stochastic Consistency Testing: Executes N=3 independent trials per sample at
  a moderate temperature (T=0.4) to reveal model "shakiness" or internal biases.
* Hybrid Validation Engine: Combines deterministic string cleaning, numeric
  extraction, Jaccard-style IoU (Intersection over Union), and semantic embedding
  similarity (all-MiniLM-L6-v2) to verify answers.
* Global Judge Recovery: Automatically batches deterministic failures and sends
  them to a high-reasoning "Judge" model (e.g., Gemini 3.1 Flash) for final
  semantic arbitration, ensuring rephrased but correct answers are identified.
* Stability Profiling: Categorizes performance into four "Stability Buckets"
  (3/3, 2/3, 1/3, 0/3) to distinguish between accidental "lucky" guesses
  and robust grounding.
* Two-Tiered Metrics: Calculates General Reliability (>= 2/3 majority vote)
  and Strict Consistency (3/3 perfect runs) to determine a model's suitability
  for high-stakes research.

Logic Flow:
-----------
1. Batch Inference: Collects all stochastic attempts for both Evidence and
   Claim contexts in a high-throughput async loop.
2. Deterministic Filtering: Uses fast text-matching utilities to catch high-
   confidence matches immediately.
3. LLM-as-a-Judge: Resolves nuanced semantic equivalencies that automated
   metrics might miss.
4. Reporting: Generates a pandas DataFrame and an interactive HTML dashboard
   summarizing the "Groundedness" of the entire model sweep.

Metrics:
--------
* Mean Success: Average accuracy across all trials and contexts.
* General Reliability (>= 2/3): Primary filter; percentage of samples where
  the model's majority vote aligns with the ground truth.
* Strict Consistency (3/3): Gold standard; percentage of samples where
  the model exhibited zero variance in its factual grounding.
"""

import re
import os
import time
import yaml
import unicodedata
import pandas as pd
from typing import List, Dict
from difflib import SequenceMatcher
from dotenv.main import load_dotenv
from IPython.display import display, HTML
from sentence_transformers import SentenceTransformer, util

from knowledge_maze.tools.api_tools import OpenRouterAPI
from knowledge_maze.evaluation.utils.select_subset import get_stratified_subset
from knowledge_maze.evaluation.utils.prompt_templates import BASELINE_PROMPT

load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')

class UnifiedValidationSweep:
    def __init__(self, model_list: List[str], judge_model: str = "google/gemini-3.1-flash-lite-preview"):
        self.model_list = model_list
        self.results = {}  # Stores consistency scores (0.0 to 1.0)
        self.full_summaries = []

        self.judge_api = OpenRouterAPI(
            system_prompt=(
                "You are a strict academic evaluator. Your task is to evaluate a model's prediction (PRED) "
                "against a Ground Truth (GT).\n\n"
                "1. **Extract**: Identify the core final answer from the PRED text.\n"
                "2. **Analyze**: Compare this core answer to the GT. Ignore casing and minor formatting "
                "but ensure factual and semantic equivalence.\n"
                "3. **Verdict**: Conclude whether the prediction is correct ('True') or incorrect ('False')."
            ),
            model=judge_model,
            temperature=0.0,
            output_format='cot'  # Enables structured reasoning + answer fields
        )

        self.validate_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.typo_threshold = 0.9
        self.iou_threshold = 1.0
        self.embedding_threshold = 0.8

    async def run_sweep(self, question_data: List[Dict], paragraphs: Dict, sample_limit: int = 50):
        actual_limit = min(len(question_data), sample_limit)

        sweep_data = get_stratified_subset(question_data, actual_limit)

        num_tries = 3

        print(f"🚀 Starting CONSISTENCY sweep: {len(self.model_list)} models, {actual_limit} samples.\n")

        for model_name in self.model_list:
            print(f"🔍 Evaluating Consistency: {model_name}...")
            model_api = OpenRouterAPI(system_prompt=BASELINE_PROMPT, model=model_name, temperature=0.4)
            start_time = time.time()

            # 1. BATCH INFERENCE: Get all 3x tries for Ev and Cl in one go
            all_prompts = []
            for s in sweep_data:
                ev_text = f"CONTEXT: {paragraphs[s['Evidence paragraph']]}\nQUESTION: {s['Question']}\nANSWER: "
                cl_text = f"CONTEXT: {paragraphs[s['Claim paragraph']]}\nQUESTION: {s['Question']}\nANSWER: "
                for _ in range(num_tries):
                    all_prompts.extend([ev_text, cl_text])

            raw_results = await model_api.async_run(all_prompts, desc=f"Inferencing {model_name}")

            # 2. DETERMINISTIC CHECK & FAILURE COLLECTION
            # We store metadata so we know where to put the judge's answer later
            judge_requests = []  # List of prompts for the judge
            judge_metadata = []  # List of (sample_idx, try_idx, type_key ['ev'|'cl'])

            # Initialize match containers with deterministic results
            # These will be lists of lists: [[bool, bool, bool], [bool, bool, bool], ...]
            ev_match_grid = []
            cl_match_grid = []

            for i in range(actual_limit):
                sample = sweep_data[i]
                sample_start = i * (num_tries * 2)

                s_ev_matches = []
                s_cl_matches = []

                for t in range(num_tries):
                    try_start = sample_start + (t * 2)
                    pred_ev = raw_results[try_start]
                    pred_cl = raw_results[try_start + 1]

                    # Check Evidence
                    is_ev_match = self.are_equal(sample['Answer']['Evidence'], pred_ev)
                    s_ev_matches.append(is_ev_match)
                    if not is_ev_match:
                        judge_requests.append(
                            f"Q: {sample['Question']}\nGT: {sample['Answer']['Evidence']}\nPRED: {pred_ev}")
                        judge_metadata.append((i, t, 'ev'))

                    # Check Claim
                    is_cl_match = self.are_equal(sample['Answer']['Claim'], pred_cl)
                    s_cl_matches.append(is_cl_match)
                    if not is_cl_match:
                        judge_requests.append(
                            f"Q: {sample['Question']}\nGT: {sample['Answer']['Claim']}\nPRED: {pred_cl}")
                        judge_metadata.append((i, t, 'cl'))

                ev_match_grid.append(s_ev_matches)
                cl_match_grid.append(s_cl_matches)

            # 3. GLOBAL JUDGE RECOVERY (The Speed Fix)
            if judge_requests:
                print(f"⚖️ Sending {len(judge_requests)} deterministic failures to Judge for global review...")
                judge_results = await self.judge_api.async_run(judge_requests, desc="Judge Batch Review")

                for (sample_idx, try_idx, type_key), ruling in zip(judge_metadata, judge_results):
                    try:
                        if "true" in str(ruling.answer.lower()):
                            if type_key == 'ev':
                                ev_match_grid[sample_idx][try_idx] = True
                            else:
                                cl_match_grid[sample_idx][try_idx] = True
                    except AttributeError:
                        pass

            # 4. CALCULATE FINAL SCORES
            final_scores = []
            for i in range(actual_limit):
                ev_score = sum(ev_match_grid[i]) / num_tries
                cl_score = sum(cl_match_grid[i]) / num_tries
                # Final result for the sample is the average of both contexts
                final_scores.append((ev_score + cl_score) / 2)

            self.results[model_name] = final_scores

            # Metrics logging...
            self.full_summaries.append({
                "Model": model_name,
                "Mean Success (%)": (sum(final_scores) / actual_limit) * 100,
                "Sec/Sample": (time.time() - start_time) / actual_limit
            })

        self.display_dashboard(self.generate_floor_report())
        return self.generate_floor_report()

    async def _judge_check_tries(self, matches, tries, true_val, question, key):
        """Internal helper to ask Judge only about failed attempts."""
        for i, matched in enumerate(matches):
            if not matched:
                ruling = await self.judge_api.async_run([f"Q: {question}\nGT: {true_val}\nPRED: {tries[i][key]}"],
                                                        desc="⚖️ Reviewing Try")
                if "true" in str(ruling[0]).lower():
                    matches[i] = True
        return matches

    # -------------------------------------------------------------------------
    # 3. TEXT COMPARISON UTILITIES (From LLMValidator)
    # -------------------------------------------------------------------------
    def clean(self, text: str):
        text = str(text).lower()
        text = "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        text = re.sub(r'(?:\d+[\.\)]|\(\d+\)|[†\-\*])\s*', ' ', text)
        text = re.sub(r'[,;."]', ' ', text)
        return " ".join(text.split())

    def get_set(self, text: str):
        return set(self.clean(text).split())

    def _is_numeric(self, s: str):
        s = str(s).replace(',', '').strip()
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        if not match: return None
        remaining_text = re.sub(r'[a-zA-Z%$\. ]', '', s.replace(match.group(), '', 1))
        return float(match.group()) if len(remaining_text) == 0 else None

    def calculate_iou(self, str_a: str, str_b: str):
        set_a, set_b = self.get_set(str_a), self.get_set(str_b)
        if not set_a and not set_b: return 1.0
        if not set_a or not set_b: return 0.0

        intersection_count = 0
        temp_set_b = list(set_b)
        for item_a in set_a:
            best_score = 0
            best_match_idx = -1
            for i, item_b in enumerate(temp_set_b):
                score = 1.0 if item_a == item_b else SequenceMatcher(None, item_a, item_b).ratio()
                if score > best_score:
                    best_score, best_match_idx = score, i
            if best_score >= self.typo_threshold:
                intersection_count += 1
                if best_match_idx != -1: temp_set_b.pop(best_match_idx)
        return intersection_count / (len(set_a) + len(set_b) - intersection_count)

    def get_semantic_similarity(self, a: str, b: str):
        try:
            emb_a = self.validate_model.encode(a, convert_to_tensor=True)
            emb_b = self.validate_model.encode(b, convert_to_tensor=True)
            return util.pytorch_cos_sim(emb_a, emb_b).item()
        except:
            return 0.0

    def are_equal(self, a: str, b: str) -> bool:
        num_a, num_b = self._is_numeric(a), self._is_numeric(b)
        if num_a is not None or num_b is not None:
            return num_a == num_b
        if self.calculate_iou(a, b) >= self.iou_threshold:
            return True
        return self.get_semantic_similarity(a, b) >= self.embedding_threshold

    # -------------------------------------------------------------------------
    # 4. REPORTING & DASHBOARD
    # -------------------------------------------------------------------------
    def generate_floor_report(self):
        model_metrics = []

        for model_name, scores in self.results.items():
            total_samples = len(scores)
            # Metric 1: Mean Success Rate (All attempts)
            mean_success = (sum(scores) / total_samples) * 100

            # Stability Buckets
            s3 = len([s for s in scores if s >= 0.99])  # 3/3
            s2 = len([s for s in scores if 0.60 <= s < 0.99])  # 2/3
            s1 = len([s for s in scores if 0.10 <= s < 0.60])  # 1/3
            s0 = len([s for s in scores if s < 0.10])  # 0/3

            # NEW: Two-Tiered Consistency
            strict_consistency = (s3 / total_samples) * 100  # Exactly 3/3
            general_reliability = ((s3 + s2) / total_samples) * 100  # At least 2/3

            model_metrics.append({
                "Model": model_name,
                "Mean Success (%)": mean_success,
                "Strict Consistency (%)": strict_consistency,
                "General Reliability (%)": general_reliability,
                "3/3": s3,
                "2/3": s2,
                "1/3": s1,
                "0/3": s0,
                "Total": total_samples
            })

        # Sorting by Mean Success for the primary view
        return pd.DataFrame(model_metrics).sort_values(by="Mean Success (%)", ascending=False)

    def display_dashboard(self, df_report: pd.DataFrame):
        actual_n = df_report['Total'].iloc[0] if not df_report.empty else 0

        rows_html = ""
        for _, row in df_report.iterrows():
            s_rate = row['Mean Success (%)']
            s_color = "#1b5e20" if s_rate >= 90 else "#2e7d32" if s_rate >= 80 else "#c62828"

            rows_html += f"""
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 15px; text-align: left;">
                    <b style="font-size: 1.1em; color: #222;">{row['Model']}</b>
                </td>
                <td style="padding: 15px; text-align: center;">
                    <span style="font-size: 1.3em; font-weight: 900; color: {s_color};">{s_rate:.1f}%</span>
                </td>
                <td style="padding: 15px; text-align: center; color: #1565c0; font-weight: 700;">
                    {row['General Reliability (%)']:.1f}%
                </td>
                <td style="padding: 15px; text-align: center; color: #555; font-weight: 600;">
                    {row['Strict Consistency (%)']:.1f}%
                </td>
                <td style="padding: 15px; text-align: center;">
                    <div style="display: flex; gap: 4px; justify-content: center; font-family: monospace;">
                        <span title="3/3: Perfect" style="background: #e8f5e9; color: #2e7d32; padding: 2px 8px; border-radius: 4px; border: 1px solid #a5d6a7;"><b>{row['3/3']}</b></span>
                        <span title="2/3: General" style="background: #e3f2fd; color: #1565c0; padding: 2px 8px; border-radius: 4px; border: 1px solid #90caf9;"><b>{row['2/3']}</b></span>
                        <span title="1/3: Lucky" style="background: #fff3e0; color: #ef6c00; padding: 2px 8px; border-radius: 4px; border: 1px solid #ffcc80;">{row['1/3']}</span>
                        <span title="0/3: Failed" style="background: #ffebee; color: #c62828; padding: 2px 8px; border-radius: 4px; border: 1px solid #ef9a9a;">{row['0/3']}</span>
                    </div>
                </td>
            </tr>
            """

        dashboard_html = f"""
        <div style="background: white; padding: 25px; border: 1px solid #ddd; border-radius: 8px; font-family: sans-serif; max-width: 1000px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 20px; border-bottom: 2px solid #333; padding-bottom: 10px;">
                <div>
                    <h2 style="margin: 0; color: #333; text-transform: uppercase; letter-spacing: 1px;">Baseline Reliability Sweep</h2>
                    <p style="margin: 0; color: #666; font-size: 0.9em;">N={actual_n} Samples | 3 Tries per Context | Temp 0.4</p>
                </div>
            </div>

            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background: #fdfdfd; text-transform: uppercase; font-size: 0.75em; color: #777; border-bottom: 1px solid #333;">
                        <th style="padding: 10px; text-align: left;">Model Architecture</th>
                        <th style="padding: 10px; text-align: center;">Mean Success</th>
                        <th style="padding: 10px; text-align: center; color: #1565c0;">General (≥2/3)</th>
                        <th style="padding: 10px; text-align: center;">Strict (3/3)</th>
                        <th style="padding: 10px; text-align: center;">Stability Profile (3 to 0)</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>

            <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 4px; font-size: 0.85em; color: #444;">
                <b>Metric Glossary:</b><br>
                • <b>Mean Success:</b> Average score across all trials.<br>
                • <b>General Reliability:</b> Percentage of samples where the model was correct at least 66% of the time.<br>
                • <b>Strict Consistency:</b> Percentage of samples where the model was correct 100% of the time.
            </div>
        </div>
        """
        display(HTML(dashboard_html))


class KnowledgeMazeRunner:
    def __init__(self, limit):
        self.config = self._load_config(os.path.join(REPO_PATH, "knowledge_maze\evaluation\models\models_config.yaml"))
        self.models = self.config.get('models_to_evaluate', [])
        self.judge = self.config.get('judge_model', '')
        self.limit = limit

        # Initialize your sweep engine
        self.sweep = UnifiedValidationSweep(
            model_list=self.models,
            judge_model=self.judge
        )

    def _load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    async def execute(self, q_data, p_data):
        return await self.sweep.run_sweep(q_data, p_data, sample_limit=self.limit)
