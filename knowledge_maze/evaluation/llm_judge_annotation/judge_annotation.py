"""
LLM Judge Evaluation and Human-in-the-Loop Audit Framework

This module provides an end-to-end framework for parsing model evaluation logs,
curating representative validation sets, distributing data packages to human
annotators, and conducting interactive audits of LLM Judge performance.

It is designed to address highly skewed label distributions (e.g., knowledge
conflicts) and to provide rigorous statistical corrections for metric reporting.

Key Methodologies Implemented:
------------------------------
1. Square-Root Stratified Curation:
   Balances structural data evaluation by allocating validation samples proportional
   to the square root of their raw subclass population size. This ensures rare judge
   verdicts (e.g., abstentions) are sufficiently sampled for error analysis without
   completely washing out dominant classes (e.g., concise claims). Residuals are
   apportioned via the Largest Remainder Method.

2. Pairwise Inter-Rater Overlap Splitting:
   Partitions the validation set among X annotators such that every unique pair of
   annotators shares exactly Y overlapping items for inter-rater reliability analysis
   (e.g., Cohen's Kappa / Fleiss' Kappa), while uniformly distributing the remaining
   items as blind uniques.

3. Population-Weighted (Normalized) Evaluation:
   Applies a post-stratification adjustment during audit processing. Because rare
   subclasses are heavily oversampled during curation, simple sample accuracy is
   statistically biased. The framework computes intra-stratum accuracies and builds a
   weighted sum matching the true un-curated population distributions to generate a
   true, unbiased "Normalized Accuracy" score.

Architecture Components:
------------------------
* JudgeDatasetManager:
    Ingests deeply nested experiment maps, flattens candidate outputs, runs square-root
    stratified sampling, computes combinatorial overlap splits, and exports structured
    JSON tracking tables for human annotation.
* JudgeAuditTagger:
    An interactive, state-preserving Jupyter/IPython Notebook widget interface.
    Provides high-contrast viewports for reviewing raw traces alongside judge verdicts.
    Tracks real-time evaluation states across auditing sessions and generates 100%
    percentage-normalized performance diagnostic charts.

Usage Example:
--------------
    # 1. Initialize and Curate
    manager = JudgeDatasetManager(raw_experiment_logs)
    validation_pool = manager.create_validation_set(total_target=400)

    # 2. Split for annotation teams
    batches = manager.split_for_annotators(x_annotators=4, y_shared=33)

    # 3. Audit judge decisions via Jupyter Notebook UI
    auditor = JudgeAuditTagger(validation_pool, save_file="audit_checkpoint.json")
    auditor.display()

    # 4. Plot population-corrected analytics
    auditor.plot_results(true_dist=manager.true_distribution)
"""

import os
import json
import math
import random
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import ipywidgets as widgets
import matplotlib.pyplot as plt
from dotenv.main import load_dotenv
from collections import defaultdict
from IPython.display import display, HTML, clear_output


load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')

class JudgeDatasetManager:
    """
    A unified class to manage, curate, and partition LLM Judge evaluation data.
    Implements Square-Root Stratified Curation and Pairwise Inter-Rater Overlap Splitting.
    """

    def __init__(self, experiment_data_map):
        self.experiment_data_map = experiment_data_map
        self.all_candidates = []
        self.validation_set = []
        self.true_distribution = {}  # Added to store raw population counts for normalization

        # Ingest raw dataset logs on initialization
        self._ingest_raw_responses()

    def _ingest_raw_responses(self):
        """Parses the nested experiment maps into a flattened candidate list."""
        for exp_type, model_results_dict in self.experiment_data_map.items():
            for model_name, model_dict in model_results_dict.items():
                for sample_id, sample_data in model_dict.items():

                    question = sample_data.get('question', '')
                    gt_answers = sample_data.get('gt_answers', {})
                    responses = sample_data.get('responses', [])

                    for r in responses:
                        judge_res = r.get('judge_results')
                        if not judge_res or not isinstance(judge_res, dict):
                            continue

                        if 'classification' not in judge_res or 'dual_awareness' not in judge_res:
                            continue

                        classification = str(judge_res.get('classification', '')).strip().lower()
                        ack = bool(judge_res.get('dual_awareness', False))
                        explanation = judge_res.get('explanation', '')
                        raw_response = r.get('raw_response', '')
                        iteration_id = r.get('iteration_id')

                        if classification not in ['memory', 'claim', 'evidence', 'abstain', 'other']:
                            classification = 'other'

                        self.all_candidates.append({
                            'sample_id': sample_id,
                            'experiment_type': exp_type,
                            'model': model_name,
                            'question': question,
                            'gt_answers': gt_answers,
                            'iteration_id': iteration_id,
                            'raw_response': raw_response,
                            'judge_ack': ack,
                            'judge_classification': classification,
                            'judge_explanation': explanation
                        })

    def create_validation_set(self, total_target=500, seed=48):
        """
        Curates a validation pool using a Square-Root Stratified Allocation framework.
        """
        random.seed(seed)

        # 1. Stratify into buckets[strat_key][experiment_type] = [samples]
        buckets = defaultdict(lambda: defaultdict(list))
        for c in self.all_candidates:
            strat_key = (c['judge_ack'], c['judge_classification'])
            buckets[strat_key][c['experiment_type']].append(c)

        # 2. Shuffle sub-lists to mix models
        for strat_key in buckets:
            for exp_type in buckets[strat_key]:
                random.shuffle(buckets[strat_key][exp_type])

        # 3. Compute Square-Root Allocations for the 10 Subclasses
        subclass_totals = {k: sum(len(lst) for lst in exp_dict.values()) for k, exp_dict in buckets.items()}
        self.true_distribution = subclass_totals  # Extracted true baseline counts

        sqrt_sum = sum(math.sqrt(count) for count in subclass_totals.values())

        allocations = {}
        for strat_key, count in subclass_totals.items():
            proportion = math.sqrt(count) / sqrt_sum
            allocations[strat_key] = min(count, int(proportion * total_target))

        # Handle rounding remainders via Largest Remainder Method
        remainder = total_target - sum(allocations.values())
        if remainder > 0:
            sorted_keys = sorted(
                subclass_totals.keys(),
                key=lambda k: ((math.sqrt(subclass_totals[k]) / sqrt_sum) * total_target) % 1,
                reverse=True
            )
            for key in sorted_keys:
                if remainder <= 0:
                    break
                if allocations[key] < subclass_totals[key]:
                    allocations[key] += 1
                    remainder -= 1

        # 4. Target-Driven Selection
        self.validation_set = []
        for strat_key, target_count in allocations.items():
            exp_dict = buckets[strat_key]
            pulled_for_subclass = 0

            while pulled_for_subclass < target_count:
                available_exps = [exp for exp, samples in exp_dict.items() if samples]
                if not available_exps:
                    break

                weights = [3 if exp == 'main' else 1 for exp in available_exps]
                chosen_exp = random.choices(available_exps, weights=weights, k=1)[0]

                self.validation_set.append(exp_dict[chosen_exp].pop())
                pulled_for_subclass += 1

        random.shuffle(self.validation_set)
        self._print_curation_diagnostics()
        return self.validation_set

    def split_for_annotators(self, x_annotators, y_shared, seed=42):
        """
        Splits the validation set such that every unique pair of annotators shares
        exactly Y samples, and all remaining samples are distributed uniquely.
        """
        random.seed(seed)
        if not self.validation_set:
            raise ValueError("Validation set is empty. Run create_validation_set() first.")

        num_pairs = math.comb(x_annotators, 2)
        total_shared_required = num_pairs * y_shared
        total_samples = len(self.validation_set)

        # Combinatorial Constraint Validation
        if total_shared_required > total_samples:
            raise ValueError(
                f"Combinatorial Conflict: Requesting {x_annotators} annotators to share {y_shared} samples "
                f"per pair requires a minimum of {total_shared_required} total overlap samples, but "
                f"the curated validation set only contains {total_samples} samples."
            )

        # Create a copy of the pool to slice up
        pool = list(self.validation_set)

        # Initialize dictionary to hold final batches
        annotator_batches = {f"Annotator_{i + 1}": [] for i in range(x_annotators)}

        # 1. Distribute Shared Pairwise Overlaps
        pairs = list(itertools.combinations(range(x_annotators), 2))
        for idx_a, idx_b in pairs:
            # Extract Y unique samples from pool for this pair
            shared_samples = [pool.pop() for _ in range(y_shared)]

            # Map back to string labels
            annotator_batches[f"Annotator_{idx_a + 1}"].extend(shared_samples)
            annotator_batches[f"Annotator_{idx_b + 1}"].extend(shared_samples)

        # 2. Distribute Remaining Uniques Uniformly
        remaining_count = len(pool)
        uniques_per_annotator = remaining_count // x_annotators

        for i in range(x_annotators):
            unique_samples = [pool.pop() for _ in range(uniques_per_annotator)]
            annotator_batches[f"Annotator_{i + 1}"].extend(unique_samples)

        # 3. Cleanly distribute any leftover odd remainder rows due to integer division floor
        leftover_idx = 0
        while pool:
            annotator_batches[f"Annotator_{leftover_idx + 1}"].append(pool.pop())
            leftover_idx = (leftover_idx + 1) % x_annotators

        # Shuffle individual batches to blind the annotator from the overlap pattern
        for annotator in annotator_batches:
            random.shuffle(annotator_batches[annotator])

        self._print_splitting_diagnostics(x_annotators, y_shared, annotator_batches, uniques_per_annotator)

        # Define the target export directory based on your repository layout
        OUTPUT_DIR = os.path.join(REPO_PATH, "knowledge_maze/evaluation/llm_judge_annotation/data_to_annotate")

        # Create the directory safely if it doesn't exist yet
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print(f"\nWriting annotator validation batches to: {OUTPUT_DIR}")

        # Iterate and save each batch
        for annotator, batch in annotator_batches.items():
            filename = f"{annotator.lower()}.json"
            file_path = os.path.join(OUTPUT_DIR, filename)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(batch, f, indent=4, ensure_ascii=False)

            print(f" -> Successfully saved {len(batch)} rows to {filename}")

        print("\nAll annotation packages exported successfully. Ready for manual audit.")

        return annotator_batches

    def _print_curation_diagnostics(self):
        print(f"\n=== Curation Completed: Gathered {len(self.validation_set)} Samples ===")
        final_dist = defaultdict(int)
        for item in self.validation_set:
            final_dist[(item['judge_ack'], item['judge_classification'])] += 1

        # Updated to show a comparative side-by-side view of true vs curated counts
        print(f"{'Subclass Profile':<32} | {'True Population':<15} | {'Curated Subsample':<17}")
        print("-" * 70)
        for k, count in sorted(final_dist.items()):
            profile_lbl = f"Ack: {str(k[0]):<5} | Class: {k[1]}"
            true_count = self.true_distribution.get(k, 0)
            print(f"{profile_lbl:<32} | {true_count:<15} | {count:<17}")

    def _print_splitting_diagnostics(self, x, y, batches, uniques_per):
        print(f"\n=== Partition Completed for {x} Annotators (Pairwise Target Overlap: {y}) ===")
        for name, samples in batches.items():
            print(f"{name}: Assigned {len(samples)} total samples (~{uniques_per} unique, {(x - 1) * y} shared)")
        print("=========================================================================")


class JudgeAuditTagger:
    def __init__(self, validation_set, save_file="judge_audit_progress.json"):
        self.data = validation_set
        self.save_file = save_file
        self.results = self._load_progress()

        self.current_idx = 0
        self.sample_pool = []
        self.out = widgets.Output()

        # Progress bar configuration
        self.progress = widgets.IntProgress(
            min=0,
            layout=widgets.Layout(width='98%', height='18px', border='1px solid #666', margin='0 auto')
        )
        self.progress_label = widgets.HTML(value="")
        self.progress_box = widgets.VBox([self.progress, self.progress_label],
                                         layout=widgets.Layout(width='100%', margin='10px 0'))

        # Session size control
        self.sample_size_input = widgets.BoundedIntText(
            value=50, min=1, max=len(self.data), step=1,
            description='Samples to Audit:', style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.start_btn = widgets.Button(description="🔍 AUDIT JUDGE", button_style='primary')
        self.start_btn.on_click(lambda x: self.run(self.sample_size_input.value))

        # Control interface buttons
        self.save_exit_btn = widgets.Button(description="⏹️ SAVE & EXIT", style={'button_color': '#bdc3c7'},
                                            layout=widgets.Layout(flex='1 1 auto', margin='0 4px', height='45px'))
        self.save_exit_btn.on_click(lambda x: self._save_and_exit())

    def display(self):
        display(self.out)
        with self.out:
            clear_output()
            display(widgets.HBox([self.sample_size_input, self.start_btn], layout=widgets.Layout(margin='15px 0')))

    def _get_unique_key(self, sample):
        return f"{sample['sample_id']}_{sample['model']}_{sample['experiment_type']}_{sample['iteration_id']}"

    def _load_progress(self):
        if os.path.exists(self.save_file):
            with open(self.save_file, 'r') as f:
                return json.load(f)
        return []

    def _save_progress(self):
        with open(self.save_file, 'w') as f:
            json.dump(self.results, f, indent=4)

    def _show_sample(self):
        with self.out:
            clear_output(wait=True)
            if self.current_idx >= len(self.sample_pool):
                print("🏁 Audit session complete! Save file updated.")
                return

            sample = self.sample_pool[self.current_idx]
            gt = sample.get('gt_answers', {})

            # Format classification color tags
            class_colors = {
                'memory': '#e8f5e9; color: #1b5e20; border: 1px solid #c8e6c9',
                'claim': '#fff3e0; color: #e65100; border: 1px solid #ffe0b2',
                'evidence': '#ffebee; color: #c62828; border: 1px solid #ffcdd2',
                'abstain': '#f3e5f5; color: #4a148c; border: 1px solid #e1bee7',
                'other': '#eceff1; color: #263238; border: 1px solid #cfd8dc'
            }
            c_style = class_colors.get(sample['judge_classification'], class_colors['other'])
            ack_style = "#e8f5e9; color: #1b5e20; border: 1px solid #c8e6c9" if sample[
                'judge_ack'] else "#ffebee; color: #c62828; border: 1px solid #ffcdd2"

            style = """
            <style>
                .judge-audit-container { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f8fafc; padding: 20px; border-radius: 12px; }
                .split-container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                .audit-card { background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03); display: flex; flex-direction: column; }
                .meta-header { font-size: 0.85em; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #f1f5f9; padding-bottom: 8px; margin-bottom: 12px; font-weight: 600; }

                /* Ground Truth Display Layout */
                .gt-box { display: flex; gap: 8px; margin: 12px 0; flex-wrap: wrap; }
                .gt-item { font-size: 0.85em; padding: 4px 10px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; color: #334155; }

                /* Unified Component Viewports for Visual Balance */
                .readable-viewport {
                    height: 320px; 
                    overflow-y: auto; 
                    border-radius: 8px; 
                    padding: 16px; 
                    font-size: 0.95em; 
                    line-height: 1.6;
                    box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.02);
                }

                /* High-Contrast Monospace Frame for Raw Data Trace */
                .code-box { 
                    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; 
                    background: #0f172a; 
                    color: #f1f5f9; 
                    border: 1px solid #1e293b;
                    white-space: pre-wrap; 
                    word-break: break-word;
                }

                /* Clean Typography Block for Explanations */
                .explanation-box { 
                    background: #fdfdfd; 
                    border: 1px solid #e2e8f0;
                    border-left: 5px solid #4f46e5; 
                    color: #1e293b;
                }

                .verdict-line { font-size: 1em; font-weight: 600; color: #334155; margin: 10px 0; display: flex; gap: 12px; align-items: center; }
                .badge { font-size: 0.8em; font-weight: 700; padding: 4px 10px; border-radius: 6px; display: inline-block; text-transform: uppercase; letter-spacing: 0.3px; }
                .section-title { font-size: 0.8em; text-transform: uppercase; color: #64748b; margin-top: 14px; margin-bottom: 6px; font-weight: 700; letter-spacing: 0.5px; }
            </style>
            """

            html_layout = f"""
            {style}
            <div class="judge-audit-container">
                <div class="split-container">

                    <div class="audit-card">
                        <div class="meta-header">
                            <b>Model:</b> {sample['model']} | <b>Exp:</b> {sample['experiment_type']}
                        </div>
                        <div style="font-weight: 600; color: #1e293b; font-size: 1.05em; margin-bottom: 6px;">Q: {sample['question']}</div>

                        <div class="gt-box">
                            <span class="gt-item"><b>Memory:</b> {gt.get('Memory', 'N/A')}</span>
                            <span class="gt-item"><b>Claim:</b> {gt.get('Claim', 'N/A')}</span>
                            <span class="gt-item"><b>Evidence:</b> {gt.get('Evidence', 'N/A')}</span>
                        </div>

                        <div class="section-title">Raw Model Response:</div>
                        <div class="code-box readable-viewport">{sample['raw_response']}</div>
                    </div>

                    <div class="audit-card">
                        <div class="meta-header">🤖 LLM Judge Verdict Appraisal</div>

                        <div class="verdict-line" style="margin-top: 5px;">
                            <span>Classification:</span>
                            <span class="badge" style="{c_style}">{sample['judge_classification']}</span>
                        </div>

                        <div class="verdict-line" style="margin-bottom: 15px;">
                            <span>Dual Answer Awareness:</span>
                            <span class="badge" style="{ack_style}">{str(sample['judge_ack'])}</span>
                        </div>

                        <div class="section-title">Judge Explanation:</div>
                        <div class="explanation-box readable-viewport">{sample['judge_explanation']}</div>
                    </div>

                </div>
            </div>
            """
            display(HTML(html_layout))

            # Interactive evaluation input configurations
            btn_layout = widgets.Layout(flex='1 1 auto', width='auto', margin='0 4px', height='48px')

            b_correct = widgets.Button(description="✅ JUDGE CORRECT", style={'button_color': '#2ecc71'},
                                       layout=btn_layout)
            b_w_class = widgets.Button(description="❌ WRONG CLASSIFICATION", style={'button_color': '#e67e22'},
                                       layout=btn_layout)
            b_w_ack = widgets.Button(description="❌ WRONG AWARENESS", style={'button_color': '#e74c3c'},
                                     layout=btn_layout)
            b_w_both = widgets.Button(description="💥 BOTH WRONG",
                                      style={'button_color': '#34495e', 'text_color': 'white'}, layout=btn_layout)

            b_correct.on_click(lambda x: self._handle_click("correct"))
            b_w_class.on_click(lambda x: self._handle_click("error_classification"))
            b_w_ack.on_click(lambda x: self._handle_click("error_acknowledgement"))
            b_w_both.on_click(lambda x: self._handle_click("error_both"))

            display(self.progress_box)
            display(widgets.HBox([b_correct, b_w_class, b_w_ack, b_w_both, self.save_exit_btn],
                                 layout=widgets.Layout(width='100%', margin='5px 0')))

    def run(self, requested_size):
        tagged_keys = {self._get_unique_key(res) for res in self.results}
        available = [s for s in self.data if self._get_unique_key(s) not in tagged_keys]

        if not available:
            with self.out:
                clear_output()
                print("🎉 All judge samples have been successfully audited!")
            return

        random.shuffle(available)
        actual_size = min(requested_size, len(available))
        self.sample_pool = available[:actual_size]

        self.current_idx = 0
        self.progress.max = len(self.sample_pool)
        self.progress.value = 0
        self.progress_label.value = f"<div style='text-align: center;'>Session Progress: 0 / {len(self.sample_pool)}</div>"

        with self.out:
            clear_output()
            self._show_sample()

    def _handle_click(self, status):
        sample = self.sample_pool[self.current_idx]

        # Append target metadata matching your auditing profile tracking properties
        self.results.append({
            "sample_id": sample['sample_id'],
            "model": sample['model'],
            "experiment_type": sample['experiment_type'],
            "iteration_id": sample['iteration_id'],
            "judge_classification": sample['judge_classification'],
            "judge_ack": sample['judge_ack'],
            "audit_status": status
        })
        self._save_progress()

        self.current_idx += 1
        self.progress.value += 1
        self.progress_label.value = f"<div style='text-align: center; font-size: 0.9em; margin-top: 5px;'>Session Progress: <b>{self.progress.value} / {self.progress.max}</b></div>"

        self._show_sample()

    def _save_and_exit(self):
        self._save_progress()
        with self.out:
            clear_output()
            print(f"💾 Progress saved! Evaluated {self.current_idx} judge assertions this session.")
            print(f"Total entries audited: {len(self.results)} / {len(self.data)}")
            display(widgets.HBox([self.sample_size_input, self.start_btn], layout=widgets.Layout(margin='15px 0')))

    def plot_results(self, true_dist):
        if not self.results:
            print("⚠️ No judge audit data tagged yet.")
            return

        # --- 1. Data Preparation ---
        df_tagged = pd.DataFrame(self.results)
        total_samples = len(df_tagged)
        total_correct = len(df_tagged[df_tagged['audit_status'] == 'correct'])
        total_failed = total_samples - total_correct

        # Calculate Raw Sample Accuracy
        accuracy_rate = (total_correct / total_samples * 100) if total_samples > 0 else 0

        # --- 1.5. Calculate Population-Normalized Accuracy ---
        total_true_pop = sum(true_dist.values())

        weighted_acc_sum = 0.0
        total_weight_used = 0.0

        grouped_subclasses = df_tagged.groupby(['judge_ack', 'judge_classification'])

        for (ack, classification), group in grouped_subclasses:
            subclass_key = (ack, classification)

            if subclass_key in true_dist:
                subclass_total = len(group)
                subclass_correct = len(group[group['audit_status'] == 'correct'])
                subclass_acc = subclass_correct / subclass_total
                subclass_weight = true_dist[subclass_key] / total_true_pop

                weighted_acc_sum += subclass_acc * subclass_weight
                total_weight_used += subclass_weight

        if total_weight_used > 0:
            norm_accuracy_rate = (weighted_acc_sum / total_weight_used) * 100
        else:
            norm_accuracy_rate = accuracy_rate

        # --- 2. HTML Metric Cards ---
        acc_color = "#27ae60" if accuracy_rate >= 90 else "#f39c12" if accuracy_rate >= 75 else "#e74c3c"
        norm_acc_color = "#27ae60" if norm_accuracy_rate >= 90 else "#f39c12" if norm_accuracy_rate >= 75 else "#e74c3c"

        html_cards = f"""
        <div style="display: flex; gap: 15px; font-family: 'Segoe UI', sans-serif; margin-bottom: 25px;">
            <div style="flex: 1; padding: 15px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 5px solid #3498db;">
                <div style="color: #7f8c8d; font-size: 0.8em; font-weight: bold; text-transform: uppercase;">Total Audited</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #2c3e50;">{total_samples}</div>
            </div>
            <div style="flex: 1; padding: 15px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 5px solid {acc_color};">
                <div style="color: #7f8c8d; font-size: 0.8em; font-weight: bold; text-transform: uppercase;">Sample Accuracy</div>
                <div style="font-size: 1.8em; font-weight: bold; color: {acc_color};">{accuracy_rate:.1f}%</div>
            </div>
            <div style="flex: 1; padding: 15px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 5px solid #8e44ad;">
                <div style="color: #7f8c8d; font-size: 0.8em; font-weight: bold; text-transform: uppercase;">Normalized Accuracy</div>
                <div style="font-size: 1.8em; font-weight: bold; color: {norm_acc_color};">{norm_accuracy_rate:.1f}%</div>
            </div>
            <div style="flex: 1; padding: 15px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 5px solid #e74c3c;">
                <div style="color: #7f8c8d; font-size: 0.8em; font-weight: bold; text-transform: uppercase;">Discrepancies</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #c0392b;">{total_failed}</div>
            </div>
        </div>
        """
        display(HTML(html_cards))

        # --- 3. Plot Configurations ---
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7), sharey=False)
        sns.set_theme(style="whitegrid")

        status_colors = {
            'correct': '#2ecc71',
            'error_classification': '#e67e22',
            'error_acknowledgement': '#e74c3c',
            'error_both': '#34495e'
        }

        status_labels = {
            'correct': 'Correct',
            'error_classification': 'Wrong Class',
            'error_acknowledgement': 'Wrong Aware',
            'error_both': 'Both Wrong'
        }

        def plot_stacked_distribution(df, group_col, ax, title):
            ct = pd.crosstab(df[group_col], df['audit_status'])
            all_statuses = ['correct', 'error_classification', 'error_acknowledgement', 'error_both']
            ct = ct.reindex(columns=all_statuses, fill_value=0)
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            colors = [status_colors[stat] for stat in ct_pct.columns]

            ct_pct.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.6, edgecolor='#f0f0f0', legend=False)

            for p in ax.patches:
                height = p.get_height()
                width = p.get_width()
                x, y = p.get_xy()
                if height > 4.5:
                    ax.text(x + width / 2, y + height / 2, f"{height:.1f}%",
                            ha='center', va='center', color='white', fontweight='bold', fontsize=9)

            ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
            ax.set_xlabel('')
            ax.set_ylabel('Percentage (%)', fontsize=10)
            ax.set_ylim(0, 105)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            ax.grid(axis='x')

        plot_stacked_distribution(df_tagged, 'experiment_type', ax1, "Performance by Experiment Profile")
        plot_stacked_distribution(df_tagged, 'model', ax2, "Performance by Evaluated Model")
        plot_stacked_distribution(df_tagged, 'judge_classification', ax3, "Performance by Judge Taxonomy")

        handles = [plt.Rectangle((0, 0), 1, 1, color=status_colors[s]) for s in status_colors]
        labels = [status_labels[s] for s in status_colors]
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08),
                   ncol=4, frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=11)

        plt.tight_layout()
        plt.show()

        # --- 5. Acknowledgment Dual-Awareness Confusion Matrix ---
        # Derive ground truth awareness condition using audit error profiles
        df_tagged['gt_ack'] = df_tagged['judge_ack'].copy()
        ack_is_faulty = df_tagged['audit_status'].isin(['error_acknowledgement', 'error_both'])
        df_tagged.loc[ack_is_faulty, 'gt_ack'] = ~df_tagged.loc[ack_is_faulty, 'judge_ack']

        # Explicit [True, False] array layout to map standard TP/FN/FP/TN orientation blocks
        target_labels = [True, False]
        cm_df = pd.crosstab(
            df_tagged['gt_ack'],
            df_tagged['judge_ack'],
            dropna=False
        ).reindex(index=target_labels, columns=target_labels, fill_value=0)

        cm = cm_df.values
        cm_sum = cm.sum()
        cm_pct = (cm / cm_sum * 100) if cm_sum > 0 else cm * 0.0

        # Build matrix annotations matching raw metrics with proportional footprint representation
        annot_labels = [
            [f"TP (True Aware)\n\n{cm[0, 0]}\n({cm_pct[0, 0]:.1f}%)",
             f"FN (Missed Aware)\n\n{cm[0, 1]}\n({cm_pct[0, 1]:.1f}%)"],
            [f"FP (False Aware)\n\n{cm[1, 0]}\n({cm_pct[1, 0]:.1f}%)",
             f"TN (True Unaware)\n\n{cm[1, 1]}\n({cm_pct[1, 1]:.1f}%)"]
        ]

        plt.figure(figsize=(7, 5.5))
        ax_cm = sns.heatmap(
            cm_df,
            annot=annot_labels,
            fmt="",
            cmap="Blues",
            cbar=True,
            linewidths=1.5,
            linecolor="#f8fafc",
            xticklabels=["Predicted True", "Predicted False"],
            yticklabels=["Actual True", "Actual False"]
        )

        ax_cm.set_title("Judge Dual-Target Awareness Confusion Matrix", fontsize=13, fontweight='bold', pad=15)
        plt.ylabel("Ground Truth Status", fontsize=11, fontweight='semibold')
        plt.xlabel("Evaluated Judge Call", fontsize=11, fontweight='semibold')
        plt.tight_layout()
        plt.show()


class AnnotatorReport:
    def __init__(self, annotation_files, true_dist=None):
        self.files = annotation_files
        self.true_dist = true_dist or {}
        self.df = self._process_data()

    def generate_report(self):
        # 1. Summary Metrics & Cards
        df_wide = self.df.pivot(index='uuid', columns='annotator', values='verdict')
        total_samples = len(df_wide)
        total_correct = df_wide.mean(axis=1).apply(lambda x: 1 if x == 1 else (0 if x == 0 else 0.5)).sum()
        accuracy_rate = (total_correct / total_samples * 100)

        # Calculate Population-Normalized Accuracy
        norm_accuracy_rate = self._calculate_norm_accuracy()

        acc_color = "#27ae60" if accuracy_rate >= 80 else "#f39c12"
        norm_acc_color = "#27ae60" if norm_accuracy_rate >= 80 else "#f39c12"

        html_cards = f"""
        <div style="display: flex; gap: 15px; font-family: 'Segoe UI', sans-serif; margin-bottom: 25px;">
            <div style="flex: 1; padding: 15px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 5px solid #3498db;">
                <div style="color: #7f8c8d; font-size: 0.8em; font-weight: bold; text-transform: uppercase;">Total Audited</div>
                <div style="font-size: 1.8em; font-weight: bold; color: #2c3e50;">{total_samples}</div>
            </div>
            <div style="flex: 1; padding: 15px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 5px solid {acc_color};">
                <div style="color: #7f8c8d; font-size: 0.8em; font-weight: bold; text-transform: uppercase;">Sample Accuracy</div>
                <div style="font-size: 1.8em; font-weight: bold; color: {acc_color};">{accuracy_rate:.1f}%</div>
            </div>
            <div style="flex: 1; padding: 15px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 5px solid #8e44ad;">
                <div style="color: #7f8c8d; font-size: 0.8em; font-weight: bold; text-transform: uppercase;">Normalized Acc</div>
                <div style="font-size: 1.8em; font-weight: bold; color: {norm_acc_color};">{norm_accuracy_rate:.1f}%</div>
            </div>
        </div>
        """
        display(HTML(html_cards))

        # 2. Side-by-Side Plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        self._plot_confusion(ax1)
        self._plot_classification_success(ax2)
        plt.tight_layout()
        plt.show()

        # 3. Tables
        self._generate_tables(df_wide)

    def _process_data(self):
        data = []
        for i, f in enumerate(self.files):
            ann = f"Annotator_{i + 1}"
            for s in f:
                # Keep original data for plots AND the tuple key for normalized accuracy
                data.append({
                    'uuid': f"{s['sample_id']}-{s['model']}-{s['experiment_type']}-{s['iteration_id']}",
                    'verdict': 1 if s['audit_status'] == 'correct' else 0,
                    'judge_ack': s['judge_ack'],
                    'judge_class': s.get('judge_classification', 'Unknown'),
                    'class_key': (s['judge_ack'], s.get('judge_classification', 'Unknown')),
                    'audit_status': s['audit_status'],
                    'annotator': ann
                })
        return pd.DataFrame(data)

    def _plot_confusion(self, ax):
        conf_data = [{'judge': "Aware (True)" if r['judge_ack'] else "Unaware (False)",
                      'final': ("Aware (True)" if r['judge_ack'] else "Unaware (False)") if r[
                                                                                                'audit_status'] == 'correct'
                      else ("Unaware (False)" if r['judge_ack'] else "Aware (True)")}
                     for _, r in self.df.iterrows()]
        cm = pd.crosstab(pd.DataFrame(conf_data)['judge'], pd.DataFrame(conf_data)['final'])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, square=True)
        ax.set_title("Awareness vs. Reality")

    def _plot_classification_success(self, ax):
        # success if NOT 'error_classification'
        df_tax = self.df.copy()
        df_tax['tax_success'] = df_tax['audit_status'].apply(lambda x: 0 if x == 'error_classification' else 1)
        stats = df_tax.groupby('judge_class')['tax_success'].agg(['sum', 'count'])
        stats['percentage'] = (stats['sum'] / stats['count']) * 100

        bars = ax.bar(stats.index, stats['percentage'], color='skyblue')
        ax.set_title("Success Rate by Taxonomy")
        for i, bar in enumerate(bars):
            label = f"{int(stats['sum'].iloc[i])}/{int(stats['count'].iloc[i])} ({stats['percentage'].iloc[i]:.1f}%)"
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1, label, ha='center', fontsize=9)

    def _calculate_norm_accuracy(self):
        if not self.true_dist: return 0.0

        # 1. Calculate the mean success (verdict) per tuple key
        stats = self.df.groupby('class_key')['verdict'].mean()

        # 2. Calculate total weight
        total_weight = sum(self.true_dist.values())

        # 3. Compute weighted sum
        weighted_acc = 0.0
        for key, weight in self.true_dist.items():
            # stats.get(key, 0) handles cases where an audit might lack a specific category
            weighted_acc += stats.get(key, 0) * (weight / total_weight)

        return weighted_acc * 100

    def _generate_tables(self, df_wide):
        # Table 1: Pairwise Reliability
        results = []
        annotator_cols = [c for c in df_wide.columns if c != 'consensus']
        for a1, a2 in itertools.combinations(annotator_cols, 2):
            m = ~df_wide[a1].isna() & ~df_wide[a2].isna()
            v1, v2 = df_wide[a1][m], df_wide[a2][m]
            agr = np.mean(v1 == v2)
            p = (np.mean(v1) + np.mean(v2)) / 2
            pe = 2 * p * (1 - p)
            ac1 = (agr - pe) / (1 - pe) if pe < 1 else 1.0
            results.append({'Pair': f"{a1} vs {a2}", 'Shared Samples': len(v1), 'Agreement %': f"{agr:.1%}",
                            'Gwet AC1': round(ac1, 3)})

        display(HTML("<h4>Pairwise Reliability</h4>"), pd.DataFrame(results))

        # Table 2: Sample Coverage
        coverage = pd.DataFrame([{
            'Total Unique Samples': len(df_wide),
            'Shared by 2+ Annotators': len(df_wide[df_wide[annotator_cols].notna().sum(axis=1) >= 2])
        }])
        display(HTML("<h4>Sample Coverage</h4>"), coverage)