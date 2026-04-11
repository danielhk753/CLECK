"""
Answer Generation Orchestrator Module
=====================================

This module provides the central orchestration logic for transforming raw table-question
pairs into fully formed counterfactual reasoning samples. It manages the execution
of various mathematical, relational, and temporal operations to generate:
1. Ground Truth Answers (Memory)
2. Distractor Claims (Claim)
3. Mutated Evidence sets (Evidence)

The orchestrator handles two primary processing flows:
- **Single-Sample Processing**: Standard operations (Aggregation, Filter, Rank)
  processed row-by-row.
- **Batch Processing**: Specialized temporal operations (Extreme/Filter Temporal)
  that require subset-wide context or optimized batch execution.

Key Components:
---------------
- AnswerGenerator: The primary class responsible for strategy mapping,
  dataset separation, and execution monitoring.
- Strategy Mapping: A registry connecting logical strategy names (e.g., 'TOP_K_RANKING')
  to their underlying functional implementations.
- Reporting: Integrated HTML dashboard generation for real-time success/rejection
  analytics within Jupyter environments.

Usage:
------
    generator = AnswerGenerator()
    processed_data = generator.process_dataset(raw_dataset)
"""

import re
import unicodedata
from IPython.display import display, HTML

from knowledge_maze.tools.utils import save_json
from knowledge_maze.create_dataset.create_answers.strategies.aggregation_ops import run_count_total, run_aggregation
from knowledge_maze.create_dataset.create_answers.strategies.extreme_ops import run_extreme_numerical, run_top_k, run_extreme_temporal
from knowledge_maze.create_dataset.create_answers.strategies.filter_ops import run_categorical_count, run_filter_numerical, run_filter_temporal
from knowledge_maze.create_dataset.create_answers.strategies.relational_ops import run_comparative_difference, run_unique_diversity, run_frequency_mode


class AnswerGenerator:
    def __init__(self):
        # Map single-sample strategies to their specific functions
        self.strategy_mapping = {
            'COUNT_TOTAL': run_count_total,
            'AGGREGATION': run_aggregation,
            'EXTREME_NUMERICAL': run_extreme_numerical,
            'TOP_K_RANKING': run_top_k,
            'CATEGORICAL_FILTER_LOOKUP': run_categorical_count,
            'FILTER_COUNT_NUMERICAL': run_filter_numerical,
            'COMPARATIVE_DIFFERENCE': run_comparative_difference,
            'UNIQUE_DIVERSITY': run_unique_diversity,
            'FREQUENCY_MODE': run_frequency_mode
        }

        # Track batch strategies separately
        self.batch_strategies = ['EXTREME_TEMPORAL', 'FILTER_COUNT_TEMPORAL']

        # All known strategies for stats initialization
        self.all_strategies = list(self.strategy_mapping.keys()) + self.batch_strategies

    @staticmethod
    def _sanitize_table(data_list: list[dict]) -> list[dict]:
        """
        Cleans a list of dicts:
        1. Removes brackets and their contents.
        2. Removes superscript-style suffixes like * and ^.
        3. Removes non-Latin characters (preserving accents and symbols).
        4. Cleans all whitespace (leading, trailing, and double spaces).
        """
        sanitized_list = []

        for row in data_list:
            new_row = {}
            for key, value in row.items():
                if not isinstance(value, str):
                    new_row[key] = value
                    continue

                # 1. Remove anything inside brackets: (), [], {}
                text = re.sub(r'\(.*?\)|\[.*?]|\{.*?}', '', value)

                # 2. Remove suffixes of * or ^ (often used for footnotes/citations)
                # This looks for * or ^ that appear at the end of words or strings
                text = re.sub(r'[*^]+', '', text)

                # 3. Filter characters
                filtered_chars = []
                for char in text:
                    # We use .name() to identify script
                    # We use .category() to identify if it's a "Symbol" or "Punctuation"
                    name = unicodedata.name(char, "UNKNOWN")
                    category = unicodedata.category(char)

                    # Logic:
                    # Keep if it contains "LATIN"
                    # Keep if it is a digit, space, or punctuation (P)
                    # Keep if it is a symbol (S) - this covers $, +, =, etc.
                    if ("LATIN" in name or
                            char.isspace() or
                            char.isdigit() or
                            category.startswith('P') or
                            category.startswith('S')):
                        filtered_chars.append(char)

                # 4. Reconstruct and fix whitespace
                final_text = "".join(filtered_chars)

                # Remove double spaces, then strip start/end
                final_text = re.sub(r'\s+', ' ', final_text).strip()

                new_row[key] = final_text

            sanitized_list.append(new_row)

        return sanitized_list

    def _format_output(self, sample):
        """
        Formats the successfully processed sample to match the requested output structure.
        Renames 'Relevant table data' to 'Original data' and adds 'Fake data' and 'Answer'.
        """
        return {
            'Group name': sample.get('Group name'),
            'Question': sample.get('Question'),
            'Selected strategy': sample.get('Selected strategy'),
            'Wiki link': sample.get('Wiki link'),
            'Original data': sample.get('Relevant table data'),
            'Fake data': self._sanitize_table(sample.get('new_evidence')),
            'Answer': {
                'Memory': sample.get('true_answer'),
                'Claim': sample.get('counter_answer_1'),
                'Evidence': sample.get('counter_answer_2')
            }
        }

    def process_dataset(self, dataset):
        output_dataset = []

        # Initialize stats tracker
        stats = {strat: {'passed': 0, 'rejected': 0} for strat in self.all_strategies}

        # --- 1. SEPARATE SAMPLES ---
        singles = []
        temporal_filter_subset = []
        temporal_extreme_subset = []

        for sample in dataset:
            strat = sample.get('Selected strategy')

            # Ensure unexpected strategies still get tracked safely
            if strat not in stats:
                stats[strat] = {'passed': 0, 'rejected': 0}

            if strat == 'FILTER_COUNT_TEMPORAL':
                temporal_filter_subset.append(sample)
            elif strat == 'EXTREME_TEMPORAL':
                temporal_extreme_subset.append(sample)
            else:
                singles.append(sample)

        # --- 2. BATCH PROCESSING ---
        # Handle FILTER_COUNT_TEMPORAL
        if temporal_filter_subset:
            # Pass full subset and set verbose to False as requested
            passed_temporal = run_filter_temporal(temporal_filter_subset, verbose=False)
            stats['FILTER_COUNT_TEMPORAL']['passed'] = len(passed_temporal)
            stats['FILTER_COUNT_TEMPORAL']['rejected'] = len(temporal_filter_subset) - len(passed_temporal)

            for res in passed_temporal:
                output_dataset.append(self._format_output(res))

        # Handle EXTREME_TEMPORAL
        if temporal_extreme_subset:
            # Pass full subset and set verbose to False
            passed_extreme = run_extreme_temporal(temporal_extreme_subset, verbose=False)
            stats['EXTREME_TEMPORAL']['passed'] = len(passed_extreme)
            stats['EXTREME_TEMPORAL']['rejected'] = len(temporal_extreme_subset) - len(passed_extreme)

            for res in passed_extreme:
                output_dataset.append(self._format_output(res))

        # --- 3. SINGLE SAMPLE PROCESSING ---
        for sample in singles:
            strat = sample.get('Selected strategy')
            func = self.strategy_mapping.get(strat)

            if not func:
                # If we encounter a strategy with no registered function
                stats[strat]['rejected'] += 1
                continue

            result = func(sample)

            if result is None:
                stats[strat]['rejected'] += 1
            else:
                stats[strat]['passed'] += 1
                sample.update(result)
                output_dataset.append(self._format_output(sample))

        # --- 4. OUTPUT SUMMARY LOGS ---
        self._print_summary(stats)

        save_json("question_answer_dataset", output_dataset)

    @staticmethod
    def _print_summary(stats):
        """Generates a beautiful HTML dashboard for strategy pass/reject rates."""
        total_passed = sum(s['passed'] for s in stats.values())
        total_rejected = sum(s['rejected'] for s in stats.values())
        total_initial = total_passed + total_rejected
        rate = (total_passed / total_initial * 100) if total_initial > 0 else 0

        # Build the strategy rows dynamically
        strategy_rows = ""
        for strat, counts in stats.items():
            if counts['passed'] == 0 and counts['rejected'] == 0:
                continue

            # Color coding: red for rejects, green for passes
            strategy_rows += f"""
            <tr style="border-bottom: 1px solid #334155; background-color: #111827;">
                <td style="padding: 10px; text-align: left; color: #94a3b8;">{strat}</td>
                <td style="padding: 10px; text-align: center; font-weight: bold; color: #34d399;">{counts['passed']:,}</td>
                <td style="padding: 10px; text-align: center; font-weight: bold; color: #f87171;">{counts['rejected']:,}</td>
            </tr>
            """

        html = f"""
        <div style="background: #0f172a; padding: 25px; border-radius: 15px; border: 1px solid #334155; color: #f8fafc; font-family: sans-serif; max-width: 800px; margin: 20px 0;">
            <h2 style="color: #38bdf8; margin-bottom: 25px; letter-spacing: 1px; text-align: center;"> 🧩 ANSWER GENERATION REPORT </h2>

            <!-- Top Stat Cards -->
            <div style="display: flex; gap: 10px; margin-bottom: 20px;">
                <div style="flex: 1; background: #1e293b; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
                    <small style="color: #94a3b8;">TOTAL INPUT</small><br/>
                    <span style="font-size: 24px; font-weight: bold;">{total_initial:,}</span>
                </div>
                <div style="flex: 1; background: #1e293b; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #34d399;">
                    <small style="color: #94a3b8;">SUCCESSFUL</small><br/>
                    <span style="font-size: 24px; font-weight: bold; color: #34d399;">{total_passed:,}</span>
                </div>
                <div style="flex: 1; background: #1e293b; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #38bdf8;">
                    <small style="color: #94a3b8;">SUCCESS RATE</small><br/>
                    <span style="font-size: 24px; font-weight: bold; color: #38bdf8;">{rate:.1f}%</span>
                </div>
            </div>

            <!-- Strategy Table -->
            <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; overflow: hidden;">
                <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                    <thead>
                        <tr style="background-color: #0f172a; border-bottom: 2px solid #334155;">
                            <th style="padding: 12px; text-align: left; color: #38bdf8;">Reasoning Strategy</th>
                            <th style="padding: 12px; text-align: center; color: #38bdf8;">Passed</th>
                            <th style="padding: 12px; text-align: center; color: #38bdf8;">Rejected</th>
                        </tr>
                    </thead>
                    <tbody>
                        {strategy_rows}
                    </tbody>
                </table>
            </div>
        </div>
        """
        display(HTML(html))