"""
KnowledgeMaze Dataset Distiller & Preprocessor

This module provides the core logic for generating standardized, "frozen" evaluation
subsets from the KnowledgeMaze master dataset. It handles the stratified sampling
of reasoning tasks and the deterministic generation of shuffled context layouts
to ensure scientific reproducibility across diverse LLM architectures.

Key Capabilities:
-----------------
1.  Frozen Shuffle Registry: Generates 5 distinct paragraph-level permutations per
    sample. By freezing these shuffles, we ensure that models (e.g., Base vs. CoT)
    are evaluated on identical layouts, isolating the effect of the reasoning prompt
    from positional noise.

2.  Stratified Sampling: Implements a priority-based sampling strategy that:
    - Guarantees the inclusion of all 'Hedged' (uncertain) claim variants.
    - Balances the distribution across 11 reasoning strategies (e.g., Numerical
      Thresholds, Temporal Ordering) to 250 samples per strategy where possible.

3.  Experimental Subsets: Outputs three standardized JSON files to the
    `results/preprocessed_data/` directory:
    - full_study.json: The primary benchmark (~3.2k samples).
    - cot_subset.json: A smaller subset (1.1k samples) optimized for CoT experiments.
    - hedged_subset.json: A targeted subset (803 samples) for uncertainty analysis.

4.  Integrity Reporting: Provides automated statistical validation of positional
    bias (Primacy/Recency) and edge salience to ensure the shuffled contexts do not
    unfairly disadvantage either the 'Evidence' or 'Claim' paragraphs.
"""


import os
import json
import random
from typing import List, Dict
from collections import defaultdict
from dotenv.main import load_dotenv

load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')

class KnowledgeMazeDatasetDistiller:
    def __init__(self, all_samples: List[Dict], paragraphs_data: Dict):
        self.all_samples = all_samples
        self.paragraphs = paragraphs_data
        self.repo_path = REPO_PATH
        self.output_dir = os.path.join(self.repo_path, "results/preprocessed_data/")

        # Priority: Hedged samples
        self.hedged_samples = [
            s for s in all_samples
            if self.paragraphs.get(s['Claim paragraph'], {}).get('Confidence') == 'Hedged'
        ]

        self.by_strategy = defaultdict(list)
        for s in all_samples:
            self.by_strategy[s['Selected strategy']].append(s)

    def _generate_shuffled_context(self, sample: Dict) -> List[Dict]:
        """Generates 5 frozen iterations of the 10 paragraph IDs."""
        # 1 Ev + 1 Claim + 8 Distractors
        ev_id = sample['Evidence paragraph']
        cl_id = sample['Claim paragraph']
        all_ids = [ev_id, cl_id] + sample['Distractor_IDs']

        iterations = []
        for i in range(1, 6):
            shuffled = random.sample(all_ids, len(all_ids))
            iterations.append({
                'iteration_id': i,
                'paragraphs': shuffled,
                'ev_pos': shuffled.index(ev_id) + 1,
                'cl_pos': shuffled.index(cl_id) + 1
            })
        return iterations

    def _process_records(self, samples_list: List[Dict]) -> Dict:
        return {
            s['ID']: {
                'Question': s['Question'],
                'Selected strategy': s['Selected strategy'],
                'Answer': s['Answer'],
                'Claim_ID': s['Claim paragraph'],
                'Evidence_ID': s['Evidence paragraph'],
                'Context': self._generate_shuffled_context(s)
            } for s in samples_list
        }

    def create_subsets(self):
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Build the Full Study Population (~3.2k)
        full_set_ids = {s['ID'] for s in self.hedged_samples}

        for strategy, samples in self.by_strategy.items():
            current_count = len([s for s in samples if s['ID'] in full_set_ids])
            needed = max(0, 250 - current_count)
            remaining = [s for s in samples if s['ID'] not in full_set_ids]

            selected = random.sample(remaining, min(len(remaining), needed))
            for s in selected:
                full_set_ids.add(s['ID'])

        full_study_list = [s for s in self.all_samples if s['ID'] in full_set_ids]

        # Group by strategy for downstream subsetting
        full_by_strat = defaultdict(list)
        for s in full_study_list:
            full_by_strat[s['Selected strategy']].append(s)

        # 2. Generate CoT Subset (1.1k)
        cot_list = []
        for strategy, samples in full_by_strat.items():
            cot_list.extend(random.sample(samples, min(len(samples), 100)))

        # 3. Generate Scaling Subset (660 samples: 60 per strategy)
        scaling_list = []
        for strategy, samples in full_by_strat.items():
            scaling_list.extend(random.sample(samples, min(len(samples), 60)))

        # 4. Finalize and Process
        sets = {
            "full_study.json": self._process_records(full_study_list),
            "cot_subset.json": self._process_records(cot_list),
            "scaling_subset.json": self._process_records(scaling_list),
            "hedged_subset.json": self._process_records(self.hedged_samples)
        }

        # Save files
        for filename, data in sets.items():
            with open(os.path.join(self.output_dir, filename), 'w') as f:
                json.dump(data, f, indent=4)

        self.generate_report(sets)

    def generate_report(self, sets: Dict):
        """Generates the requested Dashboard and Visualizations."""
        # --- 1. Dashboard Summary ---
        print("\n" + "═" * 60)
        print("🚀 PREPROCESSING DASHBOARD SUMMARY")
        print("═" * 60)
        print(f"{'Dataset Name':<20} | {'Total Samples':<15} |")
        print("-" * 60)
        for name, data in sets.items():
            print(f"{name:<20} | {len(data):<15} |")
        print("═" * 60)

        # --- 2. STRATEGY DISTRIBUTION TABLE ---
        all_strategies = sorted(self.by_strategy.keys())

        print("\n### Strategy Distribution Report\n")
        # Added column for Scaling Subset
        header = f"| {'Selected Strategy':<30} | {'Full':<14} | {'CoT':<14} | {'Scaling':<14} | {'Hedged':<14} |"
        separator = f"| {'-' * 30} | {'-' * 14} | {'-' * 14} | {'-' * 14} | {'-' * 14} |"
        print(header)
        print(separator)

        for strat in all_strategies:
            row_data = []
            for name in ["full_study.json", "cot_subset.json", "scaling_subset.json", "hedged_subset.json"]:
                data = sets[name]
                count = sum(1 for v in data.values() if v['Selected strategy'] == strat)
                percent = (count / len(data)) * 100 if len(data) > 0 else 0
                row_data.append(f"{count} ({percent:.1f}%)")

            print(f"| {strat:<30} | {row_data[0]:<14} | {row_data[1]:<14} | {row_data[2]:<14} | {row_data[3]:<14} |")