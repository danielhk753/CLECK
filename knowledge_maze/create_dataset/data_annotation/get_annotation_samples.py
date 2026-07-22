"""
Annotation Management Module
============================

This module provides tools for statistically balanced sampling and batch
distribution of datasets for human annotation.

The primary component, `AnnotationSplitter`, is designed to select a subset
of data from a larger pool while ensuring an equal representation of various
reasoning strategies (e.g., FILTER_COUNT_NUMERICAL, EXTREME_NUMERICAL).

Key Features:
-------------
*   **Strategy Equalization**: Implements a quota redistribution algorithm to
    ensure that under-represented strategies are maximized and over-represented
    strategies are capped, resulting in a balanced test set.
*   **Dual-Visibility Overlap**: Supports inter-annotator agreement (IAA) studies
    by distributing a common 'overlap' pool across annotator pairs.
*   **Unique Batch Splitting**: Handles the distribution of unique, non-overlapping
    samples to maximize total data coverage.
*   **Audit Reporting**: Generates a selection report summarizing the distribution
    of strategies across the final selected sample set.

Note:
-----
This splitter assumes the input dataset consists of dictionaries containing
at least a 'Selected strategy' key and a 'Human Validated' boolean flag.
"""

import random
from collections import Counter, defaultdict


class AnnotationSplitter:
    """
    Selects 500 unverified samples with strict strategy balancing,
    then splits them into overlapping annotator batches.
    """

    @staticmethod
    def distribute(dataset, num_annotators=4, overlap_size=200, unique_splits=[150, 50, 50, 50]):
        # 1. Hard Filter: Remove already verified data
        pool = [q for q in dataset if not q.get('Human Validated', False)]

        target_total = overlap_size + sum(unique_splits)  # 500
        if len(pool) < target_total:
            raise ValueError(f"Not enough unverified data. Need {target_total}, have {len(pool)}.")

        # 2. Group by Strategy
        strat_groups = defaultdict(list)
        for q in pool:
            strat = q.get('Selected strategy', 'Unknown')
            strat_groups[strat].append(q)

        # 3. Strategy Equalizer (Quota Redistribution)
        # We want to take exactly 'target_total' while keeping strategy counts equal
        selected_samples = []
        remaining_needed = target_total
        available_strats = sorted(strat_groups.keys(), key=lambda x: len(strat_groups[x]))

        for i, strat in enumerate(available_strats):
            # Calculate fair share based on what's left to fill
            num_remaining_strats = len(available_strats) - i
            fair_share = remaining_needed // num_remaining_strats

            # Take either our fair share or everything the strategy has
            group = strat_groups[strat]
            num_to_take = min(len(group), fair_share)

            selected_samples.extend(random.sample(group, num_to_take))
            remaining_needed -= num_to_take

        # Safety: If rounding left us short by 1-2 samples, fill from leftovers
        if remaining_needed > 0:
            leftovers = [q for q in pool if q not in selected_samples]
            selected_samples.extend(random.sample(leftovers, remaining_needed))

        # 4. Distribution Report
        report = Counter([q.get('Selected strategy', 'Unknown') for q in selected_samples])
        print("--- ⚖️ Balanced Selection Report ---")
        print(f"Total Selected: {len(selected_samples)}")
        for strat, count in sorted(report.items()):
            print(f"  ▪️ {strat:25} : {count} samples")
        print("------------------------------------\n")

        # 5. Split for Annotators
        # Shuffle now so strategies are mixed across all users
        random.shuffle(selected_samples)

        overlap_pool = selected_samples[:overlap_size]
        unique_pool = selected_samples[overlap_size:]

        annotator_batches = [[] for _ in range(num_annotators)]

        # Step A: Distribute Overlap (Dual-visibility)
        share_per_pair = overlap_size // num_annotators
        for i in range(num_annotators):
            pair_samples = overlap_pool[i * share_per_pair: (i + 1) * share_per_pair]
            annotator_batches[i].extend(pair_samples)
            annotator_batches[(i + 1) % num_annotators].extend(pair_samples)

        # Step B: Distribute Uniques
        curr = 0
        for i, count in enumerate(unique_splits):
            annotator_batches[i].extend(unique_pool[curr: curr + count])
            curr += count

        return {f"Annotator_{i + 1}": batch for i, batch in enumerate(annotator_batches)}