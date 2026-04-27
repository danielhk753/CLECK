import copy
import random


class AnnotationSplitter:
    """
    Splits a dataset into overlapping subsets for human annotation.
    Target: 500 samples total.
    Overlap: 200 samples (each seen by 2 people).
    Uniques: 300 samples (distributed 150-50-50-50).
    """

    @staticmethod
    def distribute(dataset, num_annotators=4, overlap_size=200, unique_splits=[150, 50, 50, 50]):
        # 1. Shuffle to ensure no ordering bias
        data_pool = copy.deepcopy(dataset)
        random.shuffle(data_pool)

        if len(data_pool) < (overlap_size + sum(unique_splits)):
            raise ValueError(f"Dataset too small. Need {overlap_size + sum(unique_splits)} samples.")

        # 2. Extract the pools
        overlap_pool = data_pool[:overlap_size]
        unique_pool = data_pool[overlap_size:overlap_size + sum(unique_splits)]

        # Initialize sub-datasets
        annotator_batches = [[] for _ in range(num_annotators)]

        # --- Step A: Distribute Overlap (Round Robin) ---
        # To ensure each sample has 2 annotators and each annotator has 100 shared:
        # Each annotator i shares half their quota with i-1 and half with i+1
        share_per_pair = overlap_size // num_annotators  # 200 / 4 = 50

        for i in range(num_annotators):
            start_idx = i * share_per_pair
            end_idx = (i + 1) * share_per_pair
            pair_samples = overlap_pool[start_idx:end_idx]

            # Assign to current and next (circularly)
            annotator_batches[i].extend(pair_samples)
            annotator_batches[(i + 1) % num_annotators].extend(pair_samples)

        # --- Step B: Distribute Uniques ---
        current_idx = 0
        for i, count in enumerate(unique_splits):
            annotator_batches[i].extend(unique_pool[current_idx: current_idx + count])
            current_idx += count

        # 3. Return as a dict for easy access
        return {f"Annotator_{i + 1}": batch for i, batch in enumerate(annotator_batches)}
