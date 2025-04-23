import os
import math
import json
import random
import itertools
from typing import Dict, List, Any

from dotenv import load_dotenv

load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')


# Load QA data
try:
    # Use os.path.join for better path compatibility
    QA_FILE_PATH = os.path.join(REPO_PATH, "data_files", "knowledge_maze_qa.json")
    with open(QA_FILE_PATH, "r", encoding="utf-8") as file:
        REASONING_DATASET: List[Dict[str, Any]] = json.load(file)
except FileNotFoundError:
    print(f"❌ Error: QA data file not found at {QA_FILE_PATH}")
    REASONING_DATASET = []
except Exception as e:
    print(f"❌ Error loading QA data from {QA_FILE_PATH}: {e}")
    REASONING_DATASET = []


# Load Paragraph data
try:
    PARAGRAPHS_FILE_PATH = os.path.join(REPO_PATH, "data_files", "knowledge_maze_paragraphs.json")
    with open(PARAGRAPHS_FILE_PATH, "r", encoding="utf-8") as file:
        PARAGRAPH_DATASET: Dict[str, Dict[str, str]] = json.load(file)
except FileNotFoundError:
    print(f"❌ Error: Paragraph data file not found at {PARAGRAPHS_FILE_PATH}")
    PARAGRAPH_DATASET = {}
except Exception as e:
    print(f"❌ Error loading Paragraph data from {PARAGRAPHS_FILE_PATH}: {e}")
    PARAGRAPH_DATASET = {}

def split_ids_with_overlap(
        all_ids: list,
        num_groups: int,
        overlap_percent: float
) -> None:
    """
    Splits a list of unique IDs into a specified number of groups (sub-lists)
    with a target percentage overlap between each pair of groups.

    Args:
        all_ids: A list of unique identifiers.
        num_groups: The desired number of groups (sub-lists) to create.
        overlap_percent: The target overlap percentage (0-100) between any
                         two groups, relative to the size of one group.

    Returns:
        A list of lists, where each sub-list represents a group and contains
        the IDs assigned to it.

    Raises:
        ValueError: If inputs are invalid (e.g., overlap > 100, num_groups < 2,
                    not enough IDs for the requested overlap).
    """
    total_unique_ids = len(all_ids)
    overlap_ratio = overlap_percent / 100.0

    # --- Calculate theoretical sizes ---
    # Based on the following formula: T = S * [N - (N * (N - 1) / 2) * P]
    # Where T=total_unique_ids, N=num_groups, P=overlap_ratio, S=group_size
    num_pairs = num_groups * (num_groups - 1) / 2
    denominator = num_groups - (num_pairs * overlap_ratio)

    if denominator <= 0 < total_unique_ids:
        raise ValueError(
            f"Requested overlap ({overlap_percent}%) is too high for "
            f"{num_groups} groups. Results in non-positive denominator."
        )
    elif denominator <= 0 and total_unique_ids == 0:
        # Handle case where no IDs are provided, return empty lists
        pass
    elif denominator > 0:
        # Calculate the theoretical fractional size per group
        theoretical_size_per_group = total_unique_ids / denominator
        # Round up to ensure we have enough slots *at least*
        group_size = math.ceil(theoretical_size_per_group)
    else:  # Should ideally not be reached if denominator logic is correct
        group_size = 0  # Default or fallback

    # Handle case of zero overlap explicitly to avoid division by zero if N=0/1
    if overlap_ratio == 0:
        group_size = math.ceil(total_unique_ids / num_groups) if num_groups > 0 else 0

    # Calculate target number of shared items per pair based on the rounded group size
    shared_per_pair = math.ceil(overlap_ratio * group_size) if group_size > 0 else 0

    # --- Prepare data structures ---
    groups = [set() for _ in range(num_groups)]
    # Shuffle IDs for random assignment
    available_ids = list(set(all_ids))  # Ensure uniqueness and copy
    random.shuffle(available_ids)

    # --- Assign Pairwise Overlapping IDs ---
    pairs = list(itertools.combinations(range(num_groups), 2))
    assigned_for_overlap = 0

    for i, j in pairs:
        count = 0
        ids_for_this_pair = []
        # Try to pick 'shared_per_pair' unique IDs for this pair
        while count < shared_per_pair and available_ids:
            try:
                # Pick an ID preferably not already in either group (less likely needed now)
                # but ultimately take any available if needed.
                idx_to_pop = -1  # Default to last element (efficient pop)
                picked_id = available_ids.pop(idx_to_pop)

                # Only add if it doesn't push group over the size limit unnecessarily
                # (Though overlaps might force this slightly)
                if len(groups[i]) < group_size or len(groups[j]) < group_size:
                    ids_for_this_pair.append(picked_id)
                    count += 1
                else:
                    # Put it back if both groups are already full - might happen with rounding
                    available_ids.append(picked_id)  # Add back somewhere
                    random.shuffle(available_ids)  # Reshuffle after adding back
                    break  # Stop trying to add to this pair if both full

            except IndexError:
                # Should not happen if calculations are right, but handle defensively
                print(
                    f"Warning: Ran out of available IDs while assigning overlap for pair ({i}, {j}). Needed {shared_per_pair}, got {count}.")
                break  # Stop trying for this pair

        # Add the successfully picked IDs to both groups in the pair
        for item_id in ids_for_this_pair:
            groups[i].add(item_id)
            groups[j].add(item_id)
            assigned_for_overlap += 1  # Count distinct IDs used for overlaps

    # --- Assign Remaining IDs (Unique to Groups) ---
    group_indices = list(range(num_groups))
    random.shuffle(group_indices)  # Distribute unique items randomly across groups

    ids_assigned_uniquely = 0
    while available_ids:
        filled_one = False
        # Try assigning remaining IDs to groups that are not yet full (round-robin)
        for group_idx in group_indices:
            if not available_ids: break  # No more IDs left

            if len(groups[group_idx]) < group_size:
                try:
                    unique_id = available_ids.pop()
                    groups[group_idx].add(unique_id)
                    ids_assigned_uniquely += 1
                    filled_one = True
                except IndexError:
                    break  # Should not happen here, but safety first

        if not filled_one and available_ids:
            # If all groups met the target size but IDs remain (due to rounding),
            # distribute them, e.g., round-robin or to the smallest group
            try:
                fallback_id = available_ids.pop()
                # Find the smallest group to add to
                smallest_group_idx = min(range(num_groups), key=lambda k: len(groups[k]))
                groups[smallest_group_idx].add(fallback_id)
                ids_assigned_uniquely += 1
            except IndexError:
                break  # No more IDs

    # --- Final Check and Conversion ---
    final_groups = [list(g) for g in groups]
    output = {}
    for i, g in enumerate(final_groups):
        output[i] = g

    # Verification (optional, can be slow for large datasets)
    print(f"Target group size: {group_size}")
    print(f"Target shared per pair: {shared_per_pair}")
    for i in range(num_groups):
        print(f"Group {i} size: {len(final_groups[i])}")

    total_assigned_unique = sum(len(g) for g in final_groups) - assigned_for_overlap
    print(f"Total unique IDs used: {total_assigned_unique}")

    # Save output in folder:
    output_path = os.path.join(REPO_PATH, "data_files", "annotation_ids.json")
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(output, out_file, indent=4)


def get_ids_with_paragraphs() -> list:
    """
    Extract sample IDs from REASONING_DATASET that have associated paragraphs
    present in PARAGRAPH_DATASET.

    Returns:
        list: List of sample IDs that reference at least one paragraph found in PARAGRAPH_DATASET.
    """
    valid_ids = []
    paragraph_keys = set(PARAGRAPH_DATASET.keys())

    for sample in REASONING_DATASET:
        found = False
        for path_key in ['Reasoning path 1', 'Reasoning path 2']:
            for step in sample.get(path_key, []):
                pid = step.get('Paragraph')
                if pid and pid in paragraph_keys:
                    found = True
                    break
            if found:
                break
        if found:
            valid_ids.append(sample['ID'])

    return valid_ids
