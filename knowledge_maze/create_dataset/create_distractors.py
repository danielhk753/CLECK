import copy
import random

# ---------------------------------------------------------
# 1. DISTRIBUTED DISTRACTOR ATTACHMENT
# ---------------------------------------------------------

def attach_existing_paragraph_distractors(dataset, num_distractors_per_sample=16):
    """
    Attach distractor paragraphs to each sample by taking them from other samples only.
    - Distractors cannot come from the same sample.
    - Distractors cannot duplicate any paragraph already in the sample.
    - Output: dataset with sample['context']['distractors'] = list of paragraphs
    """
    new_dataset = copy.deepcopy(dataset)

    # Build global pool of paragraphs per sample
    sample_paragraphs = {}
    for idx, sample in enumerate(new_dataset):
        sample_paragraphs[idx] = []
        for key in ['direct claim', 'co-located aggregation', 'distributed multi-hop']:
            if key in sample['context']:
                val = sample['context'][key]
                if isinstance(val, str):
                    sample_paragraphs[idx].append(val)
                elif isinstance(val, list):
                    sample_paragraphs[idx].extend(val)

    # Collect all paragraphs globally for easier sampling
    all_paragraphs = []
    for idx, paras in sample_paragraphs.items():
        for p in paras:
            all_paragraphs.append((idx, p))

    # Attach distractors per sample
    for idx, sample in enumerate(new_dataset):
        # Current sample's paragraphs
        current_paras = set(sample_paragraphs[idx])

        # Eligible paragraphs = from other samples, not duplicated
        eligible = [p for sid, p in all_paragraphs if sid != idx and p not in current_paras]

        # Sample distractors
        k = min(num_distractors_per_sample, len(eligible))
        chosen = random.sample(eligible, k) if k > 0 else []

        # Attach to sample
        sample['context']['distractors'] = chosen

    return new_dataset


# ---------------------------------------------------------
# 2. FULL PIPELINE FUNCTION
# ---------------------------------------------------------

def add_distractors(dataset, num_distractors=16):
    """
    Full pipeline: attach distractors using only existing paragraphs.
    """
    return attach_existing_paragraph_distractors(dataset, num_distractors_per_sample=num_distractors)
