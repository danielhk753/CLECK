import copy
import random
from dateutil import parser


def parse_numeric(value):
    """Extract numeric part from a string if possible, otherwise return original."""
    try:
        return float(str(value).split()[0])
    except:
        return value


def extremum_counterfactual(orig_answers, evidence, objective='max'):
    """Adjust original answers so evidence becomes max/min."""
    entities = list(orig_answers.keys())
    # Parse numeric values
    numeric_values = {k: parse_numeric(v) for k, v in orig_answers.items()}

    if evidence not in numeric_values:
        # Try matching ignoring units/whitespace
        match = [k for k in numeric_values if evidence in k or k in evidence]
        if match:
            evidence = match[0]
        else:
            raise KeyError(f"Evidence '{evidence}' not found in original answers.")

    # Sort values
    sorted_vals = sorted(numeric_values.values())

    if objective == 'max':
        target_val = max(sorted_vals)
    else:
        target_val = min(sorted_vals)

    # Assign values to make evidence the extreme
    remaining_vals = [v for v in sorted_vals if v != target_val]
    random.shuffle(remaining_vals)

    new_answers = {}
    for ent in entities:
        if ent == evidence:
            new_val = target_val
        else:
            new_val = remaining_vals.pop()
        # Preserve integer if original was integer
        if isinstance(orig_answers[ent], int):
            new_val = int(round(new_val))
        # Preserve units if string
        if isinstance(orig_answers[ent], str) and any(c.isalpha() for c in orig_answers[ent]):
            unit = ''.join([c for c in str(orig_answers[ent]) if not c.isdigit() and c not in ['.', ' ']])
            new_val = f"{new_val} {unit}"
        new_answers[ent] = new_val
    return new_answers


def temporal_counterfactual(orig_answers, evidence, objective='earliest'):
    entities = list(orig_answers.keys())
    parsed_dates = {}
    original_values = {}

    for ent, val in orig_answers.items():
        original_values[ent] = val
        try:
            parsed_dates[ent] = parser.parse(str(val), fuzzy=True)
        except:
            parsed_dates[ent] = None  # fallback for unparseable

    # Determine the extreme date
    extreme_date = min(parsed_dates.values()) if objective in ['earliest', 'first'] else max(parsed_dates.values())

    # Assign extreme to evidence
    new_dates = [d for e, d in parsed_dates.items() if e != evidence]
    random.shuffle(new_dates)

    new_dates.insert(0, extreme_date) if objective in ['earliest', 'first'] else new_dates.append(extreme_date)

    # Map back to entities
    result = {}
    other_entities = [e for e in entities if e != evidence]
    for ent, d in zip(other_entities, new_dates):
        result[ent] = original_values[ent]  # keep original formatting

    result[evidence] = original_values[evidence]  # assign extreme (keep formatting)
    return result

def rotate_counterfactual(orig_answers, memory, evidence):
    """
    Create a counterfactual mapping where the evidence entity receives the
    value originally belonging to the memory entity. Remaining values are rotated
    randomly to avoid trivial correspondences.
    """
    new_mapping = copy.deepcopy(orig_answers)

    entities = list(orig_answers.keys())
    values = list(orig_answers.values())

    # Identify source values
    if memory not in orig_answers:
        raise KeyError(f"Memory '{memory}' not found in original answer keys.")
    if evidence not in orig_answers:
        raise KeyError(f"Evidence '{evidence}' not found in original answer keys.")

    memory_value = orig_answers[memory]

    # Remove evidence from rotation pool (we assign it manually)
    entities_without_evidence = [e for e in entities if e != evidence]
    values_without_memory = [v for k, v in orig_answers.items() if k != memory]

    # Shuffle values for randomness
    random.shuffle(values_without_memory)

    # Assign values to remaining entities
    for ent, val in zip(entities_without_evidence, values_without_memory):
        new_mapping[ent] = val

    # Now enforce rule: evidence gets memory's value
    new_mapping[evidence] = memory_value

    return new_mapping


def aggregation_counterfactual(orig_answers, evidence):
    """
    Adjust aggregation answers so that `evidence` becomes the majority answer.
    Each unique answer appears at least once. Evidence gets more than half of the remaining slots.
    """
    entities = list(orig_answers.keys())
    n = len(entities)

    # Get unique answers
    unique_answers = list(set(orig_answers.values()))
    if evidence not in unique_answers:
        unique_answers.append(evidence)

    # Assign minimum 1 occurrence for each answer
    freq = {ans: 1 for ans in unique_answers}

    remaining = n - sum(freq.values())

    if remaining > 0:
        # Give the majority of remaining to evidence
        evidence_extra = remaining // 2 + 1
        freq[evidence] += evidence_extra
        remaining -= evidence_extra

        # Assign rest randomly to other answers
        other_answers = [ans for ans in unique_answers if ans != evidence]
        for _ in range(remaining):
            chosen = random.choice(other_answers)
            freq[chosen] += 1

    # Build the pool
    pool = []
    for ans, count in freq.items():
        pool.extend([ans] * count)

    random.shuffle(pool)

    # Assign back to entities
    return {ent: pool[i] for i, ent in enumerate(entities)}


def generate_counterfactual(sample):
    reasoning = sample['reasoning_type']
    evidence = sample['answer']['evidence']
    memory = sample['answer']['memory']
    orig = sample['original answer']

    if 'Extremum' in reasoning:
        # Determine if max or min
        objective = 'max' if 'max' in reasoning.lower() or 'highest' in reasoning.lower() else 'min'
        return rotate_counterfactual(orig, memory, evidence)
    elif 'Temporal' in reasoning:
        objective = 'earliest' if 'earliest' in reasoning.lower() or 'first' in reasoning.lower() else 'latest'
        return rotate_counterfactual(orig, memory, evidence)
    elif 'Aggregation' in reasoning:
        return aggregation_counterfactual(orig, evidence)
    else:
        return None  # fallback


def create_counterfactuals(samples):
    """Main entry point: returns list of samples with counterfactual original answers."""
    cf_samples = copy.deepcopy(samples)
    for s in cf_samples:
        s['original answer'] = generate_counterfactual(s)
    return cf_samples


def replace_sub_answers_with_original(samples):
    """
    For each sample, replace the sub-question's sub-answer with the corresponding value
    from 'original answer', based on the entity name appearing in the sub-question text.

    Args:
        samples (list): List of dictionaries with keys 'original answer' and 'sub-questions'

    Returns:
        list: New list of samples with updated sub-answers
    """
    updated_samples = copy.deepcopy(samples)

    for sample in updated_samples:
        orig_answers = sample.get('original answer', {})
        for sub_q in sample.get('sub-questions', []):
            sq_text = sub_q.get('sub-question', '')
            # Check which key from original answers is mentioned in the sub-question
            for entity, value in orig_answers.items():
                if entity in sq_text:
                    sub_q['sub-answer'] = value
                    break  # assume only one entity per sub-question

    for sample in updated_samples:
        del sample['original answer']
        del sample['all options']
        sample['evidences'] = []
        for subq in sample['sub-questions']:
            sample['evidences'].append(f"{subq['sub-question']} --> {subq['sub-answer']}")
        del sample['sub-questions']

    return updated_samples
