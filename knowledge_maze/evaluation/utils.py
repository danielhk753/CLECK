import random

from knowledge_maze.evaluation.ask_llm import ask_llm_from_memory, judge_answer


ANSWER_TYPES = {
    'ENT': ['PERSON', 'PLACE', 'ORG', 'EVENT', 'LANGUAGE', 'WORK_OF_ART'],
    'NUM': ['DATE', 'NUMERICAL']
}


def sample_by_exact_path_length(data, len1, len2, sample_size):
    """
    Samples `sample_size` random items from `data` where
    Reasoning path 1 length == len1 and Reasoning path 2 length == len2.

    Args:
        data (list[dict]): The dataset.
        len1 (int): Required length of Reasoning path 1.
        len2 (int): Required length of Reasoning path 2.
        sample_size (int): Number of samples to return.

    Returns:
        list[dict]: Sampled items.
    """
    # Filter to matching lengths
    filtered = [
        item for item in data
        if len(item.get("Reasoning path 1", [])) == len1
        and len(item.get("Reasoning path 2", [])) == len2
    ]

    # Sample without replacement
    if len(filtered) < sample_size:
        raise ValueError(f"Not enough samples with lengths ({len1}, {len2}) — only {len(filtered)} found.")

    return random.sample(filtered, sample_size)


def create_test_set(data, sum_per_length, answer_type=None):
    """
    Creates a test set by sampling X items from each possible (len1, len2) reasoning length pair.

    Args:
        data (list[dict]): The dataset.
        sum_per_length (int): Number of samples to take from each length pair.
        answer_type (str, optional): 'ENT' or 'NUM' to filter by answer type.

    Returns:
        list[dict]: The combined sampled test set.
    """
    # Optionally filter by answer type
    if answer_type:
        data = [
            item for item in data
            if item.get("Answer type") in ANSWER_TYPES[answer_type]
        ]

    # Find all unique (len1, len2) pairs
    length_pairs = set(
        (len(item.get("Reasoning path 1", [])), len(item.get("Reasoning path 2", [])))
        for item in data
    )

    # Create final test set
    test_set = []
    for len1, len2 in length_pairs:
        try:
            samples = sample_by_exact_path_length(
                data, len1, len2, sum_per_length
            )
            test_set.extend(samples)
        except ValueError:
            # Skip if not enough samples for this pair
            continue

    return test_set


async def get_memory_answers_results(dataset):
    memory_answers_results = []
    for sample in dataset:
        question = sample['Question']
        answers = await ask_llm_from_memory(question, num_retries=50)
        summary_results = await judge_answer(answers, options=sample['Answer'])
        memory_answers_results.append(summary_results)

    return memory_answers_results