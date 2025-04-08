"""
This module provides functions for clustering various types of structured data.

The main purpose of this module is to process datasets containing questions and
their corresponding answers and organize them into clusters based on context-specific
metrics. Each function is specialized to handle different types of data and answer
categories, including textual entities, numerical values, and date-related answers.

Functions:
    1. cluster_wiki_entities:
        Clusters questions by their textual answers, ensuring questions
        with identical or similar answers are grouped together.

    2. cluster_numerical_values:
        Clusters numerical answers (e.g., years, numerical spans) into predefined
        bins or ranges based on their values.

    3. cluster_date_values:
        Clusters date-related questions based on their answer type and context,
        using specific logic for categories such as years, months, and decades.
"""

NUMERICAL_BINS = [
    (1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 500), (501, 1000), (1001, 5000), (5001, 10_000),
    (10_001, 50_000), (50_001, 100_000), (100_001, 500_000), (500_001, 1_000_000), (1_000_001, 10_000_000),
    (10_000_001, 100_000_000), (100_000_001, 1_000_000_000), (1_000_000_001, float("inf"))
]


def cluster_general_entities(dataset: list[dict]) -> dict:
    """
    Clusters Wikipedia entities from a dataset of questions and answers.

    This function takes a dataset containing a list of dictionaries with
    "Answer", "Question" and "Type" keys, and clusters questions based
    on their corresponding answer-type pairs. Answers that appear only
    once in the dataset are filtered out.

    Args:
        dataset (list[dict]): A list of dictionaries, where each dictionary
            represents a sample with the keys:
            - 'Answer' (str): The answer text.
            - 'Question' (str): The question text.
            - 'Type' (str): The answer's entity type.

    Returns:
        dict: A dictionary where each key is an answer-type pair (tuple),
        and the value is a list of questions (list[str]) associated with
        that answer. Only answers with at least two associated questions
        are included in the returned dictionary.
    """
    # Get answers:
    answers = [sample['Answer'] for sample in dataset]
    questions = [sample['Question'] for sample in dataset]
    answer_types = [sample['Type'] for sample in dataset]

    # Initialize output dictionary:
    clustered_dataset = {}
    for answer in list(set(answers)):
        for answer_type in list(set(answer_types)):
            clustered_dataset[answer, answer_type] = []

    # Cluster questions by answer:
    for answer, question, answer_type in zip(answers, questions, answer_types):
        clustered_dataset[answer, answer_type].append(question)

    # Remove answers that only appear once in the dataset:
    filtered_clustered_dataset = {}
    for key, value in clustered_dataset.items():
        if len(value) >= 3:
            filtered_clustered_dataset[key] = value

    return filtered_clustered_dataset


def cluster_numerical_values(dataset: list[dict]) -> dict:
    """
    Clusters numerical questions into predefined numerical bins.

    This function processes a dataset containing questions and their corresponding
    numerical answers. It groups the questions into numerical bins based on the
    normalized numerical answers, which are assumed to be integers. The bins are
    specified in the `NUMERICAL_BINS` variable.

    Args:
        dataset (list[dict]): A list of dictionaries, where each dictionary contains:
            - 'Question' (str): The question text.
            - 'Answer' (str): The numerical answer as a dict with 'Normalized' key.
            - 'Type' (str): The type of the answer.

    Returns:
        dict: A dictionary where the keys are numerical bins represented as tuples
        of the form ((lower_bound, upper_bound), type), and the values are lists of questions
        (list[str]) that fall within each bin.

    """
    # Get QA pairs:
    qa_pairs = [(sample['Question'], sample['Answer']['Normalized'], sample['Type']) for sample in dataset]
    types = list(set([sample['Type'] for sample in dataset]))

    # Initialize output dictionary:
    clustered_dataset = {}
    for answer in NUMERICAL_BINS:
        for answer_type in types:
            clustered_dataset[answer, answer_type] = []

    # Cluster questions by answer span:
    for pair in qa_pairs:
        answer = int(pair[1])
        question = pair[0]
        ans_type = pair[2]
        for lower, upper in NUMERICAL_BINS:
            if lower <= answer <= upper:
                clustered_dataset[(lower, upper), ans_type].append(question)
                break

    # Make sure there are no empty clusters:
    keys_to_remove = []
    for key, value in clustered_dataset.items():
        if len(value) < 2:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del clustered_dataset[key]

    return clustered_dataset


def cluster_date_values(dataset: list[dict]) -> dict:
    """
    Clusters date-related questions based on answer types and context-specific metrics.

    This function processes a dataset of questions and their corresponding normalized
    date-related answers, and clusters them by categories such as 'generic_day',
    'generic_month', 'decade', 'century', 'full_date', 'year', 'month_year', and
    'day_month'. The clustering logic is customized for each category:
      - Name-based categories ('generic_day', 'generic_month', 'decade', 'century')
        are clustered by exact match of their values.
      - Year-based categories ('full_date', 'year', 'month_year') are clustered by
        their corresponding year.
      - Month-based categories ('day_month') are clustered by month.

    Args:
        dataset (list[dict]): A list of dictionaries, where each dictionary contains:
            - 'Question' (str): The question text.
            - 'Answer' (dict): The normalized representation of the answer, with:
                - 'category' (str): The category of the answer (e.g., 'generic_day', 'year').
                - 'parsed_value' (str): The parsed answer value.

    Returns:
        dict: A dictionary where keys represent clusters and values are lists of questions
        belonging to each cluster. Keys are:
            - For name-based categories: The exact answer string.
            - For year-based categories: A tuple of the form (category, year).
            - For month-based categories: A tuple of the form ('day_month', month).
    """
    # Cluster by type:
    type_clustered_values = {}
    ans_types = list(set([sample['Type'][1] for sample in dataset]))
    for ans_type in ans_types:
        type_clustered_values[ans_type] = []
    for sample in dataset:
        type_clustered_values[sample['Type'][1]].append({
            'Question': sample['Question'],
            'Answer': sample['Answer']['Normalized']
        })

    # Cluster per type use case:
    clustered_values = {}

    name_category_types = ['generic_day', 'generic_month', 'decade', 'century'] # Categories clustered by name
    for key in name_category_types:
        for sample in type_clustered_values[key]:
            if sample['Answer'] in clustered_values.keys():
                clustered_values[sample['Answer']].append(sample['Question'])
            else:
                clustered_values[sample['Answer']] = [sample['Question']]

    by_year_types = ['full_date', 'year', 'month_year'] # Categories clustered by year
    for key in by_year_types:
        for sample in type_clustered_values[key]:
            try:
                year = int([word for word in sample['Answer'].split() if word.isdigit() and len(word) == 4][0])
            except:
                year = int(sample['Answer'])
            if (key, year) in clustered_values.keys():
                clustered_values[(key, year)].append(sample['Question'])
            else:
                clustered_values[(key, year)] = [sample['Question']]

    for sample in type_clustered_values['day_month']: # Categories clustered by month
        month = sample['Answer'].split()[1]
        if ('day_month', month) in clustered_values.keys():
            clustered_values[('day_month', month)].append(sample['Question'])
        else:
            clustered_values[('day_month', month)] = [sample['Question']]

    # Note: For now we will not cluster year-spans type answers, will go for that if it will be required.

    return clustered_values
