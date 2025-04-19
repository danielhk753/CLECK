"""Clusters questions based on answer characteristics (text, value, type).

This module processes datasets containing questions and answers, grouping
questions into clusters based on different criteria depending on the answer
type. It handles textual entities, numerical values, and various date formats.
LLM-based helper functions are used to enrich numerical and date questions
with semantic information (objective, category) before clustering.

Clustering Strategies:
- General Entities: Groups questions by exact answer string and answer type.
- Numerical Values: Groups questions based on the answer's value falling into
  predefined numerical ranges (`NUMERICAL_BINS`) and the LLM-identified
  'Objective' of the question.
- Date Values: Employs specific logic based on the date category:
    - Generic days/months: Clustered by category & LLM-identified date 'Category'.
    - Decades/Centuries: Clustered by the specific decade/century name & LLM date 'Category'.
    - Full dates, years, month-years: Clustered by decade & LLM date 'Category'.

The module aims to create meaningful groups of questions that relate to the
same answer concept, facilitating downstream tasks like reasoning path generation
or dataset analysis.

Functions:
    cluster_general_entities: Clusters by identical answer string and type.
    cluster_numerical_values: Clusters numbers into bins based on value and objective.
    cluster_date_values: Clusters dates using category-specific logic (decade, name, category).

Helper Functions (using LLM):
    get_numerical_object: Identifies the subject being counted/measured in numerical questions.
    get_date_category: Identifies the general domain/subject of date-related questions.

Attributes:
    NUMERICAL_BINS (List[Tuple]): Predefined ranges for clustering numerical values.
    NUMERICAL_PROMPT (str): Prompt used by `get_numerical_object` LLM call.
    CATEGORY_PROMPT (str): Prompt used by `get_date_category` LLM call.
"""
from knowledge_maze.tools.api_tools import gptAPI

NUMERICAL_BINS = [
    (1, 20), (21, 100), (101, 1000), (1001, 10_000), (10_001, 100_000), (100_001, 1_000_000), (1_000_001, 10_000_000),
    (10_000_001, 100_000_000), (100_000_001, 1_000_000_000), (1_000_000_001, float("inf"))
]
NUMERICAL_PROMPT = """You will be given a question that has a numerical answer. Your task is to write, in one phrase 
                      (maximum two words), what the question is asking to count or measure — that is, what is the thing 
                      we want to find the number of. Only output the short phrase.\n
                      \n
                      Examples:\n
                      \n
                      Q: How many countries are in South America?\n
                      A: countries\n
                      \n
                      Q: What is the population of France?\n
                      A: population\n
                      \n
                      Q: How many albums has Taylor Swift released?\n
                      A: albums\n
                      \n
                      Q: What is the total length of the Great Wall of China?\n
                      A: length\n
                      \n
                      Q: How many astronauts were on Apollo 11?\n
                      A: astronauts\n
                      \n
                      Q: What is the average height of NBA players?\n
                      A: height\n
                      \n
                      Q: What percentage of the Earth's surface is covered by water?\n
                      A: percentage\n
                      \n
                      Q: How much does a Tesla Model 3 cost in the US?\n
                      A: price\n
                      \n
                      Q: What is the annual revenue of Amazon?  \n
                      A: revenue\n
                      \n
                      Now process the next question:"""
CATEGORY_PROMPT = """You will be given a question that has a date or time as its answer.
                     Your task is to classify what the question is about using a short category label (maximum two words). 
                     The label should describe the **general subject or domain** of the event being asked about (e.g., "politics", "sports", "school").
                     Return only the category as a lowercase string.\n
                     \n
                     ---\n
                     \n
                     Examples:\n
                     \n
                     Q: When was Barack Obama born?\n
                     → birth date\n
                     \n
                     Q: When was the Eiffel Tower completed?\n
                     → construction\n
                     \n
                     Q: When did Halo 3 release?\n
                     → gaming\n
                     \n
                     Q: When is the next FIFA World Cup?\n
                     → sports\n
                     \n
                     Q: When was the Declaration of Independence signed?\n
                     → historical\n
                     \n
                     Q: When is the next Apple product launch?\n
                     → technology\n
                     \n
                     Q: When did the Berlin Wall fall?\n
                     → politics\n
                     \n   
                     Q: When is the school year starting in Canada?\n
                     → school\n
                     \n
                     ---\n
                     \n
                     Now categorize the following question:\n"""


async def get_numerical_object(dataset: list[dict]) -> list[dict]:
    """
    Identifies the numerical objective of questions via LLM classification.

    Adds an 'Objective' key with the LLM's classification result to each
    dictionary in the input dataset.

    Args:
        dataset: List of dictionaries, each containing a 'Question'.

    Returns:
        A list of dictionaries, modified to include the 'Objective' key.
    """
    # Create LLM API call objects:
    objectives_classifier = gptAPI(NUMERICAL_PROMPT,
                                   output_format='string',
                                   temperature=0.2,
                                   max_concurrent_requests=100,
                                   max_rpm=4900)
    # Extract questions:
    questions = [sample['Question'] for sample in dataset]

    # Run the classification to extract objectives from the questions:
    objectives = await objectives_classifier.async_run(questions)

    # Create a copy of clusters to filter out ambiguous questions:
    output_data = dataset.copy()
    for sample, objective in zip(output_data, objectives):
        sample['Objective'] = objective

    return output_data


async def get_date_category(dataset: list[dict]) -> list[dict]:
    """
    Identifies the category of date based questions via LLM classification.

    Adds a 'Category' key with the LLM's classification result to each
    dictionary in the input dataset.

    Args:
        dataset: List of dictionaries, each containing a 'Question'.

    Returns:
        A list of dictionaries, modified to include the 'Category' key.
    """
    # Create LLM API call objects:
    category_classifier = gptAPI(CATEGORY_PROMPT,
                                 output_format='string',
                                 temperature=0.2,
                                 max_concurrent_requests=100,
                                 max_rpm=4900)
    # Extract questions:
    questions = [sample['Question'] for sample in dataset]

    # Run the classification to extract objectives from the questions:
    categories = await category_classifier.async_run(questions)

    # Create a copy of clusters to filter out ambiguous questions:
    output_data = dataset.copy()
    for sample, objective in zip(output_data, categories):
        sample['Category'] = objective

    return output_data


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


async def cluster_numerical_values(dataset: list[dict]) -> dict:
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
    # Get question's objectives:
    tagged_dataset = await get_numerical_object(dataset)

    # Get QA pairs:
    qa_pairs = [(sample['Question'], sample['Answer']['Normalized'], sample['Objective']) for sample in tagged_dataset]
    objectives = list(set([sample['Objective'] for sample in tagged_dataset]))

    # Initialize output dictionary:
    clustered_dataset = {}
    for answer in NUMERICAL_BINS:
        for answer_objective in objectives:
            clustered_dataset[answer, 'NUMERICAL', answer_objective] = []

    # Cluster questions by answer span:
    for pair in qa_pairs:
        answer = int(pair[1])
        question = pair[0]
        ans_obj = pair[2]
        for lower, upper in NUMERICAL_BINS:
            if lower <= answer <= upper:
                clustered_dataset[(lower, upper), 'NUMERICAL', ans_obj].append(question)
                break

    # Make sure there are no empty clusters:
    keys_to_remove = []
    for key, value in clustered_dataset.items():
        if len(value) < 2:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del clustered_dataset[key]

    return clustered_dataset


async def cluster_date_values(dataset: list[dict]) -> dict:
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
    # Get question's objectives:
    tagged_dataset = await get_date_category(dataset)

    # Cluster by type:
    type_clustered_values = {}
    ans_types = list(set([sample['Type'][1] for sample in tagged_dataset]))
    for ans_type in ans_types:
        type_clustered_values[ans_type] = []
    for sample in tagged_dataset:
        type_clustered_values[sample['Type'][1]].append({
            'Question': sample['Question'],
            'Answer': sample['Answer']['Normalized'],
            'Category': sample['Category']
        })

    # Cluster per type use case:
    clustered_values = {}

    generic_category_types = ['generic_day', 'generic_month']  # Categories clustered by format
    for key in generic_category_types:
        for sample in type_clustered_values[key]:
            if ((key, sample['Category']), 'DATE') in clustered_values.keys():
                clustered_values[(key, sample['Category']), 'DATE'].append(sample['Question'])
            else:
                clustered_values[(key, sample['Category']), 'DATE'] = [sample['Question']]

    name_category_types = ['decade', 'century'] # Categories clustered by name
    for key in name_category_types:
        for sample in type_clustered_values[key]:
            if ((key, sample['Answer']), 'DATE') in clustered_values.keys():
                clustered_values[(key, sample['Answer'], sample['Category']), 'DATE'].append(sample['Question'])
            else:
                clustered_values[(key, sample['Answer'], sample['Category']), 'DATE'] = [sample['Question']]

    by_year_types = ['full_date', 'year', 'month_year']  # Categories clustered by decade
    for key in by_year_types:
        for sample in type_clustered_values[key]:
            try:
                # Extract the first 4-digit number as the year
                year_match = next((word for word in sample['Answer'].split() if word.isdigit() and len(word) == 4),
                                  None)
                if year_match:
                    year = int(year_match)
                else:  # Fallback if no 4-digit number found, try converting the whole answer
                    year = int(sample['Answer'])
                # Calculate the decade (integer division by 10, then multiply back)
                decade = (year // 10) * 10
            except (ValueError, TypeError):
                # Skip if year/decade cannot be determined
                continue

            cluster_key = ((key, decade, sample['Category']), 'DATE')  # Use decade in the key

            if cluster_key in clustered_values:
                clustered_values[cluster_key].append(sample['Question'])
            else:
                clustered_values[cluster_key] = [sample['Question']]

    # Note: For now we will not cluster year-spans type answers, will go for that if it will be required.

    return clustered_values
