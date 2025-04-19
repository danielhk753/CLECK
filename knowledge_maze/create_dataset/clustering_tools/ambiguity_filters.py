"""
Filters question clusters based on ambiguity using LLM classification.

This module provides functions to filter dictionaries of question clusters based
on the perceived ambiguity or time-sensitivity of questions, or the specificity
of the answers (cluster keys).

It leverages an external LLM API tool (`gptAPI`) configured with specific
prompts to classify ambiguity or time-sensitivity. Three main filtering
functions are provided:

1.  `question_ambiguous_filter`: Removes questions from clusters if the questions
    are deemed too vague, lack necessary context, contain the answer, or
    present options.
2.  `answer_ambiguous_filter`: Removes entire clusters if the cluster key (the answer,
    typically representing a place, organization, or location) is considered too
    vague or could refer to multiple distinct entities.
3.  `date_ambiguous_filter`: Removes questions from clusters if the questions
    are identified as being time-sensitive (referencing a specific timeframe or
    having an answer that changes over time).

All functions take a dictionary of clusters as input and return a new,
filtered dictionary or modify the cluster lists in place (depending on
implementation details - check individual function docstrings).

Attributes:
    AMBIGUOUS_QUESTION_PROMPT (str): The prompt instructing the LLM on how to
        classify general question ambiguity.
    AMBIGUOUS_ANSWER_PROMPT (str): The prompt instructing the LLM on how to
        classify the specificity/ambiguity of an answer string (typically a
        place, organization, or location).
    AMBIGUOUS_DATE_PROMPT (str): The prompt instructing the LLM on how to
        classify questions as time-sensitive.
"""
from knowledge_maze.tools.api_tools import gptAPI

# Initialize prompts for ambiguity filters:
AMBIGUOUS_QUESTION_PROMPT = """Classify whether the given question is too vague to understand what it is asking about, 
                      even if an answer were provided. Mark as "Ambiguous" only if key details are missing (e.g., 
                      unclear subject or reference), making the question unanswerable. Do not mark as ambiguous 
                      just because multiple interpretations exist, only if they are very different and lead to 
                      different answers.

                      Also return 'True' if:
                      - The answer is too obvious or contained in the question itself.
                      - The question includes answer options.

                      Return 'True' for Ambiguous, 'False' for Clear."""
AMBIGUOUS_ANSWER_PROMPT = """Determine whether the following place, organization, or location refers to a specific, 
                             clearly identifiable entity. Return 'False' if the reference is vague or ambiguous and 
                             could point to multiple possible entities. Otherwise, return 'True'."""
AMBIGUOUS_DATE_PROMPT = """Classify whether the question is time-sensitive in one of the following two ways:\n
                           
                           1. The question explicitly mentions a specific timeframe (e.g., "the 2017 season", 
                           "next year", "this month").\n
                           2. The answer would vary depending on when the question is asked (e.g., "When is the next 
                           launch?", "When does the school year begin?", "When was the last time that death penalty was 
                           given in the US?").\n
                           
                           If either condition applies, return True.
                           Otherwise (e.g., for past or recurring events), return False."""


async def question_ambiguous_filter(clusters: dict) -> dict:
    """
    Filters out ambiguous questions from the given clusters. A question is considered ambiguous if it lacks key
    details (e.g., missing subject, unclear reference) that make it impossible to determine what is being asked,
    even if an answer were provided.

    Args:
        clusters (dict): A dictionary where the keys are cluster identifiers and the values are lists of questions
                          within those clusters.

    Returns:
        dict: A new dictionary of clusters with ambiguous questions removed
    """

    # Create LLM API call objects:
    question_classifier = gptAPI(AMBIGUOUS_QUESTION_PROMPT,
                                 output_format='bool',
                                 temperature=0.2,
                                 max_concurrent_requests=100,
                                 max_rpm=4900)
    # Extract questions:
    questions = []
    for cluster in clusters.values():
        for question in cluster:
            questions.append(question)

    # Run the classification to check if questions are ambiguous:
    results = await question_classifier.async_run(questions)

    # Map the results to questions:
    question_results = {}
    for question, result in zip(questions, results):
        question_results[question] = result

    # Create a copy of clusters to filter out ambiguous questions:
    filtered_clusters = clusters.copy()
    for cluster in filtered_clusters.values():
        cluster[:] = [question for question in cluster if not question_results[question]]

    return filtered_clusters


async def answer_ambiguous_filter(clusters: dict) -> dict:
    """
    Filters out questions with ambiguous answers from the given clusters.

    Args:
        clusters (dict): A dictionary where the keys are cluster identifiers and the values are lists of questions
                          within those clusters.

    Returns:
        dict: A new dictionary of clusters with ambiguous questions removed
    """

    # Create LLM API call objects:
    answer_classifier = gptAPI(AMBIGUOUS_ANSWER_PROMPT,
                               output_format='bool',
                               temperature=0.2,
                               max_concurrent_requests=100,
                               max_rpm=4900)
    # Extract answers:
    answers = [answer[0] for answer in clusters.keys()]

    # Run the classification to check if answer are ambiguous:
    results = await answer_classifier.async_run(answers)

    # Map the results to answers:
    answer_results = {}
    for answer, result in zip(answers, results):
        answer_results[answer] = result

    # Create a copy of clusters to filter out ambiguous answers:
    filtered_clusters = {}
    for answer, cluster in clusters.items():
        if answer_results[answer[0]]:
            filtered_clusters[answer] = cluster

    return filtered_clusters


async def date_ambiguous_filter(clusters: dict) -> dict:
    """
    Filters out time-sensitive questions from the given clusters.

    Identifies questions that either explicitly mention a timeframe or whose
    answer depends on *when* the question is asked, based on LLM classification
    using `AMBIGUOUS_DATE_PROMPT`. It removes such time-sensitive questions
    from the question lists within the clusters.

    Args:
        clusters (dict): A dictionary where keys are cluster identifiers and
            values are lists of question strings within those clusters.

    Returns:
        dict: A dictionary containing the clusters with time-sensitive questions
              removed from their question lists. Note: This function modifies
              the input dictionary's lists in place due to `cluster[:] = ...`.
              Consider returning a new dictionary for functional purity if needed.
    """

    # Create LLM API call objects:
    date_classifier = gptAPI(AMBIGUOUS_DATE_PROMPT,
                             output_format='bool',
                             temperature=0.2,
                             max_concurrent_requests=100,
                             max_rpm=4900)
    # Extract questions:
    questions = []
    for cluster in clusters.values():
        for question in cluster:
            questions.append(question)

    # Run the classification to check if questions are ambiguous:
    results = await date_classifier.async_run(questions)

    # Map the results to questions:
    question_results = {}
    for question, result in zip(questions, results):
        question_results[question] = result

    # Create a copy of clusters to filter out ambiguous questions:
    filtered_clusters = clusters.copy()
    for cluster in filtered_clusters.values():
        cluster[:] = [question for question in cluster if not question_results[question]]

    return filtered_clusters
