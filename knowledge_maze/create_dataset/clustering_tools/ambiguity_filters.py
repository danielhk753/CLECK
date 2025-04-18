"""
Filters question clusters based on ambiguity using LLM classification.

This module provides functions to filter dictionaries of question clusters based
on the perceived ambiguity of either the questions themselves or the answers
(which often serve as cluster keys, especially for places/organizations).

It leverages an external LLM API tool (`gptAPI`) configured with specific
prompts to classify ambiguity. Two main filtering functions are provided:

1.  `question_ambiguous_filter`: Removes questions from clusters if the questions
    are deemed too vague, lack necessary context, contain the answer, or
    present options.
2.  `answer_ambiguous_filter`: Removes entire clusters if the cluster key (the answer,
    typically representing a place, organization, or location) is considered too
    vague or could refer to multiple distinct entities.

Both functions take a dictionary of clusters as input and return a new,
filtered dictionary.

Attributes:
    AMBIGUOUS_QUESTION_PROMPT (str): The prompt instructing the LLM on how to
        classify question ambiguity.
    AMBIGUOUS_ANSWER_PROMPT (str): The prompt instructing the LLM on how to
        classify the ambiguity of an answer string (typically a place,
        organization, or location).
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
