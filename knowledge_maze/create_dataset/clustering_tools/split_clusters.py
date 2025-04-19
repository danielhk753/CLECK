"""Sentence Clustering and Refinement Module.

This module provides tools for clustering sentences, particularly question-answer
pairs, based on size constraints.

Split initial clusters based purely on minimum and maximum size limits,
using randomized chunking (`split_clusters`). Note: This function does not
use embeddings and its merging capability described in its docstring might
be inaccurate based on implementation.
"""
import random
from tqdm import tqdm


MAX_CLUSTER_SIZE = 7
MIN_CLUSTER_SIZE = 2


def split_clusters_random(data: dict, max_size:int = MAX_CLUSTER_SIZE, min_size:int = MIN_CLUSTER_SIZE) -> list[dict]:
    """
    Splits clusters of questions into smaller clusters of uniform size and merges
    clusters below the minimum size into existing clusters.

    Args:
        data (dict): A dictionary where keys are answers (str) and values are lists of questions (list[str]).
        max_size (int): The maximum allowable size of a cluster.
        min_size (int): The minimum allowable size of a cluster.

    Returns:
        list[dict]: A list of dictionaries, each with 'Answer' and 'Questions' keys.
    """
    result = []

    for answer, questions in tqdm(data.items(), desc='Splitting large clusters'):
        # If the cluster is already within the limit, add it as is
        if min_size <= len(questions) <= max_size:
            result.append({'Answer': answer, 'Questions': questions})
        else:
            # Split the cluster into smaller chunks
            while len(questions) > min_size:
                # Determine the size of the next chunk (randomized between min_size and max_size)
                chunk_size = random.randint(min_size, max_size) if len(questions) >= max_size else len(questions)
                chunk = questions[:chunk_size]  # Take the first 'chunk_size' questions
                questions = questions[chunk_size:]  # Remove the processed questions
                # Add the chunk as a new cluster
                result.append({'Answer': answer, 'Questions': chunk})

    return result
