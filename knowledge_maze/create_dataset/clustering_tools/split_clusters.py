"""Sentence Clustering and Refinement Module.

This module provides tools for clustering sentences, particularly question-answer
pairs, based on size constraints and embedding similarity using FAISS K-Means.
It includes functions to:

1.  Split initial clusters based purely on minimum and maximum size limits,
    using randomized chunking (`split_clusters`). Note: This function does not
    use embeddings and its merging capability described in its docstring might
    be inaccurate based on implementation.
2.  Cluster a list of sentences into semantically similar groups using FAISS
    K-Means based on their embeddings (`cluster_sentences_kmeans`). This involves
    estimating the optimal number of clusters (`k`), running K-Means, and then
    post-processing the resulting clusters to split any that exceed the defined
    maximum size into smaller chunks within the desired size range.
3.  Refine pre-existing clusters (e.g., questions grouped by answer value) by
    applying the embedding-based K-Means clustering (`cluster_sentences_kmeans`)
    to each group individually (`split_num_clusters`). This further splits larger
    groups into semantically coherent sub-clusters.

The module relies on FAISS for efficient clustering, NumPy for numerical
operations, and PyTorch for handling embedding tensors (which are converted
to NumPy arrays for FAISS). Progress bars are provided via `tqdm`.

Typical usage involves potentially pre-grouping questions (e.g., by answer),
then using `split_num_clusters` along with an embedding dictionary to refine
these groups into smaller, semantically related clusters constrained by size.
"""
import math
import random

import torch
import faiss
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Optional


def split_clusters(data: dict, max_size:int = 9, min_size:int = 2) -> list[dict]:
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


def cluster_sentences_kmeans(sentences: List[str],
                            embeddings_dict: Dict[str, torch.tensor],
                            min_cluster_size: int = 2,
                            max_cluster_size: int = 9,
                            n_iterations: int = 20, ) -> Optional[List[List[str]]]:
    """
    Clusters sentences using FAISS K-Means.

    Estimates k, runs K-Means, and returns clusters.

    Args:
        sentences: An ordered list of sentence strings.
        embeddings_dict: Dict mapping sentences to their embedding vectors.
        min_cluster_size: Minimum desired size (used to estimate k).
        max_cluster_size: Maximum desired size (used to estimate k).
        n_iterations: Number of iterations for K-Means.

    Returns:
        A list of lists (clusters of sentences), or None on critical error.
    """
    # Basic input validation
    if not sentences or not embeddings_dict or min_cluster_size < 1 or min_cluster_size > max_cluster_size:
        print("Error: Invalid input (empty data or invalid size constraints).")
        return None

    # 1. Prepare embeddings array and track valid sentences corresponding to rows
    embeddings_list = []
    valid_sentences = []
    embedding_dim = None

    for sentence in sentences:
        emb = embeddings_dict.get(sentence)
        if emb is not None:
            try:
                emb_np = np.asarray(emb.cpu(), dtype=np.float32)
                if embedding_dim is None:
                    embedding_dim = emb_np.shape[0]
                    if embedding_dim == 0: raise ValueError("Embedding dim is 0")
                elif emb_np.shape[0] != embedding_dim:
                    raise ValueError(f"Inconsistent embedding dims: {embedding_dim} vs {emb_np.shape[0]}")

                embeddings_list.append(emb_np)
                valid_sentences.append(sentence)
            except Exception as e:
                print(f"Warning: Skipping sentence due to embedding error ('{sentence[:50]}...'): {e}")

    if not valid_sentences or embedding_dim is None:
        print("Error: No valid embeddings found.")
        return None

    embeddings_array = np.vstack(embeddings_list)
    n_valid_sentences = embeddings_array.shape[0]

    # 2. Estimate k
    avg_size = (min_cluster_size + max_cluster_size) / 2
    k = max(1, min(int(math.ceil(n_valid_sentences / avg_size)), n_valid_sentences))

    # 3. Run FAISS K-Means (CPU)
    kmeans = faiss.Kmeans(d=embedding_dim, k=k, niter=n_iterations, gpu=False)  # Explicitly CPU
    try:
        kmeans.train(embeddings_array)
    except Exception as e:
        print(f"Error during FAISS K-Means training: {e}")
        return None

    # 4. Assign labels
    try:
        _, labels = kmeans.index.search(embeddings_array, 1)
        labels = labels.flatten()
    except Exception as e:
        print(f"Error during FAISS label assignment: {e}")
        return None

    # 5. Group sentences
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(valid_sentences[i])

    initial_clusters = list(clusters.values())  # Clusters from K-Means

    # 6. Post-process to split over-sized clusters
    final_processed_clusters = []

    for cluster in initial_clusters:
        if min_cluster_size <= len(cluster) <= max_cluster_size:
            # Keep clusters that are already within the max size limit
            final_processed_clusters.append(cluster)
        else:
            # Split over-sized clusters
            current_sub_cluster = list(cluster)  # Work on a copy
            while len(current_sub_cluster) > max_cluster_size:  # Continue as long as there are enough sentences left

                chunk_size = random.randint(int(avg_size), max_cluster_size)
                chunk_size = min(chunk_size, len(current_sub_cluster))  # Don't take more than available

                # Take the chunk
                new_chunk = current_sub_cluster[:chunk_size]
                # Add the chunk if it's not empty (shouldn't be with while condition)
                if new_chunk:
                    final_processed_clusters.append(new_chunk)
                # Update the remaining part
                current_sub_cluster = current_sub_cluster[chunk_size:]

                # Safety break if chunk_size somehow becomes 0 (shouldn't happen)
                if chunk_size == 0:
                    print("Warning: chunk_size became 0 during splitting, breaking loop.")
                    if current_sub_cluster:  # Add any remaining if break occurs unexpectedly
                        final_processed_clusters.append(current_sub_cluster)
                    break

    return final_processed_clusters


def split_num_clusters(clusters: dict, embeddings_dict: Dict[str, torch.tensor]):
    """
    Splits clusters of sentences with numerical answers into smaller sub-clusters using KMeans clustering,
    and returns the results as a list of dictionaries, where each dictionary contains
    an answer and the corresponding sub-clustered questions.

    Args:
        clusters (dict): A dictionary where keys are answers and values are lists of sentences (questions).
        embeddings_dict (Dict[str, torch.Tensor]): A dictionary mapping sentences to their embedding tensors.

    Returns:
        List[dict]: A list of dictionaries, each containing:
            - 'Answer' (str): The original answer.
            - 'Questions' (List[str]): A sub-cluster of related questions.
    """
    final_result = []

    for answer, sentences in tqdm(clusters.items(), desc='Splitting large clusters'):
        separated_clusters = cluster_sentences_kmeans(sentences, embeddings_dict)

        for cluster in separated_clusters:
            final_result.append({'Answer': answer, 'Questions': cluster})

    return final_result
