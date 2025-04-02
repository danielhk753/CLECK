"""
This module provides tools for clustering, analyzing, and visualizing relationships
between questions and their answers using embeddings and graph-based techniques.

The functions in this module support the generation of sentence embeddings,
clustering of questions based on semantic similarity, and filtering redundant
questions using vertex cover algorithms on similarity graphs. Additionally,
it includes utilities for splitting clusters into manageable sizes and
visualizing question distributions.

Functions:
    - generate_embeddings: Generates sentence embeddings using a pretrained model.
    - generate_question_embeddings: Creates embeddings for questions in a dataset in batches.
    - generate_sentence_similarity_graph: Constructs a similarity graph based on sentence embeddings.
    - plot_sentence_graph: Visualizes a sentence similarity graph with edge weights.
    - remaining_nodes_after_vertex_cover: Filters redundant nodes in a graph using a vertex cover algorithm.
    - filter_question_groups: Refines clusters of questions based on sentence similarity.
    - filter_ambiguous_questions: Filter out ambiguous questions using GPT api.
    - split_clusters: Splits clusters into smaller uniform groups while handling small clusters.
    - create_paths: Run the creation pipeline.
    - display_random_sample: Randomly samples an answer-question pair and visualizes the filtering process.
    - plot_clusters_distribution: Plots histograms of question distributions across clusters.
"""

import json
import random

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

from knowledge_maze.tools.api_tools import gptAPI
from knowledge_maze.create_dataset.clustering_tools.similarity_graphs import filter_question_groups
from knowledge_maze.create_dataset.clustering_tools.cluster_questions import cluster_general_entities, cluster_date_values, cluster_numerical_values


def divide_by_type(dataset: list[dict]) -> dict[str, list[dict]]:
    """
        Divides a dataset into groups based on the type of answer.

        This function processes a dataset of question-answer pairs and categorizes the data
        into three groups:
            - "Date": Questions with answers classified as date-related types.
            - "Number": Questions with numerical answers classified as "CARDINAL", "MONEY", or "PERCENT".
            - "General": Questions with other types of answers not falling into the above categories.

        Args:
            dataset (list[dict]): A list of dictionaries, where each dictionary contains:
                - 'Question': The question string.
                - 'Answer': The answer string.
                - 'Type': The entity type or classification of the answer.

        Returns:
            dict[str, list[dict]]: A dictionary with three keys ("Date", "Number", "General"),
            each mapping to a list of dictionaries containing the categorized question-answer pairs.
        """
    # Lists to hold categorized data
    ent_data = []  # General answers
    num_data = []  # Numerical answers
    date_data = []  # Date-related answers

    # Categorize the dataset
    for sample in dataset:
        # Date-related answers have a type stored as a list (e.g., ["DATE", "year"])
        try:
            if isinstance(sample['Type'], list | tuple):
                date_data.append(sample)
            # Numerical answers with types like "CARDINAL", "MONEY", or "PERCENT"
            elif sample['Type'] in {"CARDINAL", "MONEY", "PERCENT"}:
                num_data.append(sample)
            # All other types are classified as "General"
            else:
                ent_data.append(sample)
        except:
            print(sample['Type'])
            raise ValueError

    # Return the categorized dataset as a dictionary
    return {
        "Date": date_data,
        "Number": num_data,
        "General": ent_data
    }


def generate_embeddings(sentences: list[str],
                        model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> torch.Tensor:
    """
    Generate sentence embeddings using a pretrained model with GPU acceleration.

    Args:
        sentences (list[str]): A batch of sentences.
        model_name (str): Name of the pretrained model to use.

    Returns:
        torch.Tensor: A tensor of sentence embeddings.
    """
    # Load model and move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name).to(device)

    # Generate embeddings
    with torch.no_grad():
        embeddings = model.encode(sentences, convert_to_tensor=True, device=device)

    torch.cuda.empty_cache()
    return embeddings


def generate_question_embeddings(answer_dict, batch_size=256):
    """
    Generates embeddings for all questions in the given dictionary in batches.

    Args:
        answer_dict (dict): A dictionary where keys are answers (str), and values are lists of questions (str).
        batch_size (int): The number of questions processed per batch.

    Returns:
        dict: A dictionary where keys are questions (str), and values are their corresponding embeddings (torch.Tensor).
    """
    all_questions = [q for questions in answer_dict.values() for q in questions]  # Flatten questions
    question_embeddings = {}

    for i in tqdm(range(0, len(all_questions), batch_size), desc='Generating question embeddings'):
        batch = all_questions[i : i + batch_size]  # Get batch of questions
        batch_embeddings = generate_embeddings(batch)  # Get embeddings

        for q, emb in zip(batch, batch_embeddings):
            question_embeddings[q] = emb

    return question_embeddings


async def filter_ambiguous_questions(clusters: dict) -> dict:
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
    ambiguous_prompt = """Classify whether the given question is too vague to understand what it is asking about, even 
                          if an answer were provided. A question should be marked as "Ambiguous" only if it lacks key 
                          details (e.g., missing subject, unclear reference) that make it impossible to determine what 
                          is being asked. Otherwise, return "Not Ambiguous." Do not mark questions as ambiguous just 
                          because they allow multiple interpretations.\n
 
                          Return 'True' for 'Ambiguous' or 'False' for 'Clear'."""
    # Create LLM API call objects:
    ambiguous_classifier = gptAPI(ambiguous_prompt,
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
    results = await ambiguous_classifier.async_run(questions)

    # Map the results to questions:
    question_results = {}
    for question, result in zip(questions, results):
        question_results[question] = result

    # Create a copy of clusters to filter out ambiguous questions:
    filtered_clusters = clusters.copy()
    for cluster in filtered_clusters.values():
        cluster[:] = [question for question in cluster if not question_results[question]]

    return filtered_clusters


def split_clusters(data: dict, max_size:int = 10, min_size:int = 2) -> list[dict]:
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
        if len(questions) <= max_size:
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


async def create_paths(processed_data: list[dict], output_path: str = None) -> list[dict]:
    """
    Processes input question clusters and generates reasoning chains.

    This function performs the following steps:
    1. Generates embeddings for all entity questions in the input clusters.
    2. Filters questions based on a similarity graph.
    3. Removes ambiguous questions from the clusters.
    4. Splits the remaining question clusters into smaller groups with a maximum size of 10.
    5. Optionally saves the resulting reasoning chains to a specified file path in JSON format.

    Args:
        processed_data (list[dict]):
        output_path (str, optional): The file path where the reasoning chains will be saved as
            a JSON file. Defaults to None.

    Returns:
        list[dict]: A list of dictionaries representing the reasoning chains. Each dictionary
        contains the processed and structured reasoning paths for the input clusters.
    """
    # Divide processed data by answer types:
    print("Dividing data by type...")
    divided_data = divide_by_type(processed_data)
    general_data = divided_data['General']
    numerical_data = divided_data['Number']
    date_data = divided_data['Date']

    # Cluster data:
    print("Clustering data...")
    general_clusters = cluster_general_entities(general_data[:2000])  # TODO: Remove this. -------------------------------
    numerical_clusters = cluster_numerical_values(numerical_data)
    date_clusters = cluster_date_values(date_data)

    reasoning_chains = []

    for name, clusters in {"general clusters": general_clusters,
                           "numerical clusters": numerical_clusters,
                           "date clusters": date_clusters}.items():
        print(f"Creating reasoning paths for {name}...")

        # Generate embeddings for all entity questions:
        questions_embeddings = generate_question_embeddings(clusters, batch_size=2048)

        # Filter questions based on similarity graph:
        filtered_clusters = filter_question_groups(clusters, questions_embeddings)

        # Filter out ambiguous questions:
        not_ambiguous_clusters = await filter_ambiguous_questions(filtered_clusters)

        # Split question clusters to a maximum size of 10:
        reasoning_chains += split_clusters(not_ambiguous_clusters)

    # Save in folder:
    if output_path:
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(reasoning_chains, out_file)

    return reasoning_chains


def plot_clusters_distribution(answer_dicts: dict[list]) -> None:
    """
    Plots histograms for three dictionaries, where keys are answers, and values are lists of questions.

    Args:
        answer_dicts (list of dict): A list of three dictionaries, where each key is a string answer,
                                     and the value is a list of question strings.
    """
    fig, ax = plt.subplots(2, int(len(answer_dicts) / 2),
                           figsize=(6*int(len(answer_dicts) / 2), 4*int(len(answer_dicts) / 2)), sharey=True)

    for i, ((name, question_cluster), ax) in enumerate(zip(answer_dicts.items(), ax.flatten())):
        lengths = [len(questions['Questions']) for questions in question_cluster]
        bins = np.arange(2, 12)

        counts, edges, patches = ax.hist(lengths, bins=bins, edgecolor="black", alpha=0.75)

        ax.set_title(f"{name}")
        ax.set_xlabel("Number of Questions per Answer")
        ax.set_ylabel("Frequency")
        ax.set_xticks(bins[::5])  # Show every 5th tick for clarity

        # Add labels above bars
        for count, edge, patch in zip(counts, edges[:-1], patches):
            if count > 0:
                ax.text(edge + 0.5, count, f"{int(count)}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.show()
