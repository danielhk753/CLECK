"""Dataset Processing and Reasoning Chain Generation Module.

This module provides a pipeline for processing question-answer datasets to
generate structured reasoning chains or paths. It leverages sentence embeddings,
semantic clustering, graph-based filtering, and large language models (LLMs)
via API calls to refine and structure the data.

Core functionalities include:
- Dividing datasets based on answer types (Date, Number, General).
- Generating sentence embeddings using Sentence Transformers (GPU accelerated).
- Clustering questions associated with the same answer or answer type based
  on their semantic similarity using various clustering algorithms.
- Filtering question groups within clusters to remove redundancy using
  similarity graphs and vertex cover algorithms.
- Filtering ambiguous questions using an external LLM API call (e.g., GPT).
- Splitting oversize clusters into smaller, manageable chunks according
  to specified size constraints, potentially using embeddings for refinement.
- Visualizing the distribution of final cluster sizes.
- Randomly selects a sample answer-question pair, visualizes the sentence
  similarity graph, and filters redundant questions using graph-based methods.

The main entry point is the `create_paths` asynchronous function, which
orchestrates these steps to transform a preprocessed dataset into a list
of refined question clusters (reasoning chains).
"""
import json
import random

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

from knowledge_maze.create_dataset.clustering_tools.similarity_graphs import filter_question_groups, \
    generate_sentence_similarity_graph, plot_sentence_graph, remaining_nodes_after_vertex_cover
from knowledge_maze.create_dataset.clustering_tools.split_clusters import split_clusters, split_num_clusters
from knowledge_maze.create_dataset.clustering_tools.ambiguity_filters import question_ambiguous_filter, answer_ambiguous_filter
from knowledge_maze.create_dataset.clustering_tools.cluster_questions import cluster_general_entities, cluster_date_values, cluster_numerical_values


MAX_CLUSTER_SIZE = 7
MIN_CLUSTER_SIZE = 2


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
    place_data = []  # Place related answers

    # Categorize the dataset
    for sample in dataset:
        # Date-related answers have a type stored as a list (e.g., ["DATE", "year"])
        try:
            if isinstance(sample['Type'], tuple):
                date_data.append(sample)
            # Numerical answers with types like "CARDINAL", "MONEY", "PERCENT" or "AGE"
            elif sample['Type'] == "NUMERICAL":
                num_data.append(sample)
            elif sample['Type'] in ["PLACE", "ORG"]:
                place_data.append(sample)
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
        "Place": place_data,
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


async def create_paths(processed_data: list[dict], output_path: str = None) -> list[dict]:
    """
    Processes input question clusters and generates reasoning chains.

    This function performs the following steps:
    1. Generates embeddings for all entity questions in the input clusters.
    2. Filters questions based on a similarity graph.
    3. Removes ambiguous questions from the clusters.
    4. Splits the remaining question clusters into smaller groups with a maximum size of 10.
    5. Optionally saves the resulting reasoning chains to a specified file path in JSON format.
    6. Plot reasoning chains length distribution

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

    reasoning_chains = []

    for name, data in divided_data.items():
        print(f"Creating reasoning paths for {name} clusters...")

        # Cluster data:
        if name in ["General", "Place"]:
            clusters = cluster_general_entities(data)
        elif name == "Number":
            clusters = cluster_numerical_values(data)
        elif name == "Date":
            clusters = cluster_date_values(data)
        else:
            raise ValueError

        # Generate embeddings for all entity questions:
        questions_embeddings = generate_question_embeddings(clusters, batch_size=2048)

        # Filter questions based on similarity graph:
        filtered_clusters = filter_question_groups(clusters, questions_embeddings)

        # Filter out ambiguous questions:
        not_ambiguous_clusters = await question_ambiguous_filter(filtered_clusters)
        if name == "Place":
            not_ambiguous_clusters = await answer_ambiguous_filter(not_ambiguous_clusters)

        # Split question clusters to a maximum size of 10:
        if name == "Number":
            reasoning_chains += split_num_clusters(not_ambiguous_clusters, questions_embeddings)
        else:
            reasoning_chains += split_clusters(not_ambiguous_clusters)

    # Save in folder:
    if output_path:
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(reasoning_chains, out_file)

    # Plot reasoning chains length distribution:
    plot_clusters_distribution(reasoning_chains)

    return reasoning_chains


def plot_clusters_distribution(answer_dict: list) -> None:
    """
    Plots a histogram showing the distribution of cluster sizes.

    Args:
        answer_dict (list of dict): A list of dictionaries, where each dictionary contains:
                                    - 'Questions' (list): A list of questions belonging to an answer.
    """
    # Extract the number of questions per answer
    cluster_sizes = [len(entry['Questions']) for entry in answer_dict]

    # Set up bins for histogram
    bins = np.arange(MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE + 2)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    counts, edges, patches = ax.hist(cluster_sizes, bins=bins, edgecolor="black", alpha=0.75)

    # Center x-ticks on the bars
    ax.set_xticks((edges[:-1] + edges[1:]) / 2)
    ax.set_xticklabels([str(int(tick)) for tick in (edges[:-1] + edges[1:]) / 2])  # Convert ticks to integers

    # Add labels and title
    ax.set_title("Cluster Size Distribution")
    ax.set_xlabel("Number of Questions per Answer")
    ax.set_ylabel("Frequency")

    # Add labels above bars
    for count, edge in zip(counts, edges[:-1]):
        if count > 0:
            ax.text(edge + (edges[1] - edges[0]) / 2, count, f"{int(count)}",
                    ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.show()


def display_random_sample(dataset: list[dict]) -> None:
    """
    Displays a randomly selected sample from the dataset, showing its associated questions
    before and after filtering based on sentence similarity.

    This function selects a random answer-question pair from the given dataset, generates
    a sentence similarity graph for the questions, applies a filtering algorithm using a
    vertex cover to reduce redundancy, and visualizes the process.

    Args:
        dataset (list[dict]): A dictionary where keys are answers (str) and values are lists of questions (list[str]).
    """
    divided_data = divide_by_type(dataset)
    general_data = divided_data['General']
    general_clusters = cluster_general_entities(general_data[:10000])
    rand_sample = random.choice(list(general_clusters.items()))
    embeddings = generate_question_embeddings({rand_sample[0]: rand_sample[1]})
    print('=' * 100)
    print(f'Answer: {rand_sample[0]}')
    print('=' * 100)
    print('Questions before filtering:')
    for i, sentence in enumerate(rand_sample[1]):
        print(f'{i + 1}. {sentence}')
    graph = generate_sentence_similarity_graph(rand_sample[1], embeddings, threshold=0.7)
    plot_sentence_graph(graph)
    print('=' * 100)
    filtered_questions = remaining_nodes_after_vertex_cover(graph)
    print('Questions after filtering:')
    for i, sentence in enumerate(filtered_questions):
        print(f'{i + 1}. {sentence}')