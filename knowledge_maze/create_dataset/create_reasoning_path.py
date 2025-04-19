"""
Processes QA datasets to generate structured reasoning paths.

This module orchestrates a pipeline for transforming raw question-answer data
into refined clusters suitable for generating reasoning chains. It integrates
data categorization, sentence embedding generation, semantic clustering,
ambiguity filtering using LLMs, graph-based question filtering, and cluster
size management.

Core Workflow (`create_paths` function):
1.  **Categorization:** Divides input data by answer type (Date, Number,
    Place, General) using `divide_by_type`.
2.  **Initial Clustering:** Applies type-specific clustering logic
    (`cluster_general_entities`, `cluster_numerical_values`,
    `cluster_date_values`) to group questions associated with similar answers
    or answer characteristics.
3.  **Embedding Generation:** Computes sentence embeddings for all questions
    within the initial clusters using `generate_question_embeddings` (leveraging
    `SentenceTransformer`).
4.  **Similarity Filtering:** Refines clusters by removing semantically redundant
    questions based on embedding similarity graphs and vertex cover algorithms
    via `filter_question_groups`.
5.  **Ambiguity Filtering:** Further refines clusters by removing questions or
    entire clusters identified as ambiguous by LLM classifiers using functions
    like `question_ambiguous_filter`, `answer_ambiguous_filter`, and
    `date_ambiguous_filter`.
6.  **Cluster Splitting:** Ensures final clusters adhere to size constraints
    (MIN/MAX_CLUSTER_SIZE) using `split_clusters_random`.
7.  **Output & Visualization:** Returns the final list of processed clusters
    (reasoning chains), optionally saves them, and plots the distribution of
    their sizes using `plot_clusters_distribution`.

Helper Utilities:
- `divide_by_type`: Categorizes data based on answer type.
- `generate_embeddings`: Core function for embedding sentences (GPU-aware).
- `generate_question_embeddings`: Manages batch embedding for efficiency.
- `plot_clusters_distribution`: Visualizes cluster size frequency.
- `display_random_sample`: Selects a random cluster, visualizes its similarity
  graph, and shows the effect of similarity filtering.

Configuration Constants:
- `MAX_CLUSTER_SIZE`, `MIN_CLUSTER_SIZE`: Define cluster size limits.
- `SIMILARITY_THRESHOLDS`: Dictionary mapping data types to thresholds used
  in graph-based filtering.
"""
import json
import random

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

from knowledge_maze.create_dataset.clustering_tools.split_clusters import split_clusters_random
from knowledge_maze.create_dataset.clustering_tools.cluster_questions import (cluster_general_entities,
                                                                              cluster_date_values,
                                                                              cluster_numerical_values)
from knowledge_maze.create_dataset.clustering_tools.ambiguity_filters import (question_ambiguous_filter,
                                                                              answer_ambiguous_filter,
                                                                              date_ambiguous_filter)
from knowledge_maze.create_dataset.clustering_tools.similarity_graphs import (filter_question_groups,
                                                                              generate_sentence_similarity_graph,
                                                                              plot_sentence_graph,
                                                                              remaining_nodes_after_vertex_cover)


MAX_CLUSTER_SIZE = 7
MIN_CLUSTER_SIZE = 2
SIMILARITY_THRESHOLDS = {
    "General": 0.5,
    "Place": 0.5,
    "Number": 0.75,
    "Date": 0.75
}


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


async def create_paths(processed_data: list[dict],
                       output_path: str = None) -> list[dict]:
    """
    Orchestrates the end-to-end process of generating reasoning chains.

    Takes preprocessed QA data, divides it by answer type, applies type-specific
    clustering, generates question embeddings, filters clusters based on semantic
    similarity and LLM-based ambiguity checks, splits oversized clusters, and
    returns the final list of refined question clusters (reasoning chains).

    Args:
        processed_data: A list of dictionaries, where each dictionary likely
            contains 'Question', 'Answer', and 'Type' keys, representing the
            initial preprocessed QA pairs.
        output_path: Optional file path to save the resulting list of reasoning
            chain dictionaries as a JSON file. Defaults to None.

    Returns:
        A list of dictionaries, where each dictionary represents a final,
        refined cluster (reasoning chain) containing:
            - 'Answer': The original answer or cluster key.
            - 'Questions': A list of semantically distinct and non-ambiguous
              questions related to the answer, adhering to size constraints.
            - 'Type': The original answer type.
            - Potentially other keys added during clustering (e.g., 'Objective',
              'Category' for numerical/date types).
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
            clusters = await cluster_numerical_values(data)
        elif name == "Date":
            clusters = await cluster_date_values(data)
        else:
            raise ValueError

        # Generate embeddings for all entity questions:
        questions_embeddings = generate_question_embeddings(clusters,
                                                            batch_size=2048)

        # Filter questions based on similarity graph:
        filtered_clusters = filter_question_groups(clusters,
                                                   questions_embeddings,
                                                   threshold=SIMILARITY_THRESHOLDS[name])

        # Filter out ambiguous questions:
        not_ambiguous_clusters = await question_ambiguous_filter(filtered_clusters)
        if name == "Place":
            not_ambiguous_clusters = await answer_ambiguous_filter(not_ambiguous_clusters)
        if name == "Date":
            not_ambiguous_clusters = await date_ambiguous_filter(not_ambiguous_clusters)

        # Split question clusters to a maximum size of 10:
        reasoning_chains += split_clusters_random(not_ambiguous_clusters)

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