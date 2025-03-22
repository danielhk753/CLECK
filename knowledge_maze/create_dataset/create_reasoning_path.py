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
    - split_clusters: Splits clusters into smaller uniform groups while handling small clusters.
    - display_random_sample: Randomly samples an answer-question pair and visualizes the filtering process.
    - plot_clusters_distribution: Plots histograms of question distributions across clusters.
"""

import torch
import random
import asyncio
import numpy as np
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from knowledge_maze.tools.api_tools import gptAPI


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


def generate_sentence_similarity_graph(sentences: list[str], all_embeddings: dict, threshold: float = 0.5) -> nx.Graph:
    """
    Generates a weighted graph where each node represents a sentence,
    and edges represent cosine similarity between sentence embeddings.

    Args:
        sentences (list[str]): List of sentences.
        all_embeddings (dict): Question's embeddings.
        threshold (float): Minimum similarity value for a vertex in the graph.
    Returns:
        nx.Graph: A weighted graph with sentences as nodes and similarity scores as edges.
    """
    # Step 1: Generate sentence embeddings
    embeddings = [all_embeddings[sentence].cpu() for sentence in sentences]

    # Step 2: Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)  # (N x N) matrix

    # Step 3: Create the graph
    graph = nx.Graph()

    # Add nodes (sentences)
    for i, sentence in enumerate(sentences):
        graph.add_node(i, text=sentence)  # Nodes are indexed by sentence ID

    # Add edges (cosine similarity as weights)
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):  # Avoid self-loops and duplicate edges
            similarity = similarity_matrix[i, j]
            if similarity > threshold:  # Only add meaningful edges
                graph.add_edge(i, j, weight=similarity)

    return graph


def plot_sentence_graph(graph: nx.Graph):
    """
    Plots the sentence similarity graph with edge weights.

    Args:
        graph (nx.Graph): A NetworkX graph where nodes are sentences and edges are cosine similarities.
    """
    plt.figure(figsize=(10, 5))

    pos = nx.spring_layout(graph, seed=42)  # Positioning using force-directed layout

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=400, node_color='lightblue', edgecolors='black')

    # Draw edges with transparency
    edges = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edges(graph, pos, width=[v * 3 for v in edges.values()], alpha=0.6)

    # Draw edge labels (cosine similarity scores)
    edge_labels = {k: f"{v:.2f}" for k, v in edges.items()}  # Format similarity to 2 decimal places
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=9)

    # Draw node labels (sentences)
    node_labels = {node: i+1 for i, node in enumerate(graph.nodes)}
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=9, font_weight="bold")

    plt.title("Sentence Similarity Graph", fontsize=14)
    plt.axis("off")  # Hide axis
    plt.show()


def remaining_nodes_after_vertex_cover(graph: nx.Graph) -> list[str]:
    """
    Removes the smallest number of nodes to eliminate all edges
    and returns the remaining nodes.

    Args:
        graph (nx.Graph): The input graph where nodes are sentences.

    Returns:
        list[str]: The list of nodes that remain after applying vertex cover.
    """
    graph_copy = graph.copy()  # Preserve original graph
    cover_set = set()

    while graph_copy.edges:  # While there are edges left
        # Pick the node with the highest degree (most connections)
        node_to_remove = max(graph_copy.degree, key=lambda x: x[1])[0]
        cover_set.add(node_to_remove)

        # Remove the node and all its edges
        graph_copy.remove_node(node_to_remove)

    # Nodes that remain after removing the vertex cover
    remaining_nodes = list(set(graph.nodes) - cover_set)

    # Retrieve original sentences using node attributes
    remaining_sentences = [graph.nodes[n]['text'] for n in remaining_nodes]

    return remaining_sentences


def filter_question_groups(clustered_dataset: dict, question_embeddings: dict, threshold: float = 0.7) -> dict:
    """
    Filters redundant questions in clusters based on semantic similarity.

    This function processes a dictionary of clusters, where each cluster contains
    an answer as the key and a list of associated questions as the value. It creates
    a sentence similarity graph for the questions in each cluster, applies a vertex
    cover algorithm to remove redundant questions, and returns the filtered clusters.

    Args:
        clustered_dataset (dict): A dictionary where keys are answers (str) and
            values are lists of associated questions (list[str]).
        question_embeddings (dict): A dictionary of question embeddings, where
            keys are questions (str) and values are their corresponding embeddings.
        threshold (float): The similarity threshold for creating edges in the graph.
            Only similarities greater than or equal to this value are considered.

    Returns:
        dict: A dictionary where keys are answers (str) and values are filtered lists
        of questions (list[str]) that remain after redundant questions are removed.

    """
    filtered_clusters = {}
    for answer, questions in tqdm(clustered_dataset.items(), total=len(clustered_dataset),
                                  desc='Creating similarity graphs and filtering questions'):
        filtered_questions = remaining_nodes_after_vertex_cover(
            generate_sentence_similarity_graph(questions, question_embeddings, threshold=threshold)
        )
        filtered_clusters[answer] = filtered_questions

    return filtered_clusters


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
                                  max_rpm=5000)
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


def display_random_sample(dataset: dict, embeddings: dict) -> None:
    """
    Displays a randomly selected sample from the dataset, showing its associated questions
    before and after filtering based on sentence similarity.

    This function selects a random answer-question pair from the given dataset, generates
    a sentence similarity graph for the questions, applies a filtering algorithm using a
    vertex cover to reduce redundancy, and visualizes the process.

    Args:
        dataset (dict): A dictionary where keys are answers (str) and values are lists of questions (list[str]).
        embeddings (dict): A dictionary of sentence embeddings where keys are sentences (str) and values
            are their embedding vectors (e.g., numpy arrays).
    """
    rand_sample = random.choice(list(dataset.items()))
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
