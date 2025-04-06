"""
Module for sentence similarity analysis and graph-based clustering.

This module provides functions for constructing and visualizing sentence similarity graphs,
filtering redundant nodes based on semantic similarity, and refining clusters of questions
using graph algorithms. It enables advanced analysis and visualization of relationships
between sentences based on embeddings.

Functions:
    - generate_sentence_similarity_graph: Constructs a weighted graph where nodes represent sentences,
      and edges represent cosine similarity between sentence embeddings.
    - plot_sentence_graph: Visualizes a sentence similarity graph, displaying node connections
      and edge weights.
    - remaining_nodes_after_vertex_cover: Applies a vertex cover algorithm to filter redundant
      nodes and returns the remaining nodes in the graph.
    - filter_question_groups: Processes clusters of questions by creating similarity graphs,
      applying vertex cover, and refining the clusters to remove redundant questions.
    - display_random_sample: Randomly selects a sample answer-question pair, visualizes the sentence
      similarity graph, and filters redundant questions using graph-based methods.
"""
import random
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


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


def filter_question_groups(clustered_dataset: dict, question_embeddings: dict, threshold: float = 0.5) -> dict:
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