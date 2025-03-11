"""Group questions"""
import torch
import numpy as np
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def cluster_wiki_entities(dataset: list[dict]) -> dict:
    # Get answers:
    answers = [sample['Normalized answer'] for sample in dataset]
    questions = [sample['Question'] for sample in dataset]

    # Initialize output dictionary:
    clustered_dataset = {}
    for answer in list(set(answers)):
        clustered_dataset[answer] = []

    # Cluster questions by answer:
    for answer, question in tqdm(zip(answers, questions), desc='Clustering Entity Questions', total=len(answers)):
        clustered_dataset[answer].append(question)

    # Remove answers that only appear once in the dataset:
    filtered_clustered_dataset = {}
    for key, value in clustered_dataset.items():
        if len(value) >= 2:
            filtered_clustered_dataset[key] = value

    return filtered_clustered_dataset


def cluster_numerical_values(dataset: list[dict]) -> dict:
    pass


def cluster_date_values(dataset: list[dict]) -> dict:
    pass


def plot_answer_distributions(answer_dicts: dict[dict]) -> None:
    """
    Plots histograms for three dictionaries, where keys are answers, and values are lists of questions.

    Args:
        answer_dicts (list of dict): A list of three dictionaries, where each key is a string answer,
                                     and the value is a list of question strings.
    """
    fig, axes = plt.subplots(1, len(answer_dicts), figsize=(6*len(answer_dicts), 5), sharey=True)

    for i, (name, answer_dict) in enumerate(answer_dicts.items()):
        lengths = [len(questions) for questions in answer_dict.values()]
        bins = np.arange(2, 25)  # Bins from 0 to 50

        counts, edges, patches = axes[i].hist(lengths, bins=bins, edgecolor="black", alpha=0.75)

        axes[i].set_title(f"{name}")
        axes[i].set_xlabel("Number of Questions per Answer")
        axes[i].set_ylabel("Frequency")
        axes[i].set_xticks(bins[::5])  # Show every 5th tick for clarity

        # Add labels above bars
        for count, edge, patch in zip(counts, edges[:-1], patches):
            if count > 0:
                axes[i].text(edge + 0.5, count, f"{int(count)}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.show()


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

    for i in tqdm(range(0, len(all_questions), batch_size)):
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
    filtered_clusters = {}
    for answer, questions in tqdm(clustered_dataset.items(), total=len(clustered_dataset)):
        filtered_questions = remaining_nodes_after_vertex_cover(
            generate_sentence_similarity_graph(questions, question_embeddings, threshold=threshold)
        )
        filtered_clusters[answer] = filtered_questions

    return filtered_clusters


def plot_clusters_distribution(clustered_data: dict):
    pass
