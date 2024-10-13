import json
import networkx as nx
import pickle
import csv
import math
import random
from collections import Counter
import pandas as pd
from itertools import combinations
import copy
from datetime import datetime
import re
from itertools import product
from collections import defaultdict 
from tqdm import tqdm
import numpy as np
from collections import deque
import re
import os
from sklearn.cluster import HDBSCAN
from sklearn.datasets import load_digits
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from community import community_louvain  # For Louvain clustering

# Tags used for different edge types in the graph
TAGS = {        
        "N": "noise",  # Noise tag
        "I-1": "identity-betweeness",  # Identity edge based on betweenness centrality
        "I-2": "identity-use_clustering",  # Identity edge based on embeddings clustering
        "H-1": "hierarchy-betweeness",  # Hierarchical edge based on betweenness
        "H-2": "hierarchy-repair_directions",  # Hierarchical edge using direction repair
       }

def get_tags_status(G):
    """
    Returns the count of each tag present in the graph edges.

    Parameters:
    - G (networkx.Graph): The graph whose edge tags are to be counted.

    Returns:
    - dict: A dictionary with the tag as the key and the count as the value.
    """
    status = Counter([G.get_edge_data(u, v)["tag"] for u, v in G.edges])
    return {k: status[k] for k in sorted(status)}


def get_dag(df, phrase2id_updated):
    """
    Builds a directed acyclic graph (DAG) from a DataFrame and a phrase ID mapping.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing graph edge data (node1, node2, weight, tag).
    - phrase2id_updated (dict): A mapping from phrases to their corresponding IDs.

    Returns:
    - dag (defaultdict): A DAG where keys are node IDs and values are sets of child node IDs.
    - l (list): A list of tuples where cycles were detected in the DAG (node1, node2, id_node1, id_node2).
    """
    dag = defaultdict(set)
    l = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        node1, node2, weight, tag = row['node1'], row['node2'], row["weight"], row["tag"]
        id_node1 = phrase2id_updated[node1]
        id_node2 = phrase2id_updated[node2]
        if tag.startswith("h") and id_node1 != id_node2:
            dag[id_node1].add(id_node2)
            if id_node1 in dag[id_node2]:
                l.append((node1, node2, id_node1, id_node2))
    return dag, l

def invert_dag(dag):
    """
    Inverts the given DAG to get a parent-to-child mapping.

    Parameters:
    - dag (dict): Original DAG with keys as parents and values as children.

    Returns:
    - inverted_dag (dict): Inverted DAG where keys are children and values are lists of parents.
    """
    inverted_dag = {}
    for parent, children in dag.items():
        for child in children:
            if child not in inverted_dag:
                inverted_dag[child] = []
            inverted_dag[child].append(parent)
    return inverted_dag

def get_df_of_graph(G_final):
    """
    Converts a graph's edge information into a DataFrame.

    Parameters:
    - G_final (networkx.Graph): The input graph with edge information.

    Returns:
    - pandas.DataFrame: A DataFrame containing node1, node2, weight, and tag of edges.
    """
    node1, node2, weight, tag = [], [], [], []

    for e in tqdm(G_final.edges):
        u, v = e
        data = G_final.get_edge_data(u, v)
        if data["tag"].startswith("i") or data["tag"].startswith("n"):
            node1.append(u)
            node2.append(v)
        elif data["tag"].startswith("h") and "dir" in data:
            n1, n2 = data["dir"]
            node1.append(n1)
            node2.append(n2)
        else:
            continue
        weight.append(data["weight"])
        tag.append(data["tag"])  

    data = {
        "node1": node1,
        "node2": node2,
        "weight": weight,
        "tag": tag
    }

    df = pd.DataFrame(data)
    return df

def get_directed_graph(G):
    """
    Converts an undirected graph to a directed graph based on hierarchical edges.

    Parameters:
    - G (networkx.Graph): Input undirected graph.

    Returns:
    - G_dir (networkx.DiGraph): Directed graph where edges represent parent-child relationships.
    """
    G_dir = nx.DiGraph()
    directed_edges = []
    for u, v in G.edges:
        data = G.get_edge_data(u, v)
        if "tag" in data and data["tag"].startswith("h"):
            parent, child = data["dir"]
            directed_edges.append((parent, child, {"weight": data["weight"], "tag": data["tag"]}))

    G_dir.add_edges_from(directed_edges)
    return G_dir

def group_strings_by_class(strings, classes):
    """
    Groups strings by their corresponding class labels.

    Parameters:
    - strings (list of str): List of strings to be grouped.
    - classes (list of int/str): List of class labels corresponding to the strings.

    Returns:
    - dict: A dictionary where keys are class labels and values are lists of strings.
    """
    grouped_strings = defaultdict(list)
    for string, cls in zip(strings, classes):
        grouped_strings[cls].append(string)
    return dict(grouped_strings)

def get_embeds(texts):
    """
    Generates embeddings for a list of texts using the SentenceTransformer model.

    Parameters:
    - texts (list of str): List of input texts to be encoded.

    Returns:
    - numpy.ndarray: Matrix of embeddings for the input texts.
    """
    phrase2embed = defaultdict()
    model = SentenceTransformer('tavakolih/all-MiniLM-L6-v2-pubmed-full')
    batch_size = 64

    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        embeddings = model.encode(batch)  # Encode the current batch
        all_embeddings.append(embeddings)
    
    all_embeddings = np.vstack(all_embeddings)
    return all_embeddings

def update_concepts(G):
    """
    Updates concepts by removing non-identity edges and clustering nodes based on connected components.

    Parameters:
    - G (networkx.Graph): Input graph with edge data.

    Returns:
    - dict_com_updated (defaultdict): Mapping from component ID to nodes.
    - phrase2id_updated (defaultdict): Mapping from phrases to their new component IDs.
    """
    G_copy = copy.deepcopy(G)
    edges_not_iden = []
    for u, v in G_copy.edges:
        if not G_copy.get_edge_data(u, v)["tag"].startswith("i"):
            edges_not_iden.append((u, v))
            
    G_copy.remove_edges_from(edges_not_iden)
    cc = list(nx.connected_components(G_copy))
    
    dict_com_updated = defaultdict(set)
    phrase2id_updated = defaultdict(list)

    for com in cc:
        idx = len(dict_com_updated)
        dict_com_updated[idx] = com
        for ph in com:
            phrase2id_updated[ph] = idx
    
    return dict_com_updated, phrase2id_updated

def calculate_pmi(count_x, count_y, count_xy, total_count, epsilon=1e-15):
    """
    Calculates the Pointwise Mutual Information (PMI) between two events.

    Parameters:
    - count_x (int): Count of event x.
    - count_y (int): Count of event y.
    - count_xy (int): Count of both x and y occurring together.
    - total_count (int): Total number of events.
    - epsilon (float): A small value to avoid division by zero.

    Returns:
    - float: The PMI value between event x and event y.
    """
    P_x = max(count_x / total_count, epsilon)
    P_y = max(count_y / total_count, epsilon)
    P_xy = max(count_xy / total_count, epsilon)
    
    pmi = np.log(P_xy / (P_x * P_y))
    return pmi
