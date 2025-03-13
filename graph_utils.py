# =========================================
# Graphing Utility Module              
# =========================================

# Standard Library Imports
import json
import os

# External Libraries
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# =========================================
# Graph Construction                
# =========================================
def build_graph_from_feature_list(features_list, distance_threshold=2500):
    """
    Build a NetworkX graph from a list of Feature objects.
    Two nodes are connected by an edge if their positions are within the distance threshold.
    """
    feature_graph = nx.Graph()
    for feature in features_list:
        feature_graph.add_node(feature.feature_id, feature=feature)
    
    n = len(features_list)
    for i in range(n):
        for j in range(i + 1, n):
            pos1 = features_list[i].position
            pos2 = features_list[j].position
            if np.linalg.norm(pos1 - pos2) < distance_threshold:
                feature_graph.add_edge(features_list[i].feature_id, features_list[j].feature_id)
    return feature_graph


# =========================================
# Graph Visualization (3D)             
# =========================================
def draw_feature_graph(feature_graph):
    """
    Draws the feature graph in 3D using Matplotlib.
    
    Nodes are colored based on feature type:
      - rack: blue
      - crane: red
      - forklift: green
    Edges are drawn as gray lines connecting the nodes.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    color_map = {"rack": "blue", "crane": "red", "forklift": "green"}
    
    for node, data in feature_graph.nodes(data=True):
        feature = data['feature']
        pos = feature.position
        node_color = color_map.get(feature.feature_type, "black")
        ax.scatter(pos[0], pos[1], pos[2], color=node_color, s=50)
        ax.text(pos[0], pos[1], pos[2], f"{node}", size=10, zorder=1, color='k')
    
    for edge in feature_graph.edges():
        pos1 = feature_graph.nodes[edge[0]]['feature'].position
        pos2 = feature_graph.nodes[edge[1]]['feature'].position
        xs = [pos1[0], pos2[0]]
        ys = [pos1[1], pos2[1]]
        zs = [pos1[2], pos2[2]]
        ax.plot(xs, ys, zs, color="gray", alpha=0.7)
    
    ax.set_xlabel("X (Meters)")
    ax.set_ylabel("Y (Meters)")
    ax.set_zlabel("Z (Meters)")
    plt.title("Factor Graph Structure for Warehouse Environment")
    # plt.show()


# =========================================
# Graph Logging (JSON)
# =========================================
def save_feature_graph_to_json(feature_graph, filename="feature_graph.json"):
    """
    Save the feature graph to a json file.

    The JSON structure includes:
      - "features": a list of feature dictionaries (id, type, position, scale, etc.)
      - "edges": a list of tuples representing connectivity between features
    """
    data = {"features": [], "edges": [] }

    # Iterate over nodes and convert Feature objects to dictionaries.
    for node, node_data in feature_graph.nodes(data=True):
        feature = node_data.get("feature")
        if feature:
            data["features"].append(feature.to_dict())
    
    # Convert the edges to a simple list of tuples.
    data["edges"] = list(feature_graph.edges())
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Feature graph saved to {filename}")


# =========================================
# Noise Injection (Gaussian Noise)        
# =========================================
def apply_gaussian_noise(feature_graph, noise_stddev=0.01):
    """
    Apply Gaussian noise to the position of each feature in the graph.

    Args:
        feature_graph (networkx.Graph): The feature graph to modify.
        noise_stddev (float): Standard deviation of the Gaussian noise to apply.
    """
    
    for node, node_data in feature_graph.nodes(data=True):
        feature = node_data.get("feature")
        if feature:
            feature.position += np.random.normal(0, noise_stddev, size=3)
            # Update the feature in the graph
            node_data["feature"] = feature
    print(f"Applied Gaussian noise with stddev={noise_stddev} to feature positions.")


# =========================================
# Graph Saving            
# =========================================
def save_graph(graph, folder, filename_prefix):
    """
    Saves a given graph as both a JSON file and a PNG visualization.
    
    Args:
        graph (networkx.Graph): The feature graph to save.
        folder (str): Folder name where the files should be saved (e.g., "gt" or "noise").
        filename_prefix (str): Prefix for naming the files (e.g., "gt_feature_graph").
    """
    # Ensure the folder exists
    os.makedirs(f"results/{folder}", exist_ok=True)

    # Save JSON
    json_filename = f"results/{folder}/{filename_prefix}.json"
    save_feature_graph_to_json(graph, filename=json_filename)

    # Save PNG visualization
    png_filename = f"results/{folder}/{filename_prefix}.png"
    draw_feature_graph(graph)
    plt.savefig(png_filename)

    print(f"Saved {filename_prefix} in results/{folder}/")