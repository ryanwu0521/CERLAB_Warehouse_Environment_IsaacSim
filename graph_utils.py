import json

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