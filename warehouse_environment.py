# Omni and USD imports
import omni
from omni.isaac.kit import SimulationApp

# Initialize the simulation application
simulation_app = SimulationApp({"headless": False})

import omni.isaac.core.utils.stage as stage_utils
from omni.usd import get_context
import sys, time, atexit, numpy as np, carb, cv2
from sklearn.metrics.pairwise import cosine_similarity
from pxr import UsdGeom

# Custom graph utility imports
import graph_utils as gu

# Custom isaac utility imports
import isaac_utils as iu

# Import NetworkX for graph visualization
import networkx as nx
import matplotlib.pyplot as plt


#########################################
# Feature Class Definition              #
#########################################
class Feature:
    """
    Feature representation for multi-map fusion.

    Args:
        feature_id (str): Unique identifier.
        feature_type (str): Semantic label (e.g., 'rack', 'crane', 'forklift').
        position (tuple or list): (x, y, z) coordinates.
        size (tuple or list): (width, height, depth).
        orientation (list): Orientation of the feature.
        scale (float): Feature scale.
        covariance (list): Feature covariance.
        confidence (float): Confidence level.
        source_map (str): Source map name.
        timestamp (float): Timestamp.
        descriptor (list): 6D descriptor for feature matching (x, y, z, width, height, depth).
    """
    def __init__(self, feature_id, feature_type, position, size=None, orientation=None,
                    scale=1.0, covariance=None, confidence=1.0, source_map=None, timestamp=None):
        
        self.feature_id = feature_id
        self.feature_type = feature_type
        self.position = np.array(position)
        self.size = np.array(size) if size is not None else np.array([1.0, 1.0, 1.0])
        self.orientation = orientation
        self.scale = scale
        self.covariance = covariance if covariance is not None else np.eye(3) * 0.01
        self.confidence = confidence
        self.source_map = source_map
        self.timestamp = timestamp

        # 6D descriptor: [x, y, z, width, height, depth]
        self.descriptor = np.concatenate([self.position, self.size])

    def to_dict(self):
        """Convert the feature to a dictionary."""
        return {
            "feature_id": self.feature_id,
            "feature_type": self.feature_type,
            "position": self.position.tolist(),
            "size": self.size.tolist(),
            "orientation": self.orientation,
            "scale": self.scale,
            "covariance": self.covariance.tolist(),
            "confidence": self.confidence,
            "source_map": self.source_map,
            "timestamp": self.timestamp,
            "descriptor": self.descriptor.tolist()
        }

    def __repr__(self):
        """Return a string representation of the feature."""
        return f"Feature(id={self.feature_id}, type={self.feature_type}, pos={self.position}, size={self.size})"


#########################################
# Feature Matching using Cosine Similarity
#########################################
def match_features(features_a, features_b, threshold=0.8, max_distance=100.0):
    """
    Matches features from two maps based on cosine similarity of their descriptors.

    Args:
        features_a (list): List of Feature objects from Map A.
        features_b (list): List of Feature objects from Map B.
        threshold (float): Similarity threshold to consider a match.

    Returns:
        list: List of matched feature pairs (feature_a, feature_b).
    """

    # Extract descriptors
    descriptors_a = np.array([f.descriptor for f in features_a if f.descriptor is not None])
    descriptors_b = np.array([f.descriptor for f in features_b if f.descriptor is not None])

    # Ensure descriptors are in 2D format
    if descriptors_a.ndim == 1:
        descriptors_a = descriptors_a.reshape(1, -1)
    if descriptors_b.ndim == 1:
        descriptors_b = descriptors_b.reshape(1, -1)

    # Validate descriptor dimensions
    if descriptors_a.shape[0] == 0 or descriptors_b.shape[0] == 0:
        print("Warning: No valid descriptors found for feature matching.")
        return []

    # Compute cosine similarity
    sim_matrix = cosine_similarity(descriptors_a, descriptors_b)

    # Find best matches
    matches = []
    used_ids = set()
    
    for i, row in enumerate(sim_matrix):
        j = np.argmax(row)
        similarity = row[j]
        
        if similarity > threshold and features_b[j].feature_id not in used_ids:
            dist = np.linalg.norm(features_a[i].position - features_b[j].position)

            if dist < max_distance:  # Reject matches where distance is too large
                matches.append((features_a[i], features_b[j]))
                used_ids.add(features_b[j].feature_id)  # Mark feature in Map B as used
                print(f"Match: {features_a[i].feature_id} → {features_b[j].feature_id}, Similarity: {similarity:.4f}, Distance: {dist:.2f}")
            else:
                print(f"Rejected: {features_a[i].feature_id} → {features_b[j].feature_id}, Similarity: {similarity:.4f}, Distance: {dist:.2f} (Too large)")

    return matches


#########################################
# Map Alignment using RANSAC
#########################################
def estimate_transform_ransac(matches):
    """
    Estimates a 2D affine transformation matrix using RANSAC.

    Args:
        matches (list): List of matched feature pairs.

    Returns:
        np.array: 2D affine transformation matrix.
        np.array: Inliers from the RANSAC estimation.
    """

    if len(matches) < 3:
        print("Warning: Not enough matches for RANSAC. Using identity transformation.")
        return np.array([[1, 0, 0], [0, 1, 0]]), None  # Identity matrix

    # Extract points
    src_pts = np.array([[m[0].position[0], m[0].position[1]] for m in matches]).astype(np.float32)
    dst_pts = np.array([[m[1].position[0], m[1].position[1]] for m in matches]).astype(np.float32)

    # Estimate transformation using RANSAC
    transformation_matrix, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=0.9, maxIters=1000)

    if transformation_matrix is None:
        print("Warning: RANSAC failed. Using identity transformation.")
        return np.array([[1, 0, 0], [0, 1, 0]]), None

    print(f"RANSAC succeeded. Filtered transformation matrix:\n{transformation_matrix}")
    return transformation_matrix, inliers


#########################################
# Apply Transformation to Features
#########################################
def transform_features(features, transformation_matrix):
    """
    Applies a 2D affine transformation to feature positions.
    
    Args:
        features (list): List of Feature objects.
        transformation_matrix (np.array): 2D affine transformation matrix.
        
    Returns:
        list: List of transformed Feature objects.
    """

    if transformation_matrix is None:
        print("Warning: No valid transformation matrix. Skipping transformation.")
        return features  # Return original features if no valid transformation

    transformed_features = []
    for feature in features:
        pos_2d = np.array([feature.position[0], feature.position[1], 1])
        new_pos_2d = transformation_matrix @ pos_2d  # Apply transformation
        new_pos = np.array([new_pos_2d[0], new_pos_2d[1], feature.position[2]])  # Preserve Z

        transformed_features.append(Feature(
            feature_id=feature.feature_id,
            feature_type=feature.feature_type,
            position=new_pos,
            size=feature.size
        ))

    return transformed_features


#########################################
# Merge Overlapping Features
#########################################
def merge_features(features_a, features_b, matches):
    """
    Merges matched features by averaging positions and ensures all features are included.
    
    Args:
        features_a (list): List of Feature objects from Map A.
        features_b (list): List of Feature objects from Map B.
        matches (list): List of matched feature pairs.

    Returns:
        list: List of merged Feature objects.
    """

    merged_features = {f.feature_id: f for f in features_a}

    for feature_a, feature_b in matches:
        avg_position = (feature_a.position + feature_b.position) / 2
        merged_features[feature_a.feature_id] = Feature(
            feature_id=feature_a.feature_id,
            feature_type=feature_a.feature_type,
            position=avg_position,
            size=feature_a.size
        )

    # Add features from B that were NOT matched
    for feature_b in features_b:
        if feature_b.feature_id not in [m[1].feature_id for m in matches]:  # Ensure unique entries
            merged_features[feature_b.feature_id] = feature_b

    return list(merged_features.values())


#########################################
# Simulation Handler Class              #
#########################################
class SimulationHandler:
    """
    Handles the simulation application including stage loading and updating.
    cleanup, and signal handling.
    """
    def __init__(self, kit=None):
        """
        Initialize the simulation handler.

        Args:
            kit: The simulation application kit.
        """
        self.kit = kit
        self.load_stage()
       
    def load_stage(self):
        """
        Load the simulation stage from a USD file.
        """
        usd_path = ("omniverse://cerlabnucleus.lan.local.cmu.edu/Users/weihuanw/sim_environments/large_modular_warehouse_v1.usd")
        prim_path = "/World/Warehouse"

        # Validate USD path
        if not usd_path.startswith("omniverse://"):
            carb.log_error(f"Invalid USD path: {usd_path}")
            self.kit.close()
            sys.exit(1)

        # Attempt to load the stage reference
        try:
            stage_utils.add_reference_to_stage(usd_path, prim_path)
            print("Stage loaded successfully.")

        except Exception as e:
            # Handle the error if stage cannot be loaded
            carb.log_error(f"Failed to load stage: {e}")
            self.kit.close()
            sys.exit(1)

    def update_sim(self):
        """
        Update the simulation application.
        """
        self.kit.update()

    def close_sim(self):
        """
        Close the simulation application.
        """
        print("Closing simulation...")
        self.kit.close()


#########################################
# Main Function                         #
#########################################
def main():
    """Main function for feature extraction, matching, transformation, and visualization."""

    # Initialize simulation
    simulation_handler = SimulationHandler(simulation_app)
    atexit.register(simulation_handler.close_sim)

    # Get the current USD stage
    stage = get_context().get_stage()
    feature_prim_path = "/World/Warehouse/Features"
    features_prim = stage.GetPrimAtPath(feature_prim_path)

    if not features_prim.IsValid():
        print(f"Warning: Feature prim at {feature_prim_path} is not valid.")
        sys.exit(1)

    # ---------------------------------------------------
    # 1. Query Features from USD Stage
    # ---------------------------------------------------
    all_features = []
    print("Querying features from the USD stage...")

    for container in features_prim.GetChildren():
        container_name = container.GetName()
        print(f"\nProcessing: {container_name} ({container.GetPath()})")

        for subchild in container.GetChildren():
            if not subchild.IsValid() or not subchild.IsA(UsdGeom.Xformable):
                continue

            prim_name = subchild.GetName()
            translation, _, _ = iu.get_world_transform(subchild)
            bbox_min, bbox_max = iu.get_bounding_box(subchild)
            size = bbox_max - bbox_min  # Compute size from bounding box

            feature_type = "unknown"
            if "crane" in prim_name.lower():
                feature_type = "crane"
            elif "forklift" in prim_name.lower():
                feature_type = "forklift"
            elif "rack" in prim_name.lower():
                feature_type = "rack"

            feature = Feature(prim_name, feature_type, translation, size, "Warehouse")
            all_features.append(feature)
            print(f"  Found {feature_type.capitalize()}: {prim_name} at {translation}")

    if not all_features:
        print("Warning: No features found via prim query.")
        sys.exit(1)

    # ---------------------------------------------------
    # 2. Partition Features into Two Maps
    # ---------------------------------------------------
    threshold = np.median([f.position[0] for f in all_features])
    margin = 1000  
    map_a_features = []
    map_b_features = []
    overlapping_features = set()

    for feature in all_features:
        if feature.position[0] < threshold - margin:
            map_a_features.append(feature)
        elif feature.position[0] > threshold + margin:
            map_b_features.append(feature)
        else:
            map_a_features.append(feature)
            map_b_features.append(feature)
            overlapping_features.add(feature.feature_id)

    print("\nScene partitioning complete.")
    print(f"  Threshold (x): {threshold}")
    print(f"  Overlap margin: {margin}")
    print(f"  Map_A features: {len(map_a_features)}")
    print(f"  Map_B features: {len(map_b_features)}")
    print(f"  Overlapping features: {len(overlapping_features)}")

    # Print overlapping feature IDs before matching
    print("\nOverlapping Feature IDs:")
    for feature_id in sorted(overlapping_features):
        print(f"  - {feature_id}")

    # Debug Print: Original positions before noise
    # print("Original Map A Features:", [f.position for f in map_a_features])
    # print("Original Map B Features:", [f.position for f in map_b_features])

    # ---------------------------------------------------
    # 3. Save Ground Truth Feature Graphs
    # ---------------------------------------------------
    graphs_gt = {
        "gt_feature_graph": all_features,
        "gt_feature_graph_A": map_a_features,
        "gt_feature_graph_B": map_b_features
    }

    for name, data in graphs_gt.items():
        graph = gu.build_graph_from_feature_list(data)
        gu.save_feature_graph_to_json(graph, filename=f"{name}.json")
        gu.draw_feature_graph(graph)
        plt.savefig(f"{name}.png")

    print("\nSaved ground truth feature graphs.")

    # ---------------------------------------------------
    # 4. Apply Gaussian Noise to Both Maps
    # ---------------------------------------------------
    noise_stddev = 2.0  # Standard deviation for noise in meters

    np.random.seed(42)  # Ensure reproducibility
    noised_map_a_features = [
        Feature(f.feature_id, f.feature_type, 
                f.position + np.array([np.random.normal(0, noise_stddev), 
                                    np.random.normal(0, noise_stddev), 
                                    0]),  # No noise in z
                f.size, f.source_map)
        for f in map_a_features
    ]

    np.random.seed(24)  # Use a different seed for Map B
    noised_map_b_features = [
        Feature(f.feature_id, f.feature_type, 
                f.position + np.array([np.random.normal(0, noise_stddev), 
                                    np.random.normal(0, noise_stddev), 
                                    0]),  # No noise in z
                f.size, f.source_map)
        for f in map_b_features
    ]

    print(f"\nGaussian noise applied with stddev={noise_stddev} meters.")

    # Debug Print: Noised feature positions
    # print("Noised Map A Features:", [f.position for f in noised_map_a_features])
    # print("Noised Map B Features:", [f.position for f in noised_map_b_features])

    # ---------------------------------------------------
    # 5. Save Noised Feature Graphs
    # ---------------------------------------------------
    graphs_noised = {
        "noised_feature_graph_A": noised_map_a_features,
        "noised_feature_graph_B": noised_map_b_features
    }

    for name, data in graphs_noised.items():
        graph = gu.build_graph_from_feature_list(data)
        gu.save_feature_graph_to_json(graph, filename=f"{name}.json")
        gu.draw_feature_graph(graph)
        plt.savefig(f"{name}.png")

    print("\nSaved noised feature graphs.")

    # ---------------------------------------------------
    # 6. Feature Matching using Cosine Similarity
    # ---------------------------------------------------
    matches = match_features(noised_map_a_features, noised_map_b_features)
    print(f"\nFeature matching complete. Found {len(matches)} matched features.")

    # Debug Print: Feature matches
    for a, b in matches:
        print(f"Match: {a.feature_id} -> {b.feature_id}, Positions: {a.position} -> {b.position}")

    # ---------------------------------------------------
    # 7. Estimate Transformation using RANSAC
    # ---------------------------------------------------
    transformation_matrix, inliers = estimate_transform_ransac(matches)
    print(f"\nTransformation matrix estimated:\n{transformation_matrix}")

    # ---------------------------------------------------
    # 8. Apply Transformation to Noised Map B Features
    # ---------------------------------------------------
    transformed_noised_map_b_features = transform_features(noised_map_b_features, transformation_matrix)

    # Debugging: Print transformed feature positions
    # print("\nSample transformed feature positions:")
    # for feature in transformed_noised_map_b_features[:5]:  # Print first 5 transformed features
    #     print(f"  Transformed Feature: {feature.feature_id}, Position: {feature.position}")

    # ---------------------------------------------------
    # 9. Merge Noised Maps and Save Fused Graph
    # ---------------------------------------------------
    fused_noised_features = merge_features(noised_map_a_features, transformed_noised_map_b_features, matches)

    fused_noised_graph = gu.build_graph_from_feature_list(fused_noised_features)
    gu.save_feature_graph_to_json(fused_noised_graph, filename="fused_noised_feature_graph.json")
    gu.draw_feature_graph(fused_noised_graph)
    plt.savefig("fused_noised_feature_graph.png")

    print("\nSaved merged noised feature graph.")

    # ---------------------------------------------------
    # 10. Main Simulation Loop
    # ---------------------------------------------------
    try:
        while simulation_app.is_running():
            simulation_handler.update_sim()
            time.sleep(0.01)
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        simulation_handler.close_sim()
        sys.exit(0)


if __name__ == "__main__":
    main()

# Run Program in vs terminal: 
# Windows: python isaac_python.py
# Linux: python3 isaac_python.py

# Open Isaac Standalone Isaac Sim in Windows Command Prompt
# "C:\Users\RyanWu\AppData\Local\ov\pkg\isaac-sim\isaac-sim.selector.bat"