# =========================================
# Feature Processing Module              
# =========================================

# Standard Library Imports
import sys

# Configuration
import config

# OmniVerse & USD Imports
from omni.usd import get_context
from pxr import UsdGeom

# Custom Utilities
import isaac_utils as iu

# External Libraries
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity


# =========================================
# Feature Class Definition              
# =========================================
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


# =========================================
# Feature Extraction              
# =========================================
def query_features():
    """
    Extracts features from the USD stage and returns a list of Feature objects.
    
    Returns:
        list: A list of Feature objects.
    """
    stage = get_context().get_stage()
    features_prim = stage.GetPrimAtPath(config.PRIM_PATH)

    if not features_prim.IsValid():
        print(f"Warning: Feature prim at {config.PRIM_PATH} is not valid.")
        sys.exit(1)

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

    return all_features


# =========================================
# Feature Partitioning               
# =========================================
def partition_features(all_features):
    """
    Partitions features into two maps based on x-position and identifies overlapping features.

    Args:
        all_features (list): List of Feature objects.

    Returns:
        tuple: (map_a_features, map_b_features, overlapping_features)
    """
    threshold = np.median([f.position[0] for f in all_features])
    margin = config.OVERLAP_MARGIN
    map_a_features, map_b_features, overlapping_features = [], [], set()

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

    return map_a_features, map_b_features, overlapping_features


# =========================================
# Feature Noise Addition              
# =========================================
def apply_gaussian_noise(features, seed):
    """
    Applies Gaussian noise to feature positions.
    
    Args:
        features (list): List of Feature objects.
        seed (int): Random seed for noise.

    Returns:
        list: List of Feature objects with noise applied.
    """
    np.random.seed(seed)
    return [
        Feature(
            f.feature_id, f.feature_type,
            f.position + np.array([np.random.normal(0, config.NOISE_STDDEV), 
                                   np.random.normal(0, config.NOISE_STDDEV), 
                                   0]),  # No noise in z
            f.size, f.source_map
        )
        for f in features
    ]


# =========================================
# Feature Matching              
# =========================================
def match_features(features_a, features_b, threshold=0.8, max_distance=100.0):
    """
    Matches features from two maps based on cosine similarity of their descriptors.

    Args:
        features_a (list): List of Feature objects from Map A.
        features_b (list): List of Feature objects from Map B.
        threshold (float): Similarity threshold to consider a match.

    Returns:
        list: List of matched feature pairs (feature_a, feature_b).

    Matching Criteria:
        - Cosine similarity threshold (default: 0.8)
        - Maximum distance threshold (default: 100.0 meters)
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


# =========================================
# Feature Transformation              
# =========================================
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


# =========================================
# Apply Transformation              
# =========================================
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


# =========================================
# Feature Merging              
# =========================================
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


# =========================================
# 6D Descriptor: [x, y, z, width, height, depth]
# x, y, z      → World frame coordinates
# width        → Bounding box extent along x-axis
# height       → Bounding box extent along y-axis
# depth        → Bounding box extent along z-axis
# =========================================