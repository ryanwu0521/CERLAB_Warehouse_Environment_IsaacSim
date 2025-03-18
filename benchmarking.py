# =========================================
# RMSE Computation Utility Module              
# =========================================

# Custom Modules
from feature_processing import transform_features

# Standard Library Imports
import json

# External Libraries
import numpy as np


# =========================================
# Compute RMSE for Feature Fusion             
# =========================================
def compute_rmse(gt_filename, fused_filename):
    """
    Compute RMSE across all matched features in the final fused map.

    Args:
        gt_filename (str): Path to ground truth JSON file.
        fused_filename (str): Path to fused JSON file.

    Returns:
        dict: RMSE values for x, y, z, radius, and overall RMSE.
    """
    
    # Load Ground Truth and Fused Feature Data
    with open(gt_filename, "r") as f:
        gt_data = json.load(f)

    with open(fused_filename, "r") as f:
        fused_data = json.load(f)

    gt_features = {feature["feature_id"]: feature for feature in gt_data["features"]}
    fused_features = {feature["feature_id"]: feature for feature in fused_data["features"]}

    # Ensure we are comparing the same features
    common_feature_ids = set(gt_features.keys()) & set(fused_features.keys())

    if len(common_feature_ids) == 0:
        print("No matching feature IDs found. RMSE computation aborted.")
        return None

    errors_x, errors_y, errors_z, errors_radius = [], [], [], []

    for feature_id in common_feature_ids:
        gt_feature = gt_features[feature_id]
        fused_feature = fused_features[feature_id]

        # Compute position errors
        errors_x.append((gt_feature["position"][0] - fused_feature["position"][0]) ** 2)
        errors_y.append((gt_feature["position"][1] - fused_feature["position"][1]) ** 2)
        errors_z.append((gt_feature["position"][2] - fused_feature["position"][2]) ** 2)

        # Compute radius error
        errors_radius.append((gt_feature["radius"] - fused_feature["radius"]) ** 2)

    # Compute RMSE for each axis
    rmse_x = np.sqrt(np.mean(errors_x))
    rmse_y = np.sqrt(np.mean(errors_y))
    rmse_z = np.sqrt(np.mean(errors_z))
    rmse_radius = np.sqrt(np.mean(errors_radius))

    # Compute overall RMSE (3D position + radius)
    overall_rmse = np.sqrt(rmse_x**2 + rmse_y**2 + rmse_z**2 + rmse_radius**2)

    # Return RMSE results
    rmse_results = {
        "RMSE_X": rmse_x,
        "RMSE_Y": rmse_y,
        "RMSE_Z": rmse_z,
        "RMSE_Radius": rmse_radius,
        "Overall_RMSE": overall_rmse
    }

    return rmse_results


# =========================================
# Validate Transformation Matrix
# =========================================
def validate_transformation(ground_truth_features, transformed_features, matches, transformation_matrix):
    """
    Validates the transformation by applying it to all matched features.

    Args:
        ground_truth_features (list of Feature objects): Ground truth feature data.
        transformed_features (list of Feature objects): Transformed feature data after applying the transformation.
        matches (list of tuples): List of matched feature pairs [(gt_feature, transformed_feature)].
        transformation_matrix (numpy.ndarray): 2x3 estimated transformation matrix.

    Prints the original vs transformed feature locations for comparison.
    """
    print("\n=== Transformation Validation ===")
    
    # Apply transformation once to all features
    transformed_gt_features = transform_features(ground_truth_features, transformation_matrix)

    for gt_feature, transformed_feature in matches:
        # Extract (x, y) from ground truth
        gt_x, gt_y = gt_feature.position[:2]
        
        # Find the transformed version of the ground truth feature
        matched_transformed = next(
            (f for f in transformed_gt_features if f.feature_id == gt_feature.feature_id), None
        )

        if matched_transformed:
            transformed_x, transformed_y = matched_transformed.position[:2]
            print(f"Feature: {gt_feature.feature_id}")
            print(f"  Ground Truth:   ({gt_x:.2f}, {gt_y:.2f})")
            print(f"  Transformed:    ({transformed_x:.2f}, {transformed_y:.2f})")
            print(f"  Estimated Shift: ΔX = {transformed_x - gt_x:.2f}, ΔY = {transformed_y - gt_y:.2f}")
            print("-------------------------------------------------")

    print("\nTransformation validation complete.")