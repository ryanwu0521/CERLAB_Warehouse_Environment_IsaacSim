# Omni & Isaac Sim Imports
import omni
from omni.isaac.kit import SimulationApp\
# Configuration
import config

# Initialize the simulation application
simulation_app = SimulationApp({"headless": config.HEADLESS_MODE})

import omni.isaac.core.utils.stage as stage_utils
from omni.usd import get_context
from pxr import UsdGeom

# Custom Modules
from simulation_handler import SimulationHandler
import feature_processing as fp
import graph_utils as gu
import isaac_utils as iu

# Standard Library Imports
import sys
import time
import atexit

# External Libraries
import numpy as np
import carb
import cv2
import networkx as nx
import matplotlib.pyplot as plt


# =========================================
# Main Function                         
# =========================================
def main():
    """Main function for feature extraction, matching, transformation, and visualization."""

    # Initialize simulation handler
    simulation_handler = SimulationHandler(simulation_app)
    atexit.register(simulation_handler.close_sim)  # Ensure cleanup when exiting

    # Step 1: Query features from the USD stage
    all_features = fp.query_features()

    # Step 2: Partition features into two maps
    map_a_features, map_b_features, _ = fp.partition_features(all_features)

    # Step 3: Save Ground Truth (GT) graph and individual maps
    gu.save_graph(gu.build_graph_from_feature_list(all_features), "gt", "gt_feature_graph")
    gu.save_graph(gu.build_graph_from_feature_list(map_a_features), "gt", "gt_map_a_graph")
    gu.save_graph(gu.build_graph_from_feature_list(map_b_features), "gt", "gt_map_b_graph")

    # Step 4: Apply Gaussian noise to feature positions
    noised_map_a_features = fp.apply_gaussian_noise(map_a_features, seed=42)
    noised_map_b_features = fp.apply_gaussian_noise(map_b_features, seed=24)
    print(f"\nGaussian noise applied with stddev={config.NOISE_STDDEV} meters.")

    # Step 5: Save Noised Versions of Map A and Map B
    gu.save_graph(gu.build_graph_from_feature_list(noised_map_a_features), "noise", "noised_map_a_graph")
    gu.save_graph(gu.build_graph_from_feature_list(noised_map_b_features), "noise", "noised_map_b_graph")

    # Step 6: Match features using cosine similarity
    matches = fp.match_features(
        noised_map_a_features, 
        noised_map_b_features,
        threshold=config.MATCH_THRESHOLD, 
        max_distance=config.MAX_MATCH_DISTANCE
    )
    print(f"\nFeature matching complete. Found {len(matches)} matched features.")

    # Step 7: Estimate transformation matrix using RANSAC
    transformation_matrix, _ = fp.estimate_transform_ransac(matches)
    print(f"\nTransformation matrix estimated:\n{transformation_matrix}")

    # Step 8: Apply transformation to noised map B features
    transformed_noised_map_b_features = fp.transform_features(noised_map_b_features, transformation_matrix)

    # Step 9: Merge maps and save the fused noised feature graph
    fused_noised_features = fp.merge_features(noised_map_a_features, transformed_noised_map_b_features, matches)
    gu.save_graph(gu.build_graph_from_feature_list(fused_noised_features), "noise", "noised_fused_feature_graph")

    # Step 10: Run the simulation loop
    try:
        while simulation_app.is_running():
            simulation_handler.update_sim()
            time.sleep(0.01)
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        simulation_handler.close_sim()
        sys.exit(0)


# =========================================
# Entry Point                           
# =========================================
if __name__ == "__main__":
    main()


# =========================================
# Notes:
# Run Program in vs terminal: 
# Windows: python isaac_python.py
# Linux: python3 isaac_python.py
# Open Isaac Standalone Isaac Sim in Windows Command Prompt
# "C:\Users\RyanWu\AppData\Local\ov\pkg\isaac-sim\isaac-sim.selector.bat"
# =========================================