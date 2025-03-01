# Omni and USD imports
import omni
from omni.isaac.kit import SimulationApp
# Initialize the simulation application
simulation_app = SimulationApp({"headless": False})
import omni.isaac.core.utils.stage as stage_utils
from omni.usd import get_context

# python imports
import sys
import time 
import atexit
import numpy as np
import carb

from pxr import UsdGeom, Sdf, Gf

# Custom graph utility imports
import graph_utils as gu

# Custom isaac utility imports
import isaac_utils as iu

# Import NetworkX for graph visualization
import networkx as nx
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D  # 3D plotting


#########################################
# Feature Class Definition              #
#########################################
class Feature:
    """
        Feature representation for multi-map fusion.

        Args:
            feature_id (str): A unique identifier for the feature.
            feature_type (str): Type of semantic label (e.g., 'rack', 'crane', 'forklift').
            position (tuple or list): (x, y, z) coordinates.
            orientation (list): The orientation of the feature.
            descriptor (list): The feature descriptor.
            scale (float): The feature scale.
            covariance (list): The feature covariance.
            confidence (float): The feature confidence.
            source_map (str): The source map for the feature.
            timestamp (float): The timestamp for the feature.
    """
    def __init__(self, feature_id, feature_type, position, orientation=None,
                 descriptor=None, scale=1.0, covariance=None, confidence=1.0,
                 source_map=None, timestamp=None):
        
        self.feature_id = feature_id
        self.feature_type = feature_type
        self.position = np.array(position)
        self.orientation = orientation
        self.descriptor = descriptor
        self.scale = scale
        self.covariance = covariance if covariance is not None else np.eye(3) * 0.01
        self.confidence = confidence
        self.source_map = source_map
        self.timestamp = timestamp

    def to_dict(self):
        """Return a dictionary representation of the feature for JSON serialization."""
        data = {
            "id": self.feature_id,
            "type": self.feature_type,
            "position": self.position.tolist(),
            "scale": self.scale,
            "confidence": self.confidence,
            "source_map": self.source_map,
            "timestamp": self.timestamp,
        }
        if self.orientation is not None:
            data["orientation"] = self.orientation
        if self.covariance is not None:
            data["covariance"] = self.covariance.tolist()
        if self.descriptor is not None:
            data["descriptor"] = self.descriptor.tolist() if isinstance(self.descriptor, np.ndarray) else self.descriptor
        return data

    def __repr__(self):
        """Return a string representation of the feature."""
        return f"Feature(id={self.feature_id}, type={self.feature_type}, pos={self.position}, scale={self.scale})"


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
    """
    Main function that:
      1) Uses a prim query to extract feature positions from the USD stage.
      2) Partitions the features into two maps (MapA and MapB) based on their x-coordinate.
         Features near the partition threshold are added to both maps (overlap).
      3) Builds and save separate graphs for MapA and MapB (json & png).
      4) (To-Do) Further processing such as transformation and fusion.
    """

    # Initialize simulation
    simulation_handler = SimulationHandler(simulation_app)
    # Register cleanup function to close the simulation on exit
    atexit.register(simulation_handler.close_sim)

    # Get the current USD stage
    stage = get_context().get_stage()

    # ---------------------------------------------------
    # 1) Query Features from the USD Stage
    # ---------------------------------------------------
    feature_prim_path = "/World/Warehouse/Features"   # MAKE SURE TO MATCH THE PATH IN YOUR USD FILE
    features_prim = stage.GetPrimAtPath(feature_prim_path)
    if not features_prim.IsValid():
        print(f"Feature prim at {feature_prim_path} is not valid.")
        sys.exit(1)

    # Extract feature positions from the USD stage
    all_features = []
    direct_children = features_prim.GetChildren()
    print("Querying features from the USD stage...")

    # Process only the direct children of the feature prim
    for container in direct_children:
        container_name = container.GetName()
        print(f"\nProcessing: {container_name} ({container.GetPath()})")
        sub_children = container.GetChildren()
        if not sub_children:
            print(f"  No direct children found under {container_name}.")
        for subchild in sub_children:
            if not subchild.IsValid():
                continue

            prim_name = subchild.GetName()
            # Process only if the subchild is transformable
            if subchild.IsA(UsdGeom.Xformable):
                translation, rotation_quat, scale = iu.get_world_transform(subchild)
                position = np.array([translation[0], translation[1], translation[2]])
                lower_name = prim_name.lower()
                # Determine feature type based on prim name
                if "crane" in lower_name:
                    feature_type = "crane"
                elif "forklift" in lower_name:
                    feature_type = "forklift"
                elif "rack" in lower_name:
                    feature_type = "rack"
                else:
                    feature_type = "unknown"

                # Use the prim name as the feature ID
                feature = Feature(feature_id=prim_name, feature_type=feature_type, position=position, source_map="Warehouse")
                all_features.append(feature)
                print(f"  Found {feature_type.capitalize()}: {prim_name} at {position}")
            else:
                print(f"  Skipping non-Xformable prim: {subchild.GetName()}")
    
        if not all_features:
            print("No features found via prim query.")
            sys.exit(1)

    # ---------------------------------------------------
    # 2) partition the features into two maps (Map_A and Map_B) based on their x-coordinate
    # ---------------------------------------------------
    # Partition based on the x-coordinate. Features near the median are considered overlapping.
    all_x = np.array([f.position[0] for f in all_features])
    threshold = np.median(all_x)
    margin = 1000  # Define margin for overlap (adjust as needed)

    map_a_features = []
    map_b_features = []
    overlapping_features = set()

    for feature in all_features:
        if feature.position[0] < threshold - margin:
            map_a_features.append(feature)
        elif feature.position[0] > threshold + margin:
            map_b_features.append(feature)
        else:
            # Feature is in the overlapping region.
            map_a_features.append(feature)
            map_b_features.append(feature)
            overlapping_features.add(feature.feature_id)
    
    print(f"\nScene partitioning complete. Threshold (x): {threshold}, Overlap margin: {margin}")
    print(f"Map_A features: {len(map_a_features)}; Map_B features: {len(map_b_features)}; Overlapping features: {len(overlapping_features)}")
    
    # ---------------------------------------------------
    # 3) Build, Print & Save Graphs from the Feature Lists
    # ---------------------------------------------------
    complete_feature_graph = gu.build_graph_from_feature_list(all_features)
    feature_graph_a = gu.build_graph_from_feature_list(map_a_features)
    feature_graph_b = gu.build_graph_from_feature_list(map_b_features)

    # Apply Gaussian noise to the feature positions
    # apply_gaussian_noise(feature_graph, noise_stddev=0.01)
    # print("\nFeature Graph with Gaussian Noise:")
    # for node, data in feature_graph.nodes(data=True):
    #     print(f"{node}: {data}")   

    # Save the feature graphs to JSON files
    gu.save_feature_graph_to_json(complete_feature_graph, filename="complete_feature_graph.json")
    gu.save_feature_graph_to_json(feature_graph_a, filename="feature_graph_A.json")
    gu.save_feature_graph_to_json(feature_graph_b, filename="feature_graph_B.json")

    # Draw the feature graphs in 3D
    print("\nDrawing & saving the complete feature graph in 3D...")
    gu.draw_feature_graph(complete_feature_graph)
    plt.savefig('complete_feature_graph.png')
    print("\nDrawing & saving the feature graph A in 3D...")
    gu.draw_feature_graph(feature_graph_a)
    plt.savefig('feature_graph_A.png')
    print("\nDrawing & saving the feature graph B in 3D...")
    gu.draw_feature_graph(feature_graph_b)
    plt.savefig('feature_graph_B.png')

    # ---------------------------------------------------
    # Main simulation loop
    # ---------------------------------------------------
    try:
        # Use the simulation app's status to control the loop
        while simulation_app.is_running():
            simulation_handler.update_sim()
            time.sleep(0.01)   # Sleep for 10ms between updates
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