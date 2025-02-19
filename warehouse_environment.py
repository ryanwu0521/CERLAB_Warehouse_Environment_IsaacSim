# Omni and USD imports
import omni
from omni.isaac.kit import SimulationApp
# Initialize the simulation application
simulation_app = SimulationApp({"headless": True})
import omni.isaac.core.utils.stage as stage_utils

import sys
import signal
import time
import atexit
import numpy as np
import carb

from pxr import UsdGeom, Sdf, Gf

# graph structure and drawing imports
import networkx as nx
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D  # 3D plotting

# custom graph utility imports
from graph_utils import save_feature_graph_to_json

# custom isaac utility imports
from isaac_utils import get_world_transform


#########################################
# Feature Class Definition for Fusion   #
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
# SimulationHandler Class               #
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

    # @staticmethod
    # def create_feature_sphere(stage, path, position, radius=0.5, color=(1.0, 0.0, 0.0)):
    #     """
    #     Create a sphere at a given position with the specified radius and color.

    #     Args:
    #         stage: The USD stage.
    #         path: The path for the new sphere prim.
    #         position: A tuple representing the (x, y, z) coordinates.
    #         radius: The radius of the sphere.
    #         color: A tuple representing the RGB color.
    #     """
    #     # Define the sphere prim
    #     sphere_prim = stage.DefinePrim(path, "Sphere")

    #     # Set sphere properties
    #     sphere = UsdGeom.Sphere(sphere_prim)
    #     sphere.GetRadiusAttr().Set(radius)
        
    #     # Set the sphere's position using a transform
    #     xform = UsdGeom.XformCommonAPI(sphere_prim)
    #     xform.SetTranslate(Gf.Vec3d(*position))
        
    #     # Create a displayColor attribute and set the sphere's color
    #     sphere_prim.CreateAttribute("displayColor", Sdf.ValueTypeNames.Color3f, custom=True).Set(Gf.Vec3f(*color))

    #     # Print a message to confirm the sphere creation
    #     print(f"Sphere created at {position} with radius {radius} and color {color}.")

    @staticmethod
    def build_feature_graph(rack_positions, crane_positions, forklift_positions):
        """
        Build a graph representing the feature points with nodes for each feature and edges
        representing proximity relationships.
        
        Args:
            rack_positions: List of positions for rack features.
            crane_positions: List of positions for crane features.
            forklift_positions: List of positions for forklift features.
        
        Returns:
            A NetworkX graph with nodes and edges.
        """
        feature_graph = nx.Graph()

        # Helper function to add nodes with attributes
        def add_feature_nodes(graph, positions, feature_type, prefix):
            for i, pos in enumerate(positions, start=1):
                # Create a Feature object for each node
                feature = Feature(
                    feature_id=f"{prefix}{i}", 
                    feature_type=feature_type, 
                    position=pos,
                    orientation=None,            # For example, use None or a default quaternion [0, 0, 0, 1]
                    descriptor=None,             # Or compute/assign a descriptor if available
                    scale=1.0,                   # Adjust as needed
                    covariance=np.eye(3) * 0.01,   # Default low uncertainty
                    confidence=1.0,              # Confidence score (could be updated based on detection quality)
                    source_map="Warehouse",      # Identifier for the source map
                    timestamp=None             # Use a valid timestamp if applicable (e.g., time.time())
                )

                # Add the feature object as an attribute to the node
                graph.add_node(f"{prefix}{i}", feature=feature)

        # Add nodes for each type of feature
        add_feature_nodes(feature_graph, rack_positions, "rack", "Rack")
        add_feature_nodes(feature_graph, crane_positions, "crane", "Crane")
        add_feature_nodes(feature_graph, forklift_positions, "forklift", "Forklift")

        # Define a helper to compute Euclidean distance
        def euclidean_distance(p1, p2):
            return np.linalg.norm(p1 - p2)

        # Define a threshold below which nodes will be connected by an edge.
        distance_threshold = 2500  # Adjust as needed for your simulation

        # Iterate over all pairs of nodes and add an edge if they are close enough.
        nodes_data = list(feature_graph.nodes(data=True))
        for i in range(len(nodes_data)):
            for j in range(i + 1, len(nodes_data)):
                pos1 = nodes_data[i][1]['feature'].position
                pos2 = nodes_data[j][1]['feature'].position
                if euclidean_distance(pos1, pos2) < distance_threshold:
                    feature_graph.add_edge(nodes_data[i][0], nodes_data[j][0])

        return feature_graph

    @staticmethod
    def draw_feature_graph(feature_graph):
        """
        Draws the feature graph in 3D using Matplotlib.
        
        Nodes are colored based on feature type:
            - Rack: blue
            - Crane: red
            - Forklift: green
        Edges are drawn as gray lines connecting the nodes.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Define a color map for different feature types.
        color_map = {"rack": "blue", "crane": "red", "forklift": "green"}
        
        # Draw nodes.
        for node, data in feature_graph.nodes(data=True):
            # pos = data['position']
            # node_color = color_map.get(data['type'], "black")
            feature = data['feature']
            pos = feature.position
            node_color = color_map.get(feature.feature_type, "black")
            ax.scatter(pos[0], pos[1], pos[2], color=node_color, s=50)
            # Label the node.
            ax.text(pos[0], pos[1], pos[2], f"{node}", size=10, zorder=1, color='k')
        
        # Draw edges.
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
        plt.show()


#########################################
# Main Function                         #
#########################################
def main():
    """
    Main function to initialize the simulation, create feature spheres,
    and run the simulation loop.
    """

    # Create the simulation handler instance
    simulation_handler = SimulationHandler(simulation_app)

    # Register cleanup function to close the simulation on exit
    atexit.register(simulation_handler.close_sim)

    # Get the current USD stage
    from omni.usd import get_context
    stage = get_context().get_stage()

    # ---------------------------------------------------
    # 1) Retrieve features from the "/World/Warehouse/Features" prim
    # ---------------------------------------------------
    feature_prim_path = "/World/Warehouse/Features"   # MAKE SURE TO MATCH THE PATH IN YOUR USD FILE
    features_prim = stage.GetPrimAtPath(feature_prim_path)

    if not features_prim.IsValid():
        print(f"Feature prim at {feature_prim_path} is not valid.")
        sys.exit(1)

    # --- Debug print to check prim name ---
    direct_children = features_prim.GetChildren()
    # print(f"Direct children of {feature_prim_path}:")
    # for child in direct_children:
    #     print("  -", child.GetName(), child.GetPath())

    # # Hard-code GT positions for various features
    # crane_positions = [
    #     (1000, 0, 0),
    #     (2700, 0, 0),
    #     (4700, 0, 0),
    #     (6250, 0, 0),
    # ]

    # forklift_positions = [
    #     (1620, 0, 0),
    #     (3620, 0, 0),
    #     (5620, 0, 0),
    # ]

    # rack_positions = [
    #     (953, -2230, 0),
    #     (1620, -2230, 0),
    #     (2287, -2230, 0),
    #     (953, 2230, 0),
    #     (1620, 2230, 0),
    #     (2287, 2230, 0),

    #     (2953, -2230, 0),
    #     (3620, -2230, 0),
    #     (4287, -2230, 0),
    #     (2953, 2230, 0),
    #     (3620, 2230, 0),
    #     (4287, 2230, 0),

    #     (4953, -2230, 0),
    #     (5620, -2230, 0),
    #     (6287, -2230, 0),
    #     (4953, 2230, 0),
    #     (5620, 2230, 0),
    #     (6287, 2230, 0),
    # ]

    # # Create spheres for racks (blue), cranes (red), and forklifts (green) (for visualization)
    # for i, position in enumerate(rack_positions, start=1):
    #     SimulationHandler.create_feature_sphere(
    #         stage,
    #         path=f"/World/Spheres/Rack{i}",
    #         position=position,
    #         radius=50,
    #         color=(0.0, 0.0, 1.0),
    #     )

    # for i, position in enumerate(crane_positions, start=1):
    #     SimulationHandler.create_feature_sphere(
    #         stage,
    #         path=f"/World/Spheres/Crane{i}",
    #         position=position,
    #         radius=50,
    #         color=(1.0, 0.0, 0.0),
    #     )

    # for i, position in enumerate(forklift_positions, start=1):
    #     SimulationHandler.create_feature_sphere(
    #         stage,
    #         path=f"/World/Spheres/Forklift{i}",
    #         position=position,
    #         radius=50,
    #         color=(0.0, 1.0, 0.0),
    #     )

    # List to store the positions of the features
    rack_positions = []
    crane_positions = []
    forklift_positions = []

    # ---------------------------------------------------
    # 2) Recursively traverse the feature prim to extract feature positions
    # ---------------------------------------------------
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
                translation, rotation_quat, scale = get_world_transform(subchild)
                position = np.array([translation[0], translation[1], translation[2]])
                lower_name = prim_name.lower()

                if "crane" in lower_name:
                    crane_positions.append(position)
                    print(f"  Found Crane: {prim_name} at {position}")
                elif "forklift" in lower_name:
                    forklift_positions.append(position)
                    print(f"  Found Forklift: {prim_name} at {position}")
                elif "rack" in lower_name:
                    rack_positions.append(position)
                    print(f"  Found Rack: {prim_name} at {position}")
                else:
                    print(f"  Unknown feature type for: {prim_name}")
            else:
                print(f"  Skipping non-Xformable prim: {subchild.GetName()}")

    # ---------------------------------------------------
    # 3) Build a graph representing the feature points
    # ---------------------------------------------------
    feature_graph = SimulationHandler.build_feature_graph(rack_positions, crane_positions, forklift_positions)

    # ---------------------------------------------------
    # 4) Save the feature graph to a JSON file
    # ---------------------------------------------------
    save_feature_graph_to_json(feature_graph, filename="feature_graph.json")

    # print("\nGraph Nodes with Attributes:")
    # for node, data in feature_graph.nodes(data=True):
    #     print(f"{node}: {data}")

    # print("\nGraph Edges:")
    # for edge in feature_graph.edges():
    #     print(edge)

    # Draw the feature graph in 3D
    SimulationHandler.draw_feature_graph(feature_graph)
    plt.savefig('feature_graph.png')


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

# Run Commend in terminal: python3 isaac_python.py