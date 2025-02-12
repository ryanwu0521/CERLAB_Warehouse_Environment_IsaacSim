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


    @staticmethod
    def create_feature_sphere(stage, path, position, radius=0.5, color=(1.0, 0.0, 0.0)):
        """
        Create a sphere at a given position with the specified radius and color.

        Args:
            stage: The USD stage.
            path: The path for the new sphere prim.
            position: A tuple representing the (x, y, z) coordinates.
            radius: The radius of the sphere.
            color: A tuple representing the RGB color.
        """
        # Define the sphere prim
        sphere_prim = stage.DefinePrim(path, "Sphere")

        # Set sphere properties
        sphere = UsdGeom.Sphere(sphere_prim)
        sphere.GetRadiusAttr().Set(radius)
        
        # Set the sphere's position using a transform
        xform = UsdGeom.XformCommonAPI(sphere_prim)
        xform.SetTranslate(Gf.Vec3d(*position))
        
        # Create a displayColor attribute and set the sphere's color
        sphere_prim.CreateAttribute("displayColor", Sdf.ValueTypeNames.Color3f, custom=True).Set(Gf.Vec3f(*color))

        # Print a message to confirm the sphere creation
        print(f"Sphere created at {position} with radius {radius} and color {color}.")


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
                # Store the position as a NumPy array
                graph.add_node(f"{prefix}{i}", type=feature_type, position=np.array(pos))

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
                pos1 = nodes_data[i][1]['position']
                pos2 = nodes_data[j][1]['position']
                if euclidean_distance(pos1, pos2) < distance_threshold:
                    feature_graph.add_edge(nodes_data[i][0], nodes_data[j][0])

        return feature_graph


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
            pos = data['position']
            node_color = color_map.get(data['type'], "black")
            ax.scatter(pos[0], pos[1], pos[2], color=node_color, s=50)
            # Label the node.
            ax.text(pos[0], pos[1], pos[2], f"{node}", size=10, zorder=1, color='k')
        
        # Draw edges.
        for edge in feature_graph.edges():
            pos1 = feature_graph.nodes[edge[0]]['position']
            pos2 = feature_graph.nodes[edge[1]]['position']
            xs = [pos1[0], pos2[0]]
            ys = [pos1[1], pos2[1]]
            zs = [pos1[2], pos2[2]]
            ax.plot(xs, ys, zs, color="gray", alpha=0.7)
        
        ax.set_xlabel("X (Meters)")
        ax.set_ylabel("Y (Meters)")
        ax.set_zlabel("Z (Meters)")
        plt.title("Factor Graph Structure for Warehouse Environment")
        plt.show()


def main():
    """
    Main function to initialize the simulation, create feature spheres,
    and run the simulation loop.
    """

    # global simulation_handler
    # Create the simulation handler instance
    simulation_handler = SimulationHandler(simulation_app)

    # Register cleanup function to close the simulation on exit
    # atexit.register(lambda: simulation_handler.close_sim() if 'simulation_handler' in globals() else None)
    atexit.register(simulation_handler.close_sim)

    # Get the current USD stage
    from omni.usd import get_context

    stage = get_context().get_stage()

    # Define positions for various features
    rack_positions = [
        (953, -2230, 300),
        (1620, -2230, 300),
        (2287, -2230, 300),
        (2953, -2230, 300),
        (3620, -2230, 300),
        (4287, -2230, 300),
        (4953, -2230, 300),
        (5620, -2230, 300),
        (6287, -2230, 300),
        (953, 2130, 300),
        (1620, 2130, 300),
        (2287, 2130, 300),
        (2953, 2123, 300),
        (3620, 2130, 300),
        (4287, 2130, 300),
        (4953, 2130, 300),
        (5620, 2130, 300),
        (6287, 2130, 300),
    ]

    crane_positions = [
        (1000, 0, 300),
        (2700, 0, 300),
        (4700, 0, 300),
        (6250, 0, 300),
    ]

    forklift_positions = [
        (1620, 0, 300),
        (3620, 0, 300),
        (5620, 0, 300),
    ]

    # Create spheres for racks (blue), cranes (red), and forklifts (green)
    for i, position in enumerate(rack_positions, start=1):
        SimulationHandler.create_feature_sphere(
            stage,
            path=f"/World/Spheres/Rack{i}",
            position=position,
            radius=50,
            color=(0.0, 0.0, 1.0),
        )

    for i, position in enumerate(crane_positions, start=1):
        SimulationHandler.create_feature_sphere(
            stage,
            path=f"/World/Spheres/Crane{i}",
            position=position,
            radius=50,
            color=(1.0, 0.0, 0.0),
        )

    for i, position in enumerate(forklift_positions, start=1):
        SimulationHandler.create_feature_sphere(
            stage,
            path=f"/World/Spheres/Forklift{i}",
            position=position,
            radius=50,
            color=(0.0, 1.0, 0.0),
        )

    # Build a graph representing the feature points
    feature_graph = SimulationHandler.build_feature_graph(rack_positions, crane_positions, forklift_positions)

    print("\nGraph Nodes with Attributes:")
    for node, data in feature_graph.nodes(data=True):
        print(f"{node}: {data}")

    print("\nGraph Edges:")
    for edge in feature_graph.edges():
        print(edge)

    # Draw the feature graph in 3D
    SimulationHandler.draw_feature_graph(feature_graph)
    plt.savefig('feature_graph.png')

    # Main simulation loop
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