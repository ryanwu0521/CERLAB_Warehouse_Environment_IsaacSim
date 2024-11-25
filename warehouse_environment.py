import omni

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.isaac.core.utils.stage as stage_utils

import numpy as np
import carb
import sys
import signal



class SimulationHandler:
    """
    Class to handle the simulation application.

    Attributes:
    kit: The simulation kit.
    world: The simulation world.
    """

    def __init__(self, kit=None):
        """
        Initialize the simulation handler.

        Args:
        kit: The simulation kit.
        """
        self.kit = kit
        self.load_stage()
       

    def load_stage(self):
        """
        Load the simulation stage.
        """
        usd_path = "omniverse://cerlabnucleus.lan.local.cmu.edu/Users/weihuanw/sim_environments/large_modular_warehouse_v1.usd"

        prim_path = "/World/warehouse"

        # Check if USD path is valid
        if not usd_path.startswith("omniverse://"):
            carb.log_error(f"Invalid USD path: {usd_path}")
            self.kit.close()
            sys.exit(1)

        # Load the stage
        try:
            # Load the stage reference
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

    
    def signal_handler(self, sig, frame):
        """
        Handle termination signals to close the simulation properly.
        """
        print("Signal received, closing simulation...")
        self.kit.close()
        sys.exit(0)


def main():
    """
    Main function to run the simulation.
    """
    global simulation_handler
    simulation_handler = SimulationHandler(simulation_app)

    # Set up signal handling
    signal.signal(signal.SIGINT, simulation_handler.signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, simulation_handler.signal_handler) # Termination signal

    try:
        while True:
            simulation_handler.update_sim()
    except Exception as e:
        print(f"Error during simulation: {e}")
        simulation_handler.close_sim()
        sys.exit(1)


if __name__ == "__main__":
    main()

# Run Commend in terminal: python3 isaac_python.py