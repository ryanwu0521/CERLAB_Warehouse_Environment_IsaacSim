# =========================================
# Simulation Handler Module              
# =========================================

# Standard Library Imports
import sys

# Configuration
import config

# OmniVerse & Isaac Sim Utilities
import carb
import omni.isaac.core.utils.stage as stage_utils


# =========================================
# Simulation Handler Class              
# =========================================
class SimulationHandler:
    """
    Handles the simulation lifecycle, including:
    - Loading the simulation stage.
    - Managing updates during execution.
    - Handling cleanup and safe termination.
    """

    def __init__(self, kit):
        """
        Initialize the simulation handler.

        Args:
            kit: The simulation application kit (Isaac Sim instance).
        """
        self.kit = kit  # Store the simulation instance
        self.load_stage()  # Load the USD stage into the scene

    # =========================================
    # USD Stage Loading              
    # =========================================
    def load_stage(self):
        """
        Loads the simulation stage from a USD file.

        The stage file is specified in the configuration (config.USD_PATH).
        If loading fails, logs an error and exits the application.
        """
        try:
            # Load the USD file into the simulation
            stage_utils.add_reference_to_stage(config.USD_PATH, "/World/Warehouse")
            print("Stage loaded successfully.")

        except Exception as e:
            # Handle errors gracefully
            carb.log_error(f"Failed to load stage: {e}")
            self.kit.close()
            sys.exit(1)

    # =========================================
    # Simulation Update Loop              
    # =========================================
    def update_sim(self):
        """
        Updates the simulation application.

        This method should be called in a loop to continuously step the simulation forward.
        """
        self.kit.update()

    # =========================================
    # Simulation Cleanup              
    # =========================================
    def close_sim(self):
        """
        Safely closes the simulation application.

        Ensures that all resources are released before exiting.
        """
        print("Closing simulation...")
        self.kit.close()
