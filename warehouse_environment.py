import omni
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim

import numpy as np

# Import vehicle Usds
# usd_path = "omniverse://cerlabnucleus.lan.local.cmu.edu/Users/weihuanw/vehicles/theia.usd"  
usd_path = "file:///home/ryanwu/Documents/CERLAB/vehicles/theia.usd"
prim_path = "/World/Theia"
add_reference_to_stage(usd_path, prim_path)

# Set position
object_prim = XFormPrim(prim_path)
object_prim.set_world_pose([0, 0 ,0], [0, 0, 0, 1])


# Start the simulation loop
try:
    while True:
        # Update the simulation
        simulation_app.update()
except KeyboardInterrupt:
    # Allow graceful exit on Ctrl+C
    pass

# Run Commend in terminal: python3 isaac_python.py