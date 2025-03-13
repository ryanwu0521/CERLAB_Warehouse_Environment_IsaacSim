# =========================================
# Isaac Sim / Omniverse Utility Module              
# =========================================

# OmniVerse & USD Imports
from pxr import UsdGeom, Usd, Gf

# Standard Libraries
import math
import numpy as np


# =========================================
# USD Transform Extraction              
# =========================================
def get_world_transform(prim):
    """
    Compute and return the world-space translation, rotation, and scale for a given prim.

    Args:
        prim (pxr.Usd.Prim): The USD Prim whose transform is being queried.

    Returns:
        (translation, rotation_quat, scale)
        - translation (Gf.Vec3d): World-space translation.
        - rotation_quat (Gf.Quatf): World-space rotation in quaternion form.
        - scale (Gf.Vec3f): World-space scale.
    """
    # Create a Xformable interface from the prim
    xformable = UsdGeom.Xformable(prim)

    # Compute the local-to-world transform at the default time
    transform_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # Extract translation
    translation = transform_matrix.ExtractTranslation()

    # Extract rotation as a Gf.Quatf
    rotation_quat = transform_matrix.ExtractRotationQuat()

    # Extract scaling using column vectors
    col0 = transform_matrix.GetColumn(0)
    col1 = transform_matrix.GetColumn(1)
    col2 = transform_matrix.GetColumn(2)

    # Compute scale from column vectors
    scale_x = math.sqrt(col0[0]**2 + col0[1]**2 + col0[2]**2)
    scale_y = math.sqrt(col1[0]**2 + col1[1]**2 + col1[2]**2)
    scale_z = math.sqrt(col2[0]**2 + col2[1]**2 + col2[2]**2)
    scale = Gf.Vec3f(scale_x, scale_y, scale_z)

    return translation, rotation_quat, scale


# =========================================
# USD Bounding Box Computation              
# =========================================
def get_bounding_box(prim):
    """
    Computes the world-space bounding box (min, max) of a given USD primitive.

    Args:
        prim (pxr.Usd.Prim): The USD Prim whose bounding box is being queried.

    Returns:
        tuple: (bbox_min, bbox_max)
            - bbox_min (numpy.array): Minimum coordinates (x, y, z) of the bounding box.
            - bbox_max (numpy.array): Maximum coordinates (x, y, z) of the bounding box.
    """
    # Create a bounding box cache to compute the world-space bounding box
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    # Compute the bounding box
    bbox = bbox_cache.ComputeWorldBound(prim)

    # Extract min and max coordinates from the bounding box
    return np.array(bbox.GetBox().GetMin()), np.array(bbox.GetBox().GetMax())