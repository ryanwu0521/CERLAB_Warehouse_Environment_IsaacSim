"""
isaac_utils.py

Utility functions for Isaac Sim / Omniverse USD operations.
"""

from pxr import UsdGeom, Usd, Gf
import math
import numpy as np

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

    # Extract scaling as a Gf.Vec3f
    # scale = transform_matrix.ExtractScaling()

    col0 = transform_matrix.GetColumn(0)
    col1 = transform_matrix.GetColumn(1)
    col2 = transform_matrix.GetColumn(2)
    scale_x = math.sqrt(col0[0]**2 + col0[1]**2 + col0[2]**2)
    scale_y = math.sqrt(col1[0]**2 + col1[1]**2 + col1[2]**2)
    scale_z = math.sqrt(col2[0]**2 + col2[1]**2 + col2[2]**2)
    scale = Gf.Vec3f(scale_x, scale_y, scale_z)

    return translation, rotation_quat, scale

def get_bounding_box(prim):
    """Returns the bounding box (min, max) for a given USD primitive."""
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(prim)
    return np.array(bbox.GetBox().GetMin()), np.array(bbox.GetBox().GetMax())