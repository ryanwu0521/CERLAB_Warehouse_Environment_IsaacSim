import numpy as np

#########################################
# Rigid Transformation Functions        #
#########################################
def compute_rigid_transform(points_B, points_A):
    """
    Compute the rigid transformation (R, t) aligning points_B to points_A.
    points_B and points_A are numpy arrays of shape (N, 3) where N >= 3.
    Returns rotation matrix R and translation vector t.
    """
    assert points_B.shape == points_A.shape, "Point sets must have the same shape."
    
    centroid_B = np.mean(points_B, axis=0)
    centroid_A = np.mean(points_A, axis=0)
    
    # Center the points.
    BB = points_B - centroid_B
    AA = points_A - centroid_A
    
    H = BB.T @ AA
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Reflection case.
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        
    t = centroid_A - R @ centroid_B
    return R, t


def apply_transformation_to_graph(feature_graph, R, t):
    """
    Applies a rigid transformation (R, t) to all feature positions in the graph.
    """
    for node, data in feature_graph.nodes(data=True):
        feature = data['feature']
        feature.position = R @ feature.position + t