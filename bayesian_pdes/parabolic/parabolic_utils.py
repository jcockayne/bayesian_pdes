import numpy as np
def augment_with_time(spatial_points, time):
    return np.column_stack([spatial_points, time * np.ones((spatial_points.shape[0], 1))])
