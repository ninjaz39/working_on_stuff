import numpy as np
from shapes import Sphere
from itertools import product

def merge_domains_nd(domain_a, min_corner_a, min_corner_b, new_shape):
    min_corner_a = np.array(min_corner_a, dtype=float)
    min_corner_b = np.array(min_corner_b, dtype=float)

    min_corner   = np.minimum(min_corner_a, min_corner_b)
    max_corner   = np.maximum(min_corner_a + np.array(domain_a.shape), min_corner_b + np.array(new_shape))
    canvas_shape = tuple(np.round(max_corner - min_corner).astype(int).tolist())

    canvas = np.full(canvas_shape, -1.0)
    rel    = np.round(min_corner_a - min_corner).astype(int)
    slices = tuple(slice(int(rel[d]), int(rel[d]) + domain_a.shape[d]) for d in range(domain_a.ndim))
    canvas[slices] = domain_a

    return canvas, min_corner

def update_neighbors(array, N, v,radius = 1):
    if len(array)<=1:
        return 0
    """
    Retrieve values of all integer grid nodes within radius 1 of floating point position N.

    Args:
        array: n-dimensional numpy array of shape (d, d, ..., d)
        N:     array-like of n floats, the fractional position

    Returns:
        List of (index_tuple, value) pairs for all nodes within radius 1
    """
    N = np.asarray(N, dtype=float)
    shape = array.shape

    ranges = []
    for i, coord in enumerate(N):
        lo = int(np.ceil(coord-radius))
        hi = int(np.floor(coord+radius))
        ranges.append(range(max(0, lo), min(shape[i], hi + 1)))

    results = []
    for idx in product(*ranges):
        idx_arr = np.array(idx)
        if np.linalg.norm(idx_arr - N) <= radius:
            heat = array[tuple(idx)]
            if heat:
                array[tuple(idx)] = (heat + v)/2
            else:
                array[tuple(idx)] = v

    return results

def get_neighbors(array, N, radius = 1):
    if len(array)<=1:
        return []
    """
    Retrieve values of all integer grid nodes within radius 1 of floating point position N.

    Args:
        array: n-dimensional numpy array of shape (d, d, ..., d)
        N:     array-like of n floats, the fractional position

    Returns:
        List of (index_tuple, value) pairs for all nodes within radius 1
    """
    N = np.asarray(N, dtype=float)
    shape = array.shape

    ranges = []
    for i, coord in enumerate(N):
        lo = int(np.ceil(coord-radius))
        hi = int(np.floor(coord+radius))
        ranges.append(range(max(0, lo), min(shape[i], hi + 1)))

    results = []
    for idx in product(*ranges):
        idx_arr = np.array(idx)
        if np.linalg.norm(idx_arr - N) <= radius:
            node = array[tuple(idx)]
            if node is not None:          # ← skip unfilled cells
                results.append((idx, node))

    return results

def get_average_heat(array, N, radius = 1):
    nodes = get_neighbors(array, N, radius)
    total = 0
    num_vals = 0
    for _, i in nodes:
        if i>0:
            total += i
            num_vals += 1
    if num_vals == 0:
        return 0
    return total / num_vals
    

def dist_to_edge(point, array):
    point = np.asarray(point)
    shape = np.asarray(array.shape)
    
    dist_to_min = point
    dist_to_max = shape - 1 - point
    
    return np.min(np.minimum(dist_to_min, dist_to_max))

def dist_to_obstacle(distance_field, min_corner_field, start, radius = 1):
    dist_b = float('inf')
    dist_o = get_neighbors(distance_field, start-min_corner_field, radius)
    for _, i in dist_o:
        if i < dist_b:
            dist_b = i
    return dist_b