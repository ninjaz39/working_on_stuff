import numpy as np
from helper import get_average_heat, dist_to_obstacle
from sampling import sample_ball
'''def estimate_next():
    sampled_points = []
    sampled_vals = []
    for i in range(n_walks):
        sample = sample_ball(dim, radius, start)
        heat_vals = 0
        num_vals = 0
        for i in range(100):
            samples = get_neighbors(domain, sample_ball(dim, radius, sample) - min_corner)
            for _, heat in samples:
                if heat > 0:
                    heat_vals += heat
                    num_vals += 1

        if heat_vals>0:   
            sampled_points.append(sample)
            sampled_vals.append(heat_vals/num_vals)

    if len(sampled_points)>0:
        start += estimate_gradient(sampled_points, sampled_vals, start, center_value=None, trim_bottom=0) * radius/10'''

def estimate_gradient(center_point, domain, min_corner, dim, radius, n_walks, center_value=None, trim_bottom=0.0):
    sampled_points = []
    sampled_vals = []
    for i in range(n_walks):
        sample = sample_ball(dim, radius, center_point)
        heat_vals = 0
        num_vals = 0
        for i in range(10):
            heat_vals = get_average_heat(domain, sample_ball(dim, radius, sample) - min_corner, radius = 1)
    
        if heat_vals>0:   
            sampled_points.append(sample)
            sampled_vals.append(heat_vals)

    if len(sampled_points)==0:
        return center_point
    
    points = np.asarray(sampled_points, dtype=float)
    values = np.asarray(sampled_vals, dtype=float)
    center_point = np.asarray(center_point, dtype=float)

    # Remove bottom X% of values
    if trim_bottom > 0:
        threshold = np.quantile(values, trim_bottom)
        mask = values >= threshold
        points = points[mask]
        values = values[mask]

    A = points - center_point

    if center_value is not None:
        b = values - center_value
        gradient = np.linalg.lstsq(A, b, rcond=None)[0]
    else:
        A_aug = np.hstack([A, np.ones((len(A), 1))])
        result = np.linalg.lstsq(A_aug, values, rcond=None)[0]
        gradient = result[:-1]
    if np.linalg.norm(gradient):
        return center_point + gradient / np.linalg.norm(gradient)# * radius/5
    else:
        return center_point


def check_path(path, new_heat, domain, min_corner):
    curr_heat = get_average_heat(domain, path[-1]-min_corner, 3)
    return curr_heat <= new_heat


def filter_paths(path, distance_field, min_corner_field):
    final_path = [path[-1]]
    final_final_path = [path[-1]]
    curr_heat = 0
    checking = len(path) - 2
    while checking>1:
        checking -= 1
        if curr_heat != -1 and check_intercept(final_path[-1], path[checking], distance_field, min_corner_field):
            final_path.append(path[checking + 1])
        
    final_direction = final_path[-1] - path[0]
    final_direction_length = np.linalg.norm(final_direction)
    if final_direction_length >= 5:
        final_direction /= final_direction_length
        final_path.append(path[0] + final_direction*5)


    if len(final_path) > 2:
        for i in range(len(final_path)-1, 0, -1):
            if not check_intercept(final_path[0], final_path[i], distance_field, min_corner_field):
                break

        final_final_path = np.append(np.array([final_path[0]]), final_path[i:], axis=0)
        return final_final_path
    

    return final_path

def check_intercept(start, end, distance_field, min_corner_field):
    direction = end - start
    distance = np.linalg.norm(direction)
    if not distance:
        return False
    direction /= distance
    next = start.copy()
    while distance > 0:
        dist_o = dist_to_obstacle(distance_field, min_corner_field, next)
        if dist_o <= 5:
            return True

        step = min(dist_o - 5, distance)
        next += direction * step
        distance -= step
    return False