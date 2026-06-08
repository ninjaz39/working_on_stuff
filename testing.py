import numpy as np
import matplotlib.pyplot as plt
from sampling import sample_sphere, sample_ball
from helper import merge_domains_nd, update_neighbors, get_neighbors, dist_to_edge, get_average_heat, dist_to_obstacle
from shapes import EgoSphere
from path import filter_paths, check_path, check_intercept, estimate_gradient
from animate import animate_2d


VIEW_RANGE = 100
MOVEMENT_RANGE= 5
EPSILON = 0.4999999

def load_ndarray(name: str) -> np.ndarray:
    """Companion loader that reconstructs the array from the saved file."""
    with open(name, "r") as f:
        n     = int(f.readline().strip())
        shape = tuple(int(s) for s in f.readline().strip().split())
        flat  = list(map(float, f.readline().strip().split()))

    return np.array(flat).reshape(shape)

def wos_walk_altered_g(radius_g, dim, distance_field, min_corner_field, goal, start, overall_domain, min_corner, curr_rec):
    if curr_rec == 800:
        return 0, []
    adjusted_start = start-min_corner
    dist_g = np.linalg.norm(start-goal)
    if dist_g < radius_g:
        point = (1/dist_g*100000000)
        update_neighbors(overall_domain, adjusted_start, point)
        return point, [start]
    
    dist_b = dist_to_edge(adjusted_start, overall_domain)
    obstacle_hit = False
    dist_o = dist_to_obstacle(distance_field, min_corner_field, start, 0.9999)

    if dist_o == None:
        return 0, []
    
    if dist_o < dist_b:
        dist_b = dist_o - 5
        obstacle_hit = True

    if obstacle_hit and dist_b <= 0:
        return None, []
    
    if (not obstacle_hit and dist_b<EPSILON) or dist_to_obstacle(overall_domain, min_corner, start)<0:

        point = (1/dist_g*100000000) 
        return point, []
    
    point = None
    while point == None:
        v = sample_sphere(dim, dist_b, start)
        point, next_steps = wos_walk_altered_g(radius_g, dim, distance_field, min_corner_field, goal, v, overall_domain, min_corner, curr_rec+1)

    update_neighbors(overall_domain, adjusted_start, point)
    if point <= 0:
        return point, []
    
    next_steps.append(start)
    return point, next_steps

def WoS_altered_g(distance_field, min_corner_field, goal, start, radius, n_walks, overall_domain, min_corner):
    dim = len(start)
    max_heat = 0
    final_path = [start.copy()]
    radius_g = distance_field[tuple(np.round(goal-min_corner_field).astype(int))]
    for walk in range(n_walks):
        first_step = sample_ball(dim, radius, start)
        heat, path = wos_walk_altered_g(radius_g, dim, distance_field, min_corner_field, goal, first_step, overall_domain, min_corner, 0)
        if len(path)>0:
            if heat > max_heat:
                max_heat = heat
                path.append(start)
                final_path = path
            elif heat == max_heat:
                if len(path) < len(final_path):
                    max_heat = heat
                    path.append(start)
                    final_path = path
    print(final_path[0], 'hi')
    return overall_domain, final_path, max_heat

def ego_centric_wos_mapping(path, domain, min_corner, start, goal, distance_field, min_corner_field, direction, n_walks=1000):
    radius = min(VIEW_RANGE-5, max(0,dist_to_obstacle(distance_field, min_corner_field, start)-5))
    if not radius:
        return domain, None, []
    
    domain, new_path, temp_heat = WoS_altered_g(distance_field, min_corner_field, goal, start, radius, n_walks, domain, min_corner)
    
    if len(new_path)>1 and check_path(path, temp_heat, domain, min_corner):
        new_path = filter_paths(new_path, distance_field, min_corner_field)
        path = new_path
             

    if len(path) < 2 or get_average_heat(domain, path[-1]-min_corner) < get_average_heat(domain, start-min_corner):
        next =  estimate_gradient(start, domain, min_corner, dim, radius, n_walks)
        if dist_to_obstacle(distance_field, min_corner_field, next) > 5.0:
            return domain, next, np.array([next])
        else:
            return domain, None, None
    
    next_step = path[1] - path[0]
    next_distance = np.linalg.norm(next_step)
    if next_distance > radius/10:
        path[0] = path[0] + (path[1] - path[0])/np.linalg.norm(path[1] - path[0])
        return domain, path[0], path
    else:
        return domain, path[1], path[1:]
    

def path_mapping(start, goal, distance_field, min_corner_field, direction=None, n_walks=1000):
    dim = len(start)
    dist_g = np.ceil(np.linalg.norm(start-goal)).astype(int)
    path = [start.copy()]
    krr = [start.copy()]
    heatmaps = []
    domain = np.zeros(shape=[1]*dim)
    min_corner = start
    min_corners = []
    domain_size = VIEW_RANGE * 2 + 1
    count = 0
    max_heat = 0

    while dist_g>distance_field[tuple(np.round(goal-min_corner_field).astype(int))]:
        print(start)
        count+=1
        new_min_corner = start - np.array([VIEW_RANGE]*dim)
        domain, min_corner = merge_domains_nd(domain, min_corner, new_min_corner, tuple([domain_size]*dim))

        solved_domain, start, krr = ego_centric_wos_mapping(krr, domain, min_corner, start, goal, distance_field, min_corner_field, direction, n_walks)
        if type(start) == type(None):
            start = path[-2]
            krr = [path[-2]]
        path.append(start.copy())
        dist_g = np.linalg.norm(start-goal)
        '''idx = np.argwhere(domain == -1)
        plt.scatter(idx[:, 1], idx[:, 0], c='w')
        temp = np.array(path)
        hi = np.array(krr)
        #plot_heatmap_3d(solved_domain, temp-min_corner)
        plt.imshow(solved_domain, cmap='inferno')
        plt.plot(*(hi-min_corner).T[::-1], 'w-')
        plt.plot(*(temp-min_corner).T[::-1], 'r-')
        plt.show(block=False)
        plt.pause(0.01)
        plt.close()'''
        
        heatmaps.append(solved_domain)
        min_corners.append(min_corner)

    #path.append(goal)

    return heatmaps, path, min_corners

start = np.array([750.0, 50.0])
goal = np.array([250.0, 950.0])
dim = len(start)
distance_field = load_ndarray('/Users/TWengChu/Desktop/wos_v2/obstacle_fields/forest_2d')
min_corner_field = np.array([0,0])
direction = np.array([0, -1])

heatmaps, paths, min_corners = path_mapping(start, goal, distance_field, min_corner_field, n_walks=10)
idx = (np.argwhere(distance_field <= 0) + min_corner_field - min_corners[-1]).astype(int) 
animate_2d(
    heatmaps,
    min_corners,
    paths,
    idx,
    fps=100,
    output_path = "kyak.mp4",)