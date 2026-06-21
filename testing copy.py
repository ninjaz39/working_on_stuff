import numpy as np
import matplotlib.pyplot as plt
from sampling import sample_sphere, sample_ball
from helper import merge_domains_nd, update_neighbors_1, get_neighbors, dist_to_edge, get_average_heat, dist_to_obstacle
from shapes import EgoSphere
from path import filter_paths, check_path, check_intercept, estimate_gradient
from animate import animate_2d, animate_3d
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm


VIEW_RANGE = 100
MOVEMENT_RANGE= 5
EPSILON = 0.4999999
AVOID = 0.0
def save_ndarray(array: np.ndarray, name: str) -> None:
    """
    Saves an n-dimensional numpy array to a file.

    File format:
      Line 1: number of dimensions (n)
      Line 2: shape of the array (space-separated)
      Remaining lines: flattened array data (space-separated)

    Args:
        array: The n-dimensional numpy array to save.
        name:  The output filename (e.g. "my_array.txt").
    """
    n = array.ndim
    shape = array.shape
    flat = array.flatten()

    with open(name, "w") as f:
        # Line 1: number of dimensions
        f.write(f"{n}\n")
        # Line 2: shape
        f.write(" ".join(str(s) for s in shape) + "\n")
        # Remaining: flattened data
        f.write(" ".join(str(v) for v in flat) + "\n")

def load_ndarray(name: str) -> np.ndarray:
    """Companion loader that reconstructs the array from the saved file."""
    with open(name, "r") as f:
        n     = int(f.readline().strip())
        shape = tuple(int(s) for s in f.readline().strip().split())
        flat  = list(map(float, f.readline().strip().split()))

    return np.array(flat).reshape(shape)

def wos_walk_altered_g(radius_g, dim, distance_field, min_corner_field, goal, start, num_domain, overall_domain, min_corner, curr_rec):
    if curr_rec == 990:
        return 0
    adjusted_start = start-min_corner
    dist_g = np.linalg.norm(start-goal)
    if dist_g < radius_g:
        point = (1/dist_g*100000000)
        update_neighbors_1(overall_domain, adjusted_start, point)
        update_neighbors_1(num_domain, adjusted_start, 1)
        return point
    
    dist_b = dist_to_edge(adjusted_start, overall_domain)
    obstacle_hit = False
    dist_o = dist_to_obstacle(distance_field, min_corner_field, start, 0.9999)

    if dist_o == None:
        return 0
    
    if dist_o < dist_b:
        dist_b = dist_o - AVOID
        obstacle_hit = True

    if obstacle_hit and dist_b <= 0:
        return None
    
    if (not obstacle_hit and dist_b<EPSILON) or dist_to_obstacle(overall_domain, min_corner, start)<0:

        point = (1/dist_g*100000000) 
        return point
    
    point = None
    while point == None:
        v = sample_sphere(dim, dist_b, start)
        point = wos_walk_altered_g(radius_g, dim, distance_field, min_corner_field, goal, v, num_domain, overall_domain, min_corner, curr_rec+1)

    update_neighbors_1(overall_domain, adjusted_start, point)
    update_neighbors_1(num_domain, adjusted_start, 1)

    if point <= 0:
        return point
    
    return point

def WoS_altered_g(max_heat, distance_field, min_corner_field, goal, start, radius, n_walks, num_domain, overall_domain, min_corner):
    dim = len(start)
    final_path = []
    radius_g = distance_field[tuple(np.round(goal-min_corner_field).astype(int))]/2
    for walk in range(n_walks):
        first_step = sample_ball(dim, 100, start)
        heat = wos_walk_altered_g(radius_g, dim, distance_field, min_corner_field, goal, first_step, num_domain, overall_domain, min_corner, 0)
    return overall_domain

def ego_centric_wos_mapping(max_heat, path, num_domain, domain, min_corner, start, goal, distance_field, min_corner_field, direction, n_walks=1000):
    radius = min(VIEW_RANGE-AVOID, max(0,dist_to_obstacle(distance_field, min_corner_field, start)-AVOID))
    if not radius:
        return domain, None, [], 0
    
    domain = WoS_altered_g(max_heat, distance_field, min_corner_field, goal, start, radius, n_walks, num_domain, domain, min_corner)
    
    return domain, start, np.array([start]), 0
    

def path_mapping(start, goal, distance_field, min_corner_field, direction=None, n_walks=1000):
    '''dim = len(start)
    krr = [start.copy()]
    min_corner = start
    domain_size = VIEW_RANGE * 2 + 1
    max_heat = 0
    min_corner = start - np.array([VIEW_RANGE]*dim)
    domain = np.zeros(shape=[domain_size]*dim)
    num_domain = np.zeros(shape=[domain_size]*dim)'''
    dim = len(start)
    krr = [start.copy()]
    domain = np.zeros(shape=[1]*dim)
    num_domain = np.zeros(shape=[1]*dim)
    min_corner = start
    domain_size = VIEW_RANGE * 2 + 1
    max_heat = 0
    new_min_corner = start - np.array([VIEW_RANGE]*dim)
    domain, min_corner = merge_domains_nd(domain, min_corner, new_min_corner, tuple([domain_size]*dim))
    num_domain, min_corner = merge_domains_nd(num_domain, min_corner, new_min_corner, tuple([domain_size]*dim))
    for i in tqdm(range(1000)):
        solved_domain, start_1, krr, max_heat = ego_centric_wos_mapping(max_heat, krr, num_domain, domain, min_corner, start, goal, distance_field, min_corner_field, direction, n_walks)
        if i % 1000 == 0:
            idx = np.argwhere(distance_field == -1) +min_corner_field-min_corner
            
            #plot_heatmap_3d(solved_domain, temp-min_corner)
            plt.imshow(solved_domain/num_domain, cmap='inferno')
            plt.scatter(idx[:, 1], idx[:, 0], c='w')
            plt.show()
    

    #path.append(goal)
    save_ndarray(domain/num_domain, 'ref_data_1')
    return 0

'''start = np.array([450, 443, 450.0])
min_corner_field = np.array([350, 400, 400])
goal = np.array([450, 550, 450.0])
dim = len(start)
distance_field = load_ndarray('/Users/TWengChu/Desktop/wos_v2/obstacles_3d')
'''
start = np.array([250.0, 450.0])
goal = np.array([250.0, 600.0])
dim = len(start)
distance_field = load_ndarray(r'C:\Users\chuen\Desktop\working_on_stuff\obstacle_fields\smaller_box_2d')
min_corner_field = np.array([0,200])

heatmaps = path_mapping(start, goal, distance_field, min_corner_field, n_walks=10)