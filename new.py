import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigs
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
from completed_wos_mapping.GJK import gjk_distance
from sampling import sample_sphere, sample_ball
from helper import dist_to_boundary, plot_solved_domain, divide_solved_domain,  merge_domains_nd, get_val, line_intersect
from shapes import Sphere, ConvexHull, support_vec, EgoSphere


VIEW_RANGE = 200
EPSILON = 1

def fill_shell_indices(i: int, x:np.ndarray, shape: tuple, R: float, epsilon: float, obstacles_mask:np.ndarray, boundary_mask:np.ndarray):
    """
    Returns the flat indices of nodes within [R-epsilon, R+epsilon] of x,
    and the normalized value (1/count). No array allocation needed.
    """
    if R < epsilon:
        if boundary_mask[i]:
            return [i], 1.0
        return [], 0.0

    ranges = [np.arange(s) for s in shape]
    grids = np.meshgrid(*ranges, indexing='ij')
    
    dist = np.sqrt(sum((grids[i] - x[i]) ** 2 for i in range(len(shape))))
    
    mask = (dist > R - epsilon) & (dist < R + epsilon) & ~obstacles_mask

    count = mask.sum()
    if count == 0:
        return [], 0.0

    flat_indices = np.flatnonzero(mask)
    return flat_indices, 1.0 / count


def build_shell_matrix_sparse(A: np.ndarray, epsilon: float, obstacles_mask:np.ndarray, obstacles_hit: np.ndarray, boundary_target: np.ndarray, boundary_mask: np.ndarray) -> csr_matrix:
    """
    Builds the shell matrix in sparse format. For a 301x301 array this
    is feasible where the dense version would require ~500GB of memory.
    """
    shape = A.shape
    num_nodes = A.size
    indices = list(np.ndindex(shape))
    num_received = [[] for _ in range(num_nodes)]

    # lil_matrix is efficient for row-by-row filling
    matrix = lil_matrix((num_nodes, num_nodes), dtype=np.float32)

    for i, idx in enumerate(indices):
        if i % 1000 == 0:
            print(f"  Building row {i}/{num_nodes} ({100*i/num_nodes:.1f}%)")

        x = np.array(idx, dtype=float)
        R = float(A[idx])
        flat_indices, value = fill_shell_indices(i, x, shape, R, epsilon, obstacles_mask, boundary_mask)

        if len(flat_indices) > 0:
            matrix[i, flat_indices] = value
            for fi in flat_indices:
                num_received[fi].append(i)


    for i in range(num_nodes):
        num_receive = len(num_received[i])
        if num_receive > 0 and boundary_mask[i]:
            val = 1/num_receive
            matrix[i, num_received[i]] = val
            matrix[i, i] = 1.0
                

    return matrix.tocsr()


def stationary_distribution(start: np.ndarray, goal: np.ndarray, A: np.ndarray, epsilon: float, obstacles_hit:np.ndarray) -> np.ndarray:
    """
    Builds the sparse shell matrix and computes the stationary distribution
    using a sparse eigensolver.
    """
    boundary_mask = (obstacles_hit < 0) & (A < epsilon)
    obstacles_mask = (obstacles_hit >= 0) & (A < epsilon) & (A >= 0)
    shape = A.shape
    indices = list(np.ndindex(shape))
    boundary_target = np.ones(shape=shape)
    final_tagret = np.zeros(shape=shape)
    for i, idx in enumerate(indices):
        if boundary_mask[idx]:
            boundary_target[idx] = 1/(np.linalg.norm(goal - np.array(idx) - start))
            final_tagret[idx] = 1/(np.linalg.norm(goal - np.array(idx) - start))

    print("Building sparse shell matrix...")
    matrix = build_shell_matrix_sparse(A, epsilon, obstacles_mask, obstacles_hit, boundary_target.flatten(), boundary_mask.flatten()).T

    print("Computing stationary distribution...")
    # Sparse eigensolver — only computes the eigenvector at eigenvalue 1
    '''v = final_tagret.flatten()

    for _ in tqdm(range(1000)):
        v = v@matrix'''

    eigenvalues, eigenvectors = eigs(matrix.T, k=1, which='LM')

    m = eigenvectors[:, 0].real
    #m = m / m.sum()
    print(m.shape)

    return m
def solve_domain_dist_to_b(domain, boundary_type, domain_shape, obstacles, boundary, dim, curr):
    if len(curr) == dim:
        curr = np.array(curr)
        '''if np.linalg.norm(curr+boundary.radius_adjusted_center-boundary.center)>boundary.radius:
            return '''
        val, obstacle_hit = dist_to_boundary(curr+boundary.radius_adjusted_center, obstacles, boundary)
        plot_solved_domain(curr, domain, val)
        plot_solved_domain(curr, boundary_type, obstacle_hit)
        return
    
    for i in range(domain_shape[0]):
        next = curr.copy()
        next.append(i)
        solve_domain_dist_to_b(domain, boundary_type, domain_shape[1:], obstacles, boundary, dim, next)
    return
start = np.array([400, 600, 0])
dim = len(start)
#obstacles =[Sphere(center=[500, 500], radius=50.0)]#, Sphere(center=[460, 550], radius=5.0)]#, Sphere(center=[450, 450], radius=5.0), Sphere(center=[500, 480], radius=5.0)]
cube_3d = ConvexHull([[450,450,450],[550,550,550],[450,550,550],[450,450,550], [550,450,550],[550,550,450],[550,450,450],[450,550,450]])
slit_2d = [ConvexHull([[0,500],[300,500],[0,550],[300,550]]), ConvexHull([[750,500],[350,500],[750,550],[350,550]])]
box_2d = [ConvexHull([[200,400],[200,500]]), ConvexHull([[300,400],[300,500]]), ConvexHull([[300,500],[200,500]])]
forest = []
line = ConvexHull([[500,500], [600,500]])
offsets = []
for i in range(30):
    tree = line.copy()
    offset = np.array([np.round(np.random.uniform(-450, 450)).astype(int), np.round(np.random.uniform(-450, 450)).astype(int)])
    offsets.append(offset)
    tree.shift_center(offset)
    forest.append(tree)

n_walks = 100000
epsilon = 1


pinhole = [ConvexHull([[0,0,450],[1000,0,450],[1000,500,450],[0,500,450],[0,0,550],[1000,0,550],[1000,500,550],[0,500,550]]),
           ConvexHull([[0,550,450],[1000,550,450],[1000,1000,450],[0,1000,450],[0,550,550],[1000,550,550],[1000,1000,550],[0,1000,550]]),
           ConvexHull([[0,500,450],[500,500,450],[500,550,450],[0,550,450],[0,500,550],[500,500,550],[500,550,550],[0,550,550]]),
           ConvexHull([[550,500,450],[1000,500,450],[1000,550,450],[550,550,450],[550,500,550],[1000,500,550],[1000,550,550],[550,550,550]])
]

obstacles = slit_2d
start = np.array([270, 480])
goal = np.array([270, 800])
ego_sphere = EgoSphere(VIEW_RANGE, start, goal)
dim = len(start)
shape = [np.ceil(2*VIEW_RANGE+1).astype(int) for i in range(dim)]
solved_distance_domain = np.zeros(shape=shape)
solved_obstacle_hits = np.zeros(shape=shape)
solve_domain_dist_to_b(solved_distance_domain, solved_obstacle_hits, shape, obstacles, ego_sphere, dim, [])

domain_stationary =  stationary_distribution(start-np.array([VIEW_RANGE]*dim), goal, solved_distance_domain, 0.5, solved_obstacle_hits)
plt.imshow(abs(domain_stationary.reshape(solved_distance_domain.shape)), cmap='inferno')
plt.show()