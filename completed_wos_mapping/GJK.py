import numpy as np
from shapes import Sphere

def support_vec(shape1, shape2, d):
    """
    Support point of Minkowski difference A - B
    """
    return (shape1.support(d) - shape2.support(-d))

def del_y(simplex):
    ans = np.zeros(len(simplex))
    
    for index in range(len(simplex)):
        ans[index] = del_yj(simplex, index)
    return ans

def del_yj(simplex, index_j):
    if len(simplex) == 1:
        return 1
    
    ans = 0
    vj = simplex[index_j]
    simplex_i =  simplex.copy()
    simplex_i.pop(index_j)
    k = simplex_i[0]
    for i, j in enumerate(simplex_i):
        del_yi = del_yj(simplex_i, i)
        ans += del_yi * (j@(k-vj))
    return ans

def nearest_simplex(simplex):
    if len(simplex) == 1:
        return simplex, None
    
    inside = True
    del_simplex = del_y(simplex)
    for i in del_simplex:
        if i<=0:
            inside = False
            break
    if inside:
        return simplex, del_simplex

    for i in range(len(simplex)-1):
        new_simplex = simplex.copy()
        new_simplex.pop(i)
        new_simplex, del_simplex = nearest_simplex(new_simplex)
        if type(del_simplex) != type(None):
            return new_simplex, del_simplex
    
    return [simplex[-1]], None

def distance_subalgorithm(simplex):
    new_simplex, del_simplex = nearest_simplex(simplex)
    if type(del_simplex) == type(None):
        return new_simplex, -1 * new_simplex[0], np.linalg.norm(new_simplex[0])
    
    v = np.zeros(new_simplex[0].shape)
    for i in range(len(new_simplex)):
        v -= new_simplex[i]*del_simplex[i]
    
    del_simplex_val = sum(del_simplex)
    v /= del_simplex_val
    return new_simplex, v, np.linalg.norm(v)

def gjk_distance(shape1, shape2, simplex, distPrev = None, alg = True):#alg = True means it uses DS, False uses BP
    """Calculate the distance between two convex shapes using the GJK algorithm."""
    new_simplex, v, dist = distance_subalgorithm(simplex)
    support = support_vec(shape1, shape2, v)
    #print(simplex, "simplex", support, 'support', v, 'v', distPrev, dist, 'dist', gk(-v, support))

    if gk(v, support) == 0:
        return dist#, v, new_simplex, True

    if (not distPrev) or dist < distPrev:
        new_simplex.append(support)
        return gjk_distance(shape1, shape2, new_simplex, dist)
    
    if not alg: #backup procedure used and tolerance not satisfied
        return dist#, v, new_simplex, alg #backup procedure used and tolerance not satisfied
    
    return gjk_distance(shape1, shape2, simplex, distPrev, False)
    
def gk(v, s):
    return np.linalg.norm(v)**2 + s@v

def dist_to_obstacle(x, obstacles):
    dist = float('inf')
    start_d = np.random.uniform(0,1,len(x))
    x = Sphere(x, 0)
    for index, obstacle in enumerate(obstacles):
        dist_o = gjk_distance(x, obstacle, [support_vec(x, obstacle, start_d)])
        if dist > dist_o:
            dist = dist_o
    return dist
