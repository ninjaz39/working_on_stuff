import numpy as np
from sampling import sample_sphere
def normalize(v):
    norm_v = np.linalg.norm(v)
    if np.linalg.norm(v) == 0:
        return v
    return v / np.linalg.norm(v)

class Sphere:
    def __init__(self, center, radius):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.dim = len(center)
    def support(self, d):
        d = normalize(d)
        return self.center + self.radius * d
    def max(self):
        return self.center + np.full(self.dim, self.radius)
    def min(self):
        return self.center - np.full(self.dim, self.radius)

class ConvexHull:
    def __init__(self, points):
        self.points = np.array(points)
        #self.min_sphere()
    
    def support(self, d):
        ans = self.points[0]@d
        index = 0
        for i in range(len(self.points)):
            if self.points[i]@d > ans:
                ans = self.points[i]@d
                index = i
        return np.array(self.points[index])
    
    def shift_center(self, offset):
        self.points = self.points - offset
        return self

    def copy(self):
        return ConvexHull(self.points)
        

class SphereAsConvex(ConvexHull):
    def __init__(self, center, radius, dim):
        points = [sample_sphere(dim, radius, center) for i in range(int(radius) * 10)]
        super().__init__(points)

class LaplaceEqn:
    def __init__(self, DIM, sigma, f, g, h = None):
        self.g = g
        self.f = f
        self.sigma = sigma
        self.h = h
        self.DIM = DIM

class EgoSphere(Sphere):
    def __init__(self, radius, center, goal):
        super().__init__(center, radius)
        self.radius_adjusted_center = center - np.array([radius for i in range(len(center))])
        self.goal = np.array(goal)


    def update_ball(self, center):
        self.center = center

    def g(self, radius):
        return (1/(radius))
    

def support_vec(shape1, shape2, d):
    """
    Support point of Minkowski difference A - B
    """
    return (shape1.support(d) - shape2.support(-d))