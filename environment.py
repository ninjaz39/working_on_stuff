import numpy as np
from itertools import product
from completed_wos_mapping.GJK import dist_to_obstacle

class Domain:
    def __int__(self, range: np.ndarray, start: np.ndarray, obstacles: np.ndarray):
        self.range = range
        self.dim = len(start)
        self.start = start
        self.obstacles = obstacles
        domain_size = range*2 + 1
        self.heat_domain = np.zeros(shape=domain_size)
        self.distance_field = np.zeros(shape=domain_size)
        self.compute_distance_field()
        self.min_corner = start - range
        self.max_corner = start + range

    def expand_domain(self, next: np.ndarray):
        self.start = self.start + next
        min_corner_b = self.start - self.range

        min_corner   = np.minimum(self.min_corner, min_corner_b)
        self.max_corner   = np.maximum(self.max_corner, self.start + self.range)
        canvas_shape = tuple(np.round(self.max_corner - min_corner).astype(int).tolist())

        new_distance_field = np.full(canvas_shape, -1.0)
        new_heat_domain = np.zeros(canvas_shape)
        rel    = np.round(self.min_corner - min_corner).astype(int)
        slices_old = tuple(slice(int(rel[d]), int(rel[d]) + self.domain.shape[d]) for d in range(self.dim))
        new_distance_field[slices_old] = self.distance_field
        self.distance_field = new_distance_field
        new_indices = np.argwhere(new_distance_field == -1.0)
        self.min_corner = min_corner
        new_heat_domain[slices_old] = self.heat_domain
        self.heat_domain = new_heat_domain
        self.update_new_regions(new_indices)

    def compute_distance_field(self, idx = None):
        if type(idx) == type(None):
            canvas_shape = self.distance_field.shape
            idx = np.indices(canvas_shape).reshape(len(canvas_shape), -1).T

        for i in idx:
            self.distance_field[i] = dist_to_obstacle(i+self.min_corner, self.obstacles)

        return True


    def update_new_regions(new_regions):
        pass


    def update_obstacles(self, new_obstacles: np.ndarray):
        np.append(self.obstacles, new_obstacles)
        return True

    def update_domain(self, v: np.ndarray, val: float):
        self.update_neighbors(self.heat_domain, v, val)
        return True

    def dist_to_edge(self, v):
        dist_to_min = v
        dist_to_max = self.heat_domain.shape - 1 - v
        
        return np.min(np.minimum(dist_to_min, dist_to_max))

    def get_heat(self, v: np.ndarray):
        nodes = self.get_neighbors(self.heat_domain, v-self.min_corner)
        total = 0
        num_vals = 0
        for _, i in nodes:
            if i>0:
                total += i
                num_vals += 1
        if num_vals == 0:
            return 0
        return total / num_vals
    
    def closest_obstacle(self, v: np.ndarray):
        dist_b = float('inf')
        dist_o = self.get_neighbors(self.distance_field, v-self.min_corner)
        if len(dist_o)>0:
            for i in dist_o:
                if i[1] <= dist_b:
                    dist_b = i[1]
            return dist_b
        
        else:
            return None
        
    def get_neighbors(array: np.ndarray, N: np.ndarray, radius = 1):
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
    

    def update_neighbors(array: np.ndarray, N: np.ndarray, v: float,radius = 1):
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