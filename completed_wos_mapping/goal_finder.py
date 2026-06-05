import numpy as np
from completed_wos_mapping.wos_solver import WoSolver
from environment import Domain
from sampling import sample_ball

class WalkOnGoal:
    def __init__(self, obstacles:np.ndarray, start:np.ndarray, goal:np.ndarray, range:int, way_point_acceptance_radius: int, avoidance_distance: int):
        self.start = start
        self.goal = goal
        self.path_walked = [start.copy()]
        self.forward_path = [start.copy()]
        self.radius_g = way_point_acceptance_radius
        self.domain = Domain(range, start, obstacles)
        self.n_walks = 10
        self.solver = WoSolver(0.5, 1000, avoidance_distance, self.dim, self.n_walks)
        self.avoidance_distance = avoidance_distance
    


    def add_obstacle(self, new_obstacles:np.ndarray):
        self.domain.update_obstacles(new_obstacles)



    def walk_to_goal(self):
        dist_g = np.ceil(np.linalg.norm(self.start-self.goal)).astype(int)

        while dist_g>self.radius_g:
            radius = min(self.range -5, max(0, self.distance_field.closest_obstacle(self.start) - 5))
            if radius == 0:
                self.start = self.path_walked[-2]
                self.forward_path = [start.copy()]
            else:
                unfiltered_path = self.solver.solve_domain(self.domain, self.start, self.goal, radius, self.radius_g)
                self.start = self.compute_path(unfiltered_path, radius)
                 self.domain.expand_domain(self.start)

            self.path_walked.append(self.start.copy())
            dist_g = np.ceil(np.linalg.norm(self.start-self.goal)).astype(int)

        return True



    def compute_path(self, unfiltered_path, radius):
        if len(unfiltered_path)>1 and self.check_path(unfiltered_path):
                unfiltered_path = self.filter_paths(unfiltered_path)
                self.forward_path = unfiltered_path
        if len(self.forward_path) < 2:
            next =  compute_gradient_ascent(self, radius)
            if self.domain.closest_obstacle(next) > 5:
                self.forward_path = np.array([next])
                return next
            else:
                return start
        next_step = self.forward_path[1] - self.forward_path[0]
        next_distance = np.linalg.norm(next_step)
        if next_distance > radius:
            path[0] = self.forward_path[0] + next_step/next_distance * radius
            return self.forward_path[0]
        else:
            self.forward_path = self.forward_path[1:]
            return self.forward_path[1]



    def compute_gradient_ascent(self, radius, trim_bottom=0.0):
        sampled_points = []
        sampled_vals = []
        for i in range(self.n_walks):
            sample = sample_ball(self.dim, radius, self.start)
            heat_vals = 0
            num_vals = 0
            for i in range(10):
                heat_vals = self.domain.get_heat(sample_ball(self.dim, radius, sample))

            if heat_vals>0:   
                sampled_points.append(sample)
                sampled_vals.append(heat_vals)

        if len(sampled_points)==0:
            return start
        
        sampled_points = np.asarray(points, dtype=float)
        sampled_vals = np.asarray(values, dtype=float)

        # Remove bottom X% of values
        if trim_bottom > 0:
            threshold = np.quantile(values, trim_bottom)
            mask = values >= threshold
            points = points[mask]
            values = values[mask]

        A = points - start

        A_aug = np.hstack([A, np.ones((len(A), 1))])
        result = np.linalg.lstsq(A_aug, values, rcond=None)[0]
        gradient = result[:-1]

        return start + gradient / np.linalg.norm(gradient) * radius/10


    def check_path(self, new_path):
        curr_heat = self.domain.get_heat(self.forward_path[-1])
        new_heat = self.domain.get_heat(new_path[-1])
        return curr_heat <= new_heat


    def filter_paths(self, path):
        final_path = [path[-1]]
        final_final_path = [path[-1]]
        curr_heat = 0
        checking = len(path) - 1
        total_steps = len(path) - 3
        while checking>0:
            checking -= 1
            if curr_heat != -1 and self.check_intercept(final_path[-1], path[checking]):
                final_path.append(path[checking + 1])
 
            
        final_direction = final_path[-1] - path[0]
        final_direction_length = np.linalg.norm(final_direction)
        if final_direction_length >= 5:
            final_direction /= final_direction_length
            final_path.append(path[0] + final_direction*5)
        if len(final_path) > 2:
            for i in range(len(final_path)-1, 0, -1):
                if not self.check_intercept(final_path[0], final_path[i]):
                    break

            final_final_path = np.append(np.array([final_path[0]]), final_path[i:], axis=0)
            return final_final_path
        

        return final_path

    def check_intercept(self, start, end):
        direction = end - start
        distance = np.linalg.norm(direction)
        if not distance:
            return False
        direction /= distance
        next = start.copy()
        while distance>0:
            dist_o = self.domain.closest_obstacle(next)
            if dist_o <= self.avoidance_distance:
                return True

            next += direction * dist_o
            distance -= dist_o
        return False