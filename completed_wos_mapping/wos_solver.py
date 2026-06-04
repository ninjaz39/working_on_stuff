import numpy as np
from sampling import sample_sphere, sample_ball

class WoSolver:
    def __init__(self, epsilon_shell, max_recursion, dim, n_walks):
        self.max_recursion = max_recursion
        self.epsilon_shell = epsilon_shell
        self.dim = dim
        self.n_walks = n_walks


    def solve_domain(self, domain, start, goal, radius, radius_g):
        max_heat = 0
        final_path = [start.copy()]
        for walk in range(self.n_walks):
            first_step = sample_ball(self.dim, radius, start)
            heat, path = self.walk(radius_g, domain, goal, first_step, 0)
            if len(path)>0:
                if heat > max_heat:
                    max_heat = heat
                    path.append(self.start)
                    final_path = path
                elif heat == max_heat:
                    if len(path) < len(final_path):
                        path.append(self.start)
                        final_path = path
        return final_path


    def walk(self, radius_g, domain, start, goal, curr_rec):
        if curr_rec == self.max_recursion:
            return 0, []

        dist_g = np.linalg.norm(start - goal)
        if dist_g < radius_g:
            point = 1/dist_g*100000000
            domain.update_heat(start, point)
            return point, [start]
        
        dist_b = domain.dist_to_edge(start)
        obstacle_hit = False
        dist_o = domain.closest_obstacle(start)
        if dist_o == None:
            return 0, []
        if dist_o < dist_b:
            dist_b = dist_o
            obstacle_hit = True

        if obstacle_hit and dist_b < 5:
            return None, []
        
        if dist_b<self.epsilon_shell:
            point = 1/dist_g*100000000
            domain.update_heat(start, point)
            return point, [start]
        
        point = None
        while point == None:
            v = sample_sphere(self.dim, dist_b, start)
            point, next_steps = self.walk(radius_g, domain, v, goal, curr_rec+1)

        domain.update_heat(start, point)
        if point == 0:
            return 0, []
        next_steps.append(start)
        return point, next_steps