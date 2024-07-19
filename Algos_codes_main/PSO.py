import numpy as np
# Particule Swarm Optimization


# Classe d'une particule mutli-objective

class Particle_MO:
    def __init__(self, dim, phi1, phi2, num_objectives):
        self.position = np.random.uniform(-5, 5, dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = self.position
        self.best_value = np.zeros(num_objectives)
        self.phi1 = phi1
        self.phi2 = phi2

    def update_velocity(self, global_best_position, inertia_weight):
        b1 = np.random.uniform(0, self.phi1)
        b2 = np.random.uniform(0, self.phi2)
        self.velocity = inertia_weight * self.velocity + b1 * (self.best_position - self.position) + b2 * (global_best_position - self.position)

    def update_position(self):
        self.position += self.velocity

        # Clamping position to avoid going out of bounds
        self.position = np.clip(self.position, -5, 5)


# Algorithme PSO multi-objectif

class MO_PSO:
    def __init__(self, objective_function, num_particles, dim, max_iterations, phi1, phi2, inertia_weight, num_objectives):
        self.objective_function = objective_function
        self.num_particles = num_particles
        self.dim = dim
        self.max_iterations = max_iterations
        self.phi1 = phi1
        self.phi2 = phi2
        self.inertia_weight = inertia_weight
        self.num_objectives = num_objectives
        self.particles = [Particle_MO(dim, phi1, phi2, num_objectives) for _ in range(num_particles)]
        self.pareto_front = self.initialize_pareto_front()

    def dominates(self, particle1, particle2):
        """Check if particle1 dominates particle2."""
        return all(particle1.best_value[i] <= particle2.best_value[i] for i in range(self.num_objectives)) \
               and any(particle1.best_value[i] < particle2.best_value[i] for i in range(self.num_objectives))

    def initialize_pareto_front(self):
        pareto_front = []
        for particle in self.particles:
            particle.best_value = self.objective_function(particle.position)
            pareto_front.append(particle)
        return pareto_front

    def optimize(self):
        for _ in range(self.max_iterations):
            for particle in self.particles:
                particle.update_velocity(self.get_global_best_position(), self.inertia_weight)
                particle.update_position()
                fitness = self.objective_function(particle.position)
                if all(fitness[i] < particle.best_value[i] for i in range(self.num_objectives)):
                    particle.best_value = fitness
                    particle.best_position = particle.position.copy()

            self.update_pareto_front()

        return [(particle.best_position, particle.best_value) for particle in self.pareto_front]

    def get_global_best_position(self):
        if not self.pareto_front:
            # Si le front de Pareto est vide, choisir aléatoirement une particule
            return np.random.uniform(-5, 5, self.dim)
        best_particle = min(self.pareto_front, key=lambda x: np.sum(x.best_value))
        return best_particle.best_position


    def update_pareto_front(self):
        updated_front = []
        for particle in self.particles:
            if all(self.dominates(particle, other) for other in self.pareto_front if other != particle):
                updated_front.append(particle)
        self.pareto_front = updated_front


# Classe particule pour mono-objectif

class Particle:
    def __init__(self, dim, phi1, phi2):
        self.position = np.random.uniform(-5, 5, dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = self.position
        self.best_value = float('inf')
        self.phi1 = phi1
        self.phi2 = phi2

    def update_velocity(self, global_best_position, inertia_weight):
        b1 = np.random.uniform(0, self.phi1)
        b2 = np.random.uniform(0, self.phi2)
        self.velocity = inertia_weight * self.velocity + b1 * (self.best_position - self.position) + b2 * (global_best_position - self.position)

    def update_position(self):
        self.position += self.velocity

        # Clamping position to avoid going out of bounds
        self.position = np.clip(self.position, -5, 5)


# Algorithme PSO mono-objectif

class PSO:
    def __init__(self, objective_function, num_particles, dim, max_iterations, phi1, phi2, inertia_weight):
        self.objective_function = objective_function
        self.num_particles = num_particles
        self.dim = dim
        self.max_iterations = max_iterations
        self.phi1 = phi1
        self.phi2 = phi2
        self.inertia_weight = inertia_weight
        self.particles = [Particle(dim, phi1, phi2) for _ in range(num_particles)]
        self.global_best_position = np.random.uniform(-5, 5, dim)
        self.global_best_value = float('inf')

    def optimize(self):
        for _ in range(self.max_iterations):
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.inertia_weight)
                particle.update_position()
                fitness = self.objective_function(particle.position)
                if fitness < particle.best_value:
                    particle.best_value = fitness
                    particle.best_position = particle.position.copy()
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = particle.position.copy()
        return self.global_best_position, self.global_best_value

