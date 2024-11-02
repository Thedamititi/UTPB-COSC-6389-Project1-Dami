import math
import random
import tkinter as tk
from tkinter import *

num_cities = 35
city_scale = 5
road_width = 2
padding = 100

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, canvas, color='black'):
        canvas.create_oval(self.x - city_scale, self.y - city_scale,
                           self.x + city_scale, self.y + city_scale, fill=color)

class Edge:
    def __init__(self, a, b):
        self.city_a = a
        self.city_b = b
        self.length = math.hypot(a.x - b.x, a.y - b.y)

    def draw(self, canvas, color='grey', style=(2, 4)):
        canvas.create_line(self.city_a.x,
                           self.city_a.y,
                           self.city_b.x,
                           self.city_b.y,
                           fill=color,
                           width=road_width,
                           dash=style)

class UI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Traveling Salesman")
        self.option_add("*tearOff", FALSE)
        width, height = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry("%dx%d+0+0" % (width, height))
        self.state("zoomed")

        self.canvas = Canvas(self)
        self.canvas.place(x=0, y=0, width=width, height=height)
        self.w = width - padding
        self.h = height - padding * 2

        self.cities_list = []
        self.edges_list = []

        menu_bar = Menu(self)
        self['menu'] = menu_bar

        menu_TS = Menu(menu_bar)
        menu_bar.add_cascade(menu=menu_TS, label='Salesman', underline=0)

        menu_TS.add_command(label="Generate", command=self.generate, underline=0)
        menu_TS.add_command(label="Solve with 2-Opt", command=self.solve_two_opt, underline=0)
        menu_TS.add_command(label="Solve with Genetic Algorithm", command=self.solve_genetic, underline=0)
        menu_TS.add_command(label="Solve with PSO", command=self.solve_pso, underline=0)

        self.mainloop()

    def generate(self):
        self.cities_list = []
        self.edges_list = []
        self.generate_cities()
        self.generate_edges()
        self.draw_cities_and_edges()

    def generate_cities(self):
        for _ in range(num_cities):
            x = random.randint(padding, self.w)
            y = random.randint(padding, self.h)
            node = Node(x, y)
            self.cities_list.append(node)

    def generate_edges(self):
        self.edges_list = []
        for i in range(len(self.cities_list)):
            for j in range(i+1, len(self.cities_list)):
                edge = Edge(self.cities_list[i], self.cities_list[j])
                self.edges_list.append(edge)

    def draw_cities_and_edges(self):
        self.canvas.delete("all")
        for edge in self.edges_list:
            edge.draw(self.canvas, color='grey', style=(2, 4))
        for city in self.cities_list:
            city.draw(self.canvas, color='black')

    def solve_two_opt(self):
        if not self.cities_list:
            return
        self.initialize_tour()
        self.draw_tour()
        self.optimize_step()

    def initialize_tour(self):
        self.tour = list(range(len(self.cities_list)))
        random.shuffle(self.tour)
        self.best_distance = self.tour_length(self.tour)
        self.iter_i = 1
        self.iter_k = self.iter_i + 1

    def tour_length(self, tour):
        total_length = 0
        for i in range(len(tour)):
            a = self.cities_list[tour[i]]
            b = self.cities_list[tour[(i + 1) % len(tour)]]
            length = math.hypot(a.x - b.x, a.y - b.y)
            total_length += length
        return total_length

    def two_opt_swap(self, tour, i, k):
        new_tour = tour[:i] + tour[i:k + 1][::-1] + tour[k + 1:]
        return new_tour

    def optimize_step(self):
        if self.iter_i >= len(self.tour) - 1:
            print("2-Opt Optimization complete. Final distance: {:.2f}".format(self.best_distance))
            return
        i = self.iter_i
        k = self.iter_k
        new_tour = self.two_opt_swap(self.tour, i, k)
        new_distance = self.tour_length(new_tour)
        if new_distance < self.best_distance:
            self.tour = new_tour
            self.best_distance = new_distance
            print(f"New best distance: {self.best_distance:.2f}")
            self.iter_i = 1
            self.iter_k = self.iter_i + 1
            self.draw_tour()
            self.after(1, self.optimize_step)
            return
        self.iter_k += 1
        if self.iter_k >= len(self.tour):
            self.iter_i += 1
            self.iter_k = self.iter_i + 1
        self.after(1, self.optimize_step)

    def draw_tour(self):
        self.canvas.delete("all")
        # First draw all edges as dotted lines
        for edge in self.edges_list:
            edge.draw(self.canvas, color='grey', style=(2, 4))
        # Then draw the tour edges as solid red lines
        for i in range(len(self.tour)):
            a = self.cities_list[self.tour[i]]
            b = self.cities_list[self.tour[(i + 1) % len(self.tour)]]
            self.canvas.create_line(a.x, a.y, b.x, b.y, fill='red', width=road_width)
        for n in self.cities_list:
            n.draw(self.canvas, 'red')

    def solve_genetic(self):
        if not self.cities_list:
            return
        self.population_size = 100
        self.generations = 500
        self.mutation_rate = 0.01
        self.initialize_population()
        self.generation = 0
        self.best_individual = None
        self.best_distance = float('inf')
        self.evolve()

    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            individual = list(range(len(self.cities_list)))
            random.shuffle(individual)
            self.population.append(individual)

    def evolve(self):
        if self.generation >= self.generations:
            print("Genetic Algorithm complete. Final distance: {:.2f}".format(self.best_distance))
            self.tour = self.best_individual
            self.draw_tour()
            return
        self.generation += 1
        self.evaluate_population()
        self.selection()
        self.crossover()
        self.mutation()
        self.after(1, self.evolve)

    def evaluate_population(self):
        fitness_scores = []
        for individual in self.population:
            distance = self.tour_length(individual)
            fitness_scores.append((1 / distance, individual))
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_individual = individual.copy()
                print(f"Generation {self.generation}: New best distance: {self.best_distance:.2f}")
                self.tour = self.best_individual
                self.draw_tour()
        self.fitness_scores = fitness_scores

    def selection(self):
        # Roulette Wheel Selection
        self.population = [ind for _, ind in sorted(self.fitness_scores, reverse=True)]
        total_fitness = sum([score for score, _ in self.fitness_scores])
        probabilities = [score / total_fitness for score, _ in self.fitness_scores]
        cumulative_probabilities = []
        cumulative = 0
        for p in probabilities:
            cumulative += p
            cumulative_probabilities.append(cumulative)
        new_population = []
        for _ in range(self.population_size):
            r = random.random()
            for i, cumulative_probability in enumerate(cumulative_probabilities):
                if r <= cumulative_probability:
                    new_population.append(self.population[i])
                    break
        self.population = new_population

    def crossover(self):
        # Ordered Crossover
        new_population = []
        for i in range(0, self.population_size, 2):
            parent1 = self.population[i]
            if i+1 < self.population_size:
                parent2 = self.population[i+1]
            else:
                parent2 = self.population[0]
            child1, child2 = self.ordered_crossover(parent1, parent2)
            new_population.extend([child1, child2])
        self.population = new_population

    def ordered_crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child1 = [None]*size
        child1[start:end+1] = parent1[start:end+1]
        pointer = 0
        for i in range(size):
            if parent2[i] not in child1:
                while child1[pointer] is not None:
                    pointer +=1
                child1[pointer] = parent2[i]
        # Same for child2
        child2 = [None]*size
        child2[start:end+1] = parent2[start:end+1]
        pointer = 0
        for i in range(size):
            if parent1[i] not in child2:
                while child2[pointer] is not None:
                    pointer +=1
                child2[pointer] = parent1[i]
        return child1, child2

    def mutation(self):
        for individual in self.population:
            if random.random() < self.mutation_rate:
                i, j = random.sample(range(len(individual)), 2)
                individual[i], individual[j] = individual[j], individual[i]

    def solve_pso(self):
        if not self.cities_list:
            return
        self.swarm_size = 30
        self.iterations = 500
        self.initialize_swarm()
        self.iteration = 0
        self.best_swarm_distance = float('inf')
        self.pso_step()

    def initialize_swarm(self):
        self.swarm = []
        self.global_best_position = None
        self.global_best_distance = float('inf')
        for _ in range(self.swarm_size):
            position = list(range(len(self.cities_list)))
            random.shuffle(position)
            particle = {
                'position': position,
                'velocity': [],
                'best_position': position.copy(),
                'best_distance': self.tour_length(position)
            }
            if particle['best_distance'] < self.global_best_distance:
                self.global_best_distance = particle['best_distance']
                self.global_best_position = particle['best_position']
            self.swarm.append(particle)
        self.tour = self.global_best_position
        self.best_distance = self.global_best_distance
        self.draw_tour()

    def pso_step(self):
        if self.iteration >= self.iterations:
            print("PSO complete. Final distance: {:.2f}".format(self.global_best_distance))
            self.tour = self.global_best_position
            self.draw_tour()
            return
        self.iteration += 1
        for particle in self.swarm:
            self.update_velocity(particle)
            self.update_position(particle)
            current_distance = self.tour_length(particle['position'])
            if current_distance < particle['best_distance']:
                particle['best_distance'] = current_distance
                particle['best_position'] = particle['position'].copy()
            if current_distance < self.global_best_distance:
                self.global_best_distance = current_distance
                self.global_best_position = particle['position'].copy()
                self.tour = self.global_best_position
                self.best_distance = self.global_best_distance
                print(f"Iteration {self.iteration}: New best distance: {self.best_distance:.2f}")
                self.draw_tour()
        self.after(1, self.pso_step)

    def update_velocity(self, particle):
        w = 0.5  # Inertia weight
        c1 = 1   # Cognitive (particle)
        c2 = 2   # Social (swarm)
        velocity = particle['velocity']
        position = particle['position']
        new_velocity = []

        size = len(position)
        for _ in range(size):
            if random.random() < w:
                # Swap two cities in the position
                idx1, idx2 = random.sample(range(size), 2)
                swap = (idx1, idx2)
                new_velocity.append(swap)

        for _ in range(size):
            if random.random() < c1:
                # Swaps towards personal best
                idx = random.randint(0, size - 1)
                if position[idx] != particle['best_position'][idx]:
                    idx2 = position.index(particle['best_position'][idx])
                    swap = (idx, idx2)
                    new_velocity.append(swap)

        for _ in range(size):
            if random.random() < c2:
                # Swaps towards global best
                idx = random.randint(0, size - 1)
                if position[idx] != self.global_best_position[idx]:
                    idx2 = position.index(self.global_best_position[idx])
                    swap = (idx, idx2)
                    new_velocity.append(swap)

        particle['velocity'] = new_velocity

    def update_position(self, particle):
        position = particle['position']
        for swap in particle['velocity']:
            idx1, idx2 = swap
            position[idx1], position[idx2] = position[idx2], position[idx1]
        # Clear velocity after applying
        particle['velocity'] = []

    def draw_tour(self):
        self.canvas.delete("all")
        # First draw all edges as dotted lines
        for edge in self.edges_list:
            edge.draw(self.canvas, color='grey', style=(2, 4))
        # Then draw the tour edges as solid red lines
        for i in range(len(self.tour)):
            a = self.cities_list[self.tour[i]]
            b = self.cities_list[self.tour[(i + 1) % len(self.tour)]]
            self.canvas.create_line(a.x, a.y, b.x, b.y, fill='red', width=road_width)
        for n in self.cities_list:
            n.draw(self.canvas, 'red')

if __name__ == '__main__':
    UI()
