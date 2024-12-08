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

    def draw(self, canvas, color="black"):
        canvas.create_oval(
            self.x - city_scale,
            self.y - city_scale,
            self.x + city_scale,
            self.y + city_scale,
            fill=color,
        )


class Edge:
    def __init__(self, a, b):
        self.city_a = a
        self.city_b = b
        self.length = math.hypot(a.x - b.x, a.y - b.y)

    def draw(self, canvas, color="grey", style=(2, 4)):
        canvas.create_line(
            self.city_a.x,
            self.city_a.y,
            self.city_b.x,
            self.city_b.y,
            fill=color,
            width=road_width,
            dash=style,
        )


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

        # Status info label
        self.info_label = tk.Label(self, text="", font=("Helvetica", 14), bg="white")
        self.info_label.place(x=10, y=10)

        menu_bar = Menu(self)
        self["menu"] = menu_bar

        menu_TS = Menu(menu_bar)
        menu_bar.add_cascade(menu=menu_TS, label="Salesman", underline=0)

        menu_TS.add_command(label="Generate", command=self.generate, underline=0)
        menu_TS.add_command(
            label="Solve with 2-Opt", command=self.solve_two_opt, underline=0
        )
        menu_TS.add_command(
            label="Solve with Genetic Algorithm",
            command=self.solve_genetic,
            underline=0,
        )
        # Replace PSO with ACO
        menu_TS.add_command(label="Solve with ACO", command=self.solve_aco, underline=0)

        self.mainloop()

    def update_info(self, text):
        self.info_label.config(text=text)
        self.info_label.update_idletasks()

    def generate(self):
        self.cities_list = []
        self.edges_list = []
        self.generate_cities()
        self.generate_edges()
        self.draw_cities_and_edges()
        self.update_info("Generated new set of cities.")

    def generate_cities(self):
        for _ in range(num_cities):
            x = random.randint(padding, self.w)
            y = random.randint(padding, self.h)
            node = Node(x, y)
            self.cities_list.append(node)

    def generate_edges(self):
        self.edges_list = []
        for i in range(len(self.cities_list)):
            for j in range(i + 1, len(self.cities_list)):
                edge = Edge(self.cities_list[i], self.cities_list[j])
                self.edges_list.append(edge)

    def draw_cities_and_edges(self):
        self.canvas.delete("all")
        for edge in self.edges_list:
            edge.draw(self.canvas, color="grey", style=(2, 4))
        for city in self.cities_list:
            city.draw(self.canvas, color="black")

    def tour_length(self, tour):
        total_length = 0
        for i in range(len(tour)):
            a = self.cities_list[tour[i]]
            b = self.cities_list[tour[(i + 1) % len(tour)]]
            length = math.hypot(a.x - b.x, a.y - b.y)
            total_length += length
        return total_length

    def two_opt_swap(self, tour, i, k):
        new_tour = tour[:i] + tour[i : k + 1][::-1] + tour[k + 1 :]
        return new_tour

    # --------------------
    # 2-Opt Method
    # --------------------
    def solve_two_opt(self):
        if not self.cities_list:
            return
        self.update_info("Starting 2-Opt optimization...")
        self.initialize_tour()
        self.draw_tour()
        self.optimize_step()

    def initialize_tour(self):
        self.tour = list(range(len(self.cities_list)))
        random.shuffle(self.tour)
        self.best_distance = self.tour_length(self.tour)
        self.iter_i = 1
        self.iter_k = self.iter_i + 1
        self.update_info(
            f"2-Opt Iteration: {self.iter_i}, Best Distance: {self.best_distance:.2f}"
        )

    def optimize_step(self):
        if self.iter_i >= len(self.tour) - 1:
            self.update_info(
                f"2-Opt complete. Final Distance: {self.best_distance:.2f}"
            )
            print("2-Opt Optimization complete.")
            return
        i = self.iter_i
        k = self.iter_k
        new_tour = self.two_opt_swap(self.tour, i, k)
        new_distance = self.tour_length(new_tour)
        if new_distance < self.best_distance:
            self.tour = new_tour
            self.best_distance = new_distance
            print(f"New best distance: {self.best_distance:.2f}")
            self.update_info(
                f"2-Opt Iteration: {self.iter_i}, Best Distance: {self.best_distance:.2f}"
            )
            self.iter_i = 1
            self.iter_k = self.iter_i + 1
            self.draw_tour()
            self.after(1, self.optimize_step)
            return
        self.iter_k += 1
        if self.iter_k >= len(self.tour):
            self.iter_i += 1
            self.iter_k = self.iter_i + 1
        self.update_info(
            f"2-Opt Iteration: {self.iter_i}, Best Distance: {self.best_distance:.2f}"
        )
        self.after(1, self.optimize_step)

    def draw_tour(self):
        self.canvas.delete("all")
        # First draw all edges as dotted lines
        for edge in self.edges_list:
            edge.draw(self.canvas, color="grey", style=(2, 4))
        # Then draw the tour edges as solid red lines
        for i in range(len(self.tour)):
            a = self.cities_list[self.tour[i]]
            b = self.cities_list[self.tour[(i + 1) % len(self.tour)]]
            self.canvas.create_line(a.x, a.y, b.x, b.y, fill="red", width=road_width)
        for n in self.cities_list:
            n.draw(self.canvas, "red")

    # --------------------
    # Genetic Algorithm
    # --------------------
    def solve_genetic(self):
        if not self.cities_list:
            return
        self.population_size = 100
        self.generations = 500
        self.mutation_rate = 0.01
        self.update_info("Initializing Genetic Algorithm...")
        self.initialize_population()
        self.generation = 0
        self.best_individual = None
        self.best_distance = float("inf")
        self.evolve()

    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            individual = list(range(len(self.cities_list)))
            random.shuffle(individual)
            self.population.append(individual)

    def evolve(self):
        if self.generation >= self.generations:
            self.update_info(f"GA complete. Final Distance: {self.best_distance:.2f}")
            print("Genetic Algorithm complete.")
            self.tour = self.best_individual
            self.draw_tour()
            return
        self.generation += 1
        self.evaluate_population()
        self.selection()
        self.crossover()
        self.mutation()
        self.update_info(
            f"GA Generation: {self.generation}, Best Distance: {self.best_distance:.2f}"
        )
        self.after(1, self.evolve)

    def evaluate_population(self):
        fitness_scores = []
        for individual in self.population:
            distance = self.tour_length(individual)
            fitness_scores.append((1 / distance, individual))
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_individual = individual.copy()
                print(
                    f"Generation {self.generation}: New best distance: {self.best_distance:.2f}"
                )
                self.update_info(
                    f"GA Generation: {self.generation}, Best Distance: {self.best_distance:.2f}"
                )
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
            if i + 1 < self.population_size:
                parent2 = self.population[i + 1]
            else:
                parent2 = self.population[0]
            child1, child2 = self.ordered_crossover(parent1, parent2)
            new_population.extend([child1, child2])
        self.population = new_population

    def ordered_crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child1 = [None] * size
        child1[start : end + 1] = parent1[start : end + 1]
        pointer = 0
        for i in range(size):
            if parent2[i] not in child1:
                while child1[pointer] is not None:
                    pointer += 1
                child1[pointer] = parent2[i]

        child2 = [None] * size
        child2[start : end + 1] = parent2[start : end + 1]
        pointer = 0
        for i in range(size):
            if parent1[i] not in child2:
                while child2[pointer] is not None:
                    pointer += 1
                child2[pointer] = parent1[i]
        return child1, child2

    def mutation(self):
        for individual in self.population:
            if random.random() < self.mutation_rate:
                i, j = random.sample(range(len(individual)), 2)
                individual[i], individual[j] = individual[j], individual[i]

    # --------------------
    # Ant Colony Optimization (ACO)
    # --------------------
    def solve_aco(self):
        if not self.cities_list:
            return

        # ACO Parameters
        self.number_of_ants = 20
        self.alpha = 1.0  # pheromone importance
        self.beta = 5.0  # heuristic importance (1/distance)
        self.evaporation_rate = 0.5
        self.Q = 100.0
        self.iterations = 200
        self.iteration = 0

        # Construct distance matrix
        self.dist_matrix = [
            [0] * len(self.cities_list) for _ in range(len(self.cities_list))
        ]
        for i in range(len(self.cities_list)):
            for j in range(len(self.cities_list)):
                if i != j:
                    a = self.cities_list[i]
                    b = self.cities_list[j]
                    self.dist_matrix[i][j] = math.hypot(a.x - b.x, a.y - b.y)
                else:
                    self.dist_matrix[i][j] = float("inf")

        # Initialize pheromone matrix
        initial_pheromone = 1.0 / (len(self.cities_list) * self.average_distance())
        self.pheromone = [
            [initial_pheromone] * (len(self.cities_list))
            for _ in range(len(self.cities_list))
        ]

        self.best_distance = float("inf")
        self.best_tour = None

        self.update_info("Initializing ACO...")
        self.aco_step()

    def average_distance(self):
        # For initialization, let's compute average edge length as a heuristic
        total = 0
        count = 0
        for i in range(len(self.cities_list)):
            for j in range(i + 1, len(self.cities_list)):
                total += self.dist_matrix[i][j]
                count += 1
        return total / count

    def aco_step(self):
        if self.iteration >= self.iterations:
            self.update_info(f"ACO complete. Final Distance: {self.best_distance:.2f}")
            print("ACO complete.")
            self.tour = self.best_tour
            self.draw_tour()
            return

        self.iteration += 1
        # Construct solutions
        all_tours = []
        all_distances = []
        for _ in range(self.number_of_ants):
            tour = self.construct_solution()
            distance = self.tour_length(tour)
            all_tours.append(tour)
            all_distances.append(distance)
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_tour = tour
                print(
                    f"Iteration {self.iteration}: New best distance: {self.best_distance:.2f}"
                )
                self.update_info(
                    f"ACO Iteration: {self.iteration}, Best Distance: {self.best_distance:.2f}"
                )
                self.tour = self.best_tour
                self.draw_tour()

        # Update pheromones
        self.update_pheromones(all_tours, all_distances)

        self.update_info(
            f"ACO Iteration: {self.iteration}, Best Distance: {self.best_distance:.2f}"
        )
        self.after(1, self.aco_step)

    def construct_solution(self):
        # Ant's tour construction (greedy-probabilistic)
        n = len(self.cities_list)
        start = random.randint(0, n - 1)
        tour = [start]
        visited = set([start])

        for _ in range(n - 1):
            current = tour[-1]
            next_city = self.select_next_city(current, visited)
            tour.append(next_city)
            visited.add(next_city)
        return tour

    def select_next_city(self, current, visited):
        # Probability based on pheromone^alpha * (1/distance)^beta
        n = len(self.cities_list)
        probabilities = []
        total = 0.0
        for j in range(n):
            if j not in visited:
                tau = self.pheromone[current][j] ** self.alpha
                eta = (1.0 / self.dist_matrix[current][j]) ** self.beta
                val = tau * eta
                probabilities.append((j, val))
                total += val

        # Roulette wheel selection
        r = random.random() * total
        cumulative = 0.0
        for city, val in probabilities:
            cumulative += val
            if cumulative >= r:
                return city
        # Fallback (should never get here if everything is correct)
        return probabilities[-1][0]

    def update_pheromones(self, all_tours, all_distances):
        # Evaporate
        n = len(self.cities_list)
        for i in range(n):
            for j in range(n):
                self.pheromone[i][j] *= 1 - self.evaporation_rate

        # Deposit
        for tour, dist in zip(all_tours, all_distances):
            contribution = self.Q / dist
            for k in range(len(tour)):
                a = tour[k]
                b = tour[(k + 1) % len(tour)]
                self.pheromone[a][b] += contribution
                self.pheromone[b][a] += contribution


if __name__ == "__main__":
    UI()
