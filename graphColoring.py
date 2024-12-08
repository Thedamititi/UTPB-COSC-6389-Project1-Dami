import tkinter as tk
import random
import math


NUM_COLORS = 4
POP_SIZE = 100
MAX_GENERATIONS = 500
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 5
UPDATE_DELAY = 100
NO_IMPROVEMENT_LIMIT = 50

COLOR_PALETTE = ["red", "green", "blue", "yellow"]


def generate_random_graph(num_nodes, num_edges):
    edges = set()
    nodes = list(range(num_nodes))
    random.shuffle(nodes)

    # Create a spanning tree
    for i in range(num_nodes - 1):
        edges.add(tuple(sorted((nodes[i], nodes[i + 1]))))

    # Add random edges until we reach desired count
    while len(edges) < num_edges:
        a = random.randint(0, num_nodes - 1)
        b = random.randint(0, num_nodes - 1)
        if a != b:
            edge = tuple(sorted([a, b]))
            edges.add(edge)

    return list(edges)


def initialize_population(pop_size, num_nodes, num_colors):
    population = []
    for _ in range(pop_size):
        individual = [random.randint(0, num_colors - 1) for _ in range(num_nodes)]
        population.append(individual)
    return population


def fitness(individual, edges):
    # Conflicts: edges whose endpoints share the same color
    conflicts = sum(1 for (u, v) in edges if individual[u] == individual[v])
    return -conflicts  # fewer conflicts = higher fitness


def tournament_selection(population, edges, tournament_size):
    contenders = random.sample(population, tournament_size)
    best = max(contenders, key=lambda ind: fitness(ind, edges))
    return best


def one_point_crossover(parent1, parent2):
    # One-point crossover: choose a point and swap tails
    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point] + parent2[point:]
    return child


def mutate(individual, num_colors, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.randint(0, num_colors - 1)
    return individual


def local_search(individual, edges, num_colors):
    """
    Attempt a small local search:
    If there's a conflict, try recoloring one conflicting node
    to reduce conflicts.
    """
    current_fitness = fitness(individual, edges)
    if current_fitness == 0:
        return individual  # Already optimal

    # Identify a conflicting edge
    for u, v in edges:
        if individual[u] == individual[v]:
            # Try to fix this by recoloring 'u'
            original_color = individual[u]
            best_color = original_color
            best_fit = current_fitness
            for c in range(num_colors):
                if c != original_color:
                    individual[u] = c
                    test_fit = fitness(individual, edges)
                    if test_fit > best_fit:
                        best_fit = test_fit
                        best_color = c
            individual[u] = best_color  # commit the best found color
            break

    return individual


def evolve_population(population, edges, num_colors, mutation_rate, tournament_size):
    new_population = []
    # Elitism
    best_ind = max(population, key=lambda ind: fitness(ind, edges))
    new_population.append(best_ind)

    while len(new_population) < len(population):
        parent1 = tournament_selection(population, edges, tournament_size)
        parent2 = tournament_selection(population, edges, tournament_size)
        child = one_point_crossover(parent1, parent2)
        child = mutate(child, num_colors, mutation_rate)
        new_population.append(child)

    return new_population


class GraphColoringApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Graph Coloring with Genetic Algorithms")

        # Top frame for controls
        self.top_frame = tk.Frame(self.master)
        self.top_frame.pack(side=tk.TOP, pady=10)

        # Input for number of nodes
        tk.Label(self.top_frame, text="Number of Nodes:").pack(side=tk.LEFT, padx=5)
        self.nodes_entry = tk.Entry(self.top_frame, width=5)
        self.nodes_entry.pack(side=tk.LEFT)
        self.nodes_entry.insert(0, "15")  # default

        # Input for number of edges
        tk.Label(self.top_frame, text="Number of Edges:").pack(side=tk.LEFT, padx=5)
        self.edges_entry = tk.Entry(self.top_frame, width=5)
        self.edges_entry.pack(side=tk.LEFT)
        self.edges_entry.insert(0, "30")  # default

        self.generate_button = tk.Button(
            self.top_frame, text="Generate Graph", command=self.generate_graph
        )
        self.generate_button.pack(side=tk.LEFT, padx=5)

        self.solve_button = tk.Button(
            self.top_frame, text="Solve", command=self.start_solving, state=tk.DISABLED
        )
        self.solve_button.pack(side=tk.LEFT, padx=5)

        # Canvas to draw the graph
        self.canvas = tk.Canvas(self.master, width=800, height=600, bg="white")
        self.canvas.pack(pady=20)

        # Variables to store state
        self.num_nodes = 15
        self.num_edges = 30
        self.edges = []
        self.population = []
        self.generation = 0
        self.running = False
        self.best_fitness_history = []
        self.generations_since_improvement = 0

        # Prepare node positions (in a circle)
        self.node_positions = []

    def generate_node_positions(self, n):
        """
        Arrange nodes in a circular layout for better visibility.
        """
        radius = 200
        center_x, center_y = 400, 300
        positions = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            positions.append((x, y))
        return positions

    def generate_graph(self):
        # Get user-specified number of nodes and edges
        try:
            self.num_nodes = int(self.nodes_entry.get())
        except ValueError:
            self.num_nodes = 15

        try:
            self.num_edges = int(self.edges_entry.get())
        except ValueError:
            self.num_edges = 30

        # Generate new graph
        self.edges = generate_random_graph(self.num_nodes, self.num_edges)

        # Reinitialize population and positions based on new number of nodes
        self.node_positions = self.generate_node_positions(self.num_nodes)
        self.population = initialize_population(POP_SIZE, self.num_nodes, NUM_COLORS)

        self.generation = 0
        self.running = False
        self.best_fitness_history = []
        self.generations_since_improvement = 0

        # Draw initial (random) solution
        best_ind = max(self.population, key=lambda ind: fitness(ind, self.edges))
        self.draw_graph(best_ind)

        # Enable the solve button now that we have a graph
        self.solve_button.config(state=tk.NORMAL)

    def start_solving(self):
        if not self.running:
            self.running = True
            self.generation = 0
            self.best_fitness_history = []
            self.generations_since_improvement = 0
            self.evolve_step()

    def draw_graph(self, individual):
        self.canvas.delete("all")

        # Draw edges
        for u, v in self.edges:
            x1, y1 = self.node_positions[u]
            x2, y2 = self.node_positions[v]
            # Highlight conflicts in red
            color = "black" if individual[u] != individual[v] else "red"
            self.canvas.create_line(x1, y1, x2, y2, width=2, fill=color)

        # Draw nodes
        for i, color_idx in enumerate(individual):
            x, y = self.node_positions[i]
            node_color = COLOR_PALETTE[color_idx]
            self.canvas.create_oval(
                x - 15, y - 15, x + 15, y + 15, fill=node_color, outline="black"
            )

        # Display generation and best fitness
        best_ind = max(self.population, key=lambda ind: fitness(ind, self.edges))
        best_fit = fitness(best_ind, self.edges)
        self.canvas.create_text(
            100,
            20,
            text=f"Generation: {self.generation}",
            font=("Arial", 14),
            fill="black",
        )
        self.canvas.create_text(
            300, 20, text=f"Best Fitness: {best_fit}", font=("Arial", 14), fill="black"
        )

        # If solved (no conflicts), show a message
        if best_fit == 0:
            self.canvas.create_text(
                500, 20, text="SOLVED! No conflicts.", font=("Arial", 14), fill="green"
            )

    def partial_population_restart(self):
        """
        If stuck, reinitialize half of the population randomly to increase diversity.
        Keep the best half and replace the worst half.
        """
        sorted_pop = sorted(
            self.population, key=lambda ind: fitness(ind, self.edges), reverse=True
        )
        half = len(self.population) // 2
        new_half = initialize_population(half, self.num_nodes, NUM_COLORS)
        self.population = sorted_pop[:half] + new_half

    def evolve_step(self):
        if self.generation < MAX_GENERATIONS and self.running:
            # Evolve population
            self.population = evolve_population(
                self.population, self.edges, NUM_COLORS, MUTATION_RATE, TOURNAMENT_SIZE
            )

            # Local search on the best individual
            best_ind = max(self.population, key=lambda ind: fitness(ind, self.edges))
            best_ind = local_search(best_ind, self.edges, NUM_COLORS)

            # Put best_ind back into population (replace worst)
            worst_index = min(
                range(len(self.population)),
                key=lambda i: fitness(self.population[i], self.edges),
            )
            self.population[worst_index] = best_ind

            current_best_fit = fitness(best_ind, self.edges)
            if not self.best_fitness_history or current_best_fit > max(
                self.best_fitness_history
            ):
                self.best_fitness_history.append(current_best_fit)
                self.generations_since_improvement = 0
            else:
                self.generations_since_improvement += 1

            # If stuck for too long, do a partial restart
            if self.generations_since_improvement > NO_IMPROVEMENT_LIMIT:
                self.partial_population_restart()
                self.generations_since_improvement = 0

            # Increment generation
            self.generation += 1

            # Update UI
            self.draw_graph(best_ind)

            # Continue if not solved
            if current_best_fit != 0:
                self.master.after(UPDATE_DELAY, self.evolve_step)
            else:
                # Solved early
                self.running = False
        else:
            # Done or max generations reached
            self.running = False


if __name__ == "__main__":
    root = tk.Tk()
    app = GraphColoringApp(root)
    root.mainloop()
