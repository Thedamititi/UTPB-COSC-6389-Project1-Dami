import math
import random
import tkinter as tk
from tkinter import *
import threading

num_items = 100
frac_target = 0.7
min_value = 128
max_value = 2048

screen_padding = 25
item_padding = 5
stroke_width = 5

num_generations = 1000
pop_size = 50
elitism_count = 2
mutation_rate = 0.01  # mutation rate for per-bit mutation

sleep_time = 0.1


def random_rgb_color():
    red = random.randint(0x10, 0xFF)
    green = random.randint(0x10, 0xFF)
    blue = random.randint(0x10, 0xFF)
    hex_color = "#{:02x}{:02x}{:02x}".format(red, green, blue)
    return hex_color


class Item:
    def __init__(self):
        self.value = random.randint(min_value, max_value)
        self.color = random_rgb_color()
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0

    def place(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def draw(self, canvas, active=False):
        canvas.create_text(
            self.x + self.w + item_padding + stroke_width * 2,
            self.y + self.h / 2,
            text=f"{self.value}",
        )
        if active:
            canvas.create_rectangle(
                self.x,
                self.y,
                self.x + self.w,
                self.y + self.h,
                fill=self.color,
                outline=self.color,
                width=stroke_width,
            )
        else:
            canvas.create_rectangle(
                self.x,
                self.y,
                self.x + self.w,
                self.y + self.h,
                fill="",
                outline=self.color,
                width=stroke_width,
            )


class UI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Knapsack")
        self.option_add("*tearOff", FALSE)
        self.width, self.height = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry("%dx%d+0+0" % (self.width, self.height))
        self.state("zoomed")

        self.canvas = Canvas(self)
        self.canvas.place(x=0, y=0, width=self.width, height=self.height)

        self.items_list = []

        menu_bar = Menu(self)
        self["menu"] = menu_bar
        menu_K = Menu(menu_bar)
        menu_bar.add_cascade(menu=menu_K, label="Knapsack", underline=0)

        def generate():
            self.generate_knapsack()
            self.draw_items()

        menu_K.add_command(label="Generate", command=generate, underline=0)

        self.target = 0

        def set_target():
            target_set = []
            for _ in range(int(num_items * frac_target)):
                item = self.items_list[random.randint(0, len(self.items_list) - 1)]
                while item in target_set:
                    item = self.items_list[random.randint(0, len(self.items_list) - 1)]
                target_set.append(item)
            total = sum(item.value for item in target_set)
            self.target = total
            self.draw_target()

        menu_K.add_command(label="Get Target", command=set_target, underline=0)

        def start_thread():
            thread = threading.Thread(target=self.run, args=())
            thread.start()

        menu_K.add_command(label="Run", command=start_thread, underline=0)

        self.mainloop()

    def get_rand_item(self):
        i1 = Item()
        for i2 in self.items_list:
            if i1.value == i2.value:
                return None
        return i1

    def add_item(self):
        item = self.get_rand_item()
        while item is None:
            item = self.get_rand_item()
        self.items_list.append(item)

    def generate_knapsack(self):
        self.items_list = []
        unique_values = random.sample(range(min_value, max_value + 1), num_items)
        for val in unique_values:
            new_item = Item()
            new_item.value = val
            self.items_list.append(new_item)

        item_max = max(item.value for item in self.items_list)
        w = self.width - screen_padding
        h = self.height - screen_padding
        num_rows = math.ceil(num_items / 6)
        row_w = w / 8 - item_padding
        row_h = (h - 200) / num_rows

        for x in range(0, 6):
            for y in range(0, num_rows):
                if x * num_rows + y >= num_items:
                    break
                item = self.items_list[x * num_rows + y]
                item_w = row_w / 2
                item_h = max(item.value / item_max * row_h, 1)
                item.place(
                    screen_padding + x * row_w + x * item_padding,
                    screen_padding + y * row_h + y * item_padding,
                    item_w,
                    item_h,
                )

    def clear_canvas(self):
        self.canvas.delete("all")

    def draw_items(self):
        for item in self.items_list:
            item.draw(self.canvas)

    def draw_target(self):
        x = (self.width - screen_padding) / 8 * 7
        y = screen_padding
        w = (self.width - screen_padding) / 8 - screen_padding
        h = self.height / 2 - screen_padding
        self.canvas.create_rectangle(x, y, x + w, y + h, fill="black")
        self.canvas.create_text(
            x + w // 2,
            y + h + screen_padding,
            text=f"{self.target}",
            font=("Arial", 18),
        )

    def draw_sum(self, item_sum, target):
        x = (self.width - screen_padding) / 8 * 6
        y = screen_padding
        w = (self.width - screen_padding) / 8 - screen_padding
        h = self.height / 2 - screen_padding
        h *= (item_sum / target) if target > 0 else 0
        self.canvas.create_rectangle(x, y, x + w, y + h, fill="black")
        self.canvas.create_text(
            x + w // 2,
            y + h + screen_padding,
            text=f'{item_sum} ({"+" if item_sum > target else "-"}{abs(item_sum - target)})',
            font=("Arial", 18),
        )

    def draw_genome(self, genome, gen_num):
        for i in range(num_items):
            item = self.items_list[i]
            active = genome[i]
            item.draw(self.canvas, active)
        x = (self.width - screen_padding) / 8 * 6
        y = screen_padding
        w = (self.width - screen_padding) / 8 - screen_padding
        h = self.height / 4 * 3
        self.canvas.create_text(
            x + w,
            y + h + screen_padding * 2,
            text=f"Generation {gen_num}",
            font=("Arial", 18),
        )

    def run(self):
        global pop_size
        global num_generations

        def gene_sum(genome):
            return sum(
                item.value for gene, item in zip(genome, self.items_list) if gene
            )

        # Soft-penalty fitness function
        def fitness(genome):
            total = gene_sum(genome)
            if total <= self.target:
                return total
            else:
                # Soft penalty if over target
                penalty_factor = 0.5  # adjust as needed
                return max(self.target - (total - self.target) * penalty_factor, 0)

        def select_parents_tournament(population, k=3):
            def tournament():
                subset = random.sample(population, k)
                return max(subset, key=fitness)

            return tournament(), tournament()

        # Uniform crossover
        def crossover_uniform(parent1, parent2):
            child = []
            for g1, g2 in zip(parent1, parent2):
                child.append(g1 if random.random() < 0.5 else g2)
            return child

        # Per-bit mutation
        def mutate(genome):
            return [
                not gene if random.random() < mutation_rate else gene for gene in genome
            ]

        def get_population(last_pop=None):
            if last_pop is None:
                # Initialize random population
                population = []
                for _ in range(pop_size):
                    genome = [random.random() < frac_target for _ in range(num_items)]
                    population.append(genome)
                return population
            else:
                # Sort population by fitness
                sorted_pop = sorted(last_pop, key=fitness, reverse=True)

                # Elitism
                new_population = sorted_pop[:elitism_count]

                # Fill the rest using tournament selection, uniform crossover, and mutation
                while len(new_population) < pop_size:
                    parent1, parent2 = select_parents_tournament(last_pop, k=3)
                    child = crossover_uniform(parent1, parent2)
                    child = mutate(child)
                    new_population.append(child)
                return new_population

        def generation_step(generation=0, pop=None):
            if generation >= num_generations:
                return  # Stop after set number of generations

            if pop is None:
                pop = get_population()

            best_of_gen = max(pop, key=fitness)
            best_fitness = fitness(best_of_gen)

            print(f"Best fitness of generation {generation}: {best_fitness}")
            print(best_of_gen)
            print()

            # UI updates
            self.after(0, self.clear_canvas)
            self.after(0, self.draw_target)
            self.after(0, self.draw_sum, gene_sum(best_of_gen), self.target)
            self.after(0, self.draw_genome, best_of_gen, generation)

            # If we haven't achieved target, continue evolving
            if best_fitness < self.target:
                self.after(
                    int(sleep_time * 1000),
                    generation_step,
                    generation + 1,
                    get_population(pop),
                )
            else:
                print(f"Solution found at generation {generation}")

        # Start the evolutionary process
        generation_step()


if __name__ == "__main__":
    UI()
