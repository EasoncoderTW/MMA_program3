import time
import matplotlib.pyplot as plt
from dataset import read_tsp_files
from scipy.spatial.distance import euclidean

import numpy as np
from tqdm import tqdm

from ANT import AntColonyOptimization
from GA import GeneticAlgorithm
from PSO import ParticleSwarmOptimization

class TravelingSalesman:
    def __init__(self, coordinates, population_size, generations):
        self.coordinates = coordinates
        self.population_size = population_size
        self.generations = generations

    def run_part_i(self):
        results = {}
        for num_cities in [12, 36, 52]:
            subset_coordinates = self.coordinates[:num_cities]
            print(f"Running for {num_cities} cities...")

            # PSO
            start_time = time.time()
            pso = ParticleSwarmOptimization(subset_coordinates, self.population_size, self.generations, inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5)
            best_individual, best_fitness = pso.run()
            pso_time = time.time() - start_time
            results[f"PSO_{num_cities}"] = (best_individual, best_fitness, pso_time)

            # GA
            start_time = time.time()
            ga = GeneticAlgorithm(subset_coordinates, population_size=self.population_size, generations=self.generations, mutation_rate=0.1, num_slices=4)
            best_individual, best_fitness = ga.run()
            ga_time = time.time() - start_time
            results[f"GA_{num_cities}"] = (best_individual, best_fitness, ga_time)

            # ACO
            start_time = time.time()
            aco = AntColonyOptimization(subset_coordinates, num_ants=self.population_size, num_iterations=self.generations, evaporation_rate=0.1, alpha=1, beta=2)
            best_path, best_distance = aco.run()
            aco_time = time.time() - start_time
            results[f"ACO_{num_cities}"] = (best_path, best_distance, aco_time)

        return results

    def run_part_ii(self):
        subset_coordinates = self.coordinates[:36]
        filtered_coordinates = [subset_coordinates[i] for i in range(len(subset_coordinates)) if i % 2 == 0]
        print("Running Part II with odd-numbered cities excluded...")

        # Modify fitness function for GA
        ga = GeneticAlgorithm(filtered_coordinates, population_size=self.population_size, generations=self.generations, mutation_rate=0.1, num_slices=4)
        best_individual, best_fitness = ga.run()
        print("Best Individual from GA (Part II):", best_individual)
        print("Best Fitness from GA (Part II):", best_fitness)

        return best_individual, best_fitness

    def visualize_path(self, coordinates, path, title):
        plt.figure(figsize=(10, 6))
        for i in range(len(path)):
            start = coordinates[path[i]]
            end = coordinates[path[(i + 1) % len(path)]]
            plt.plot([start[0], end[0]], [start[1], end[1]], 'bo-')
            plt.text(start[0], start[1], str(path[i]), fontsize=8, color='red')  # Mark index on graph
        plt.title(title)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.show()

if __name__ == "__main__":
    coordinates = list(read_tsp_files("../dataset").values())[0]  # 讀取第一個TSP文件的座標
    population_size = 100
    generations = 100

    tsp = TravelingSalesman(coordinates, population_size, generations)

    # Part I
    results_part_i = tsp.run_part_i()
    for key, value in results_part_i.items():
        print(f"{key}: Best Individual: {value[0]}, Best Fitness: {value[1]}, Computation Time: {value[2]} seconds")

    # Part II
    best_individual_part_ii, best_fitness_part_ii = tsp.run_part_ii()

    # Visualization
    tsp.visualize_path(coordinates[:12], results_part_i["ACO_12"][0], "Best Roundtrip Route (ACO, 12 Cities)")
    tsp.visualize_path(coordinates[:12], results_part_i["PSO_12"][0], "Best Roundtrip Route (GA, 12 Cities)")
    tsp.visualize_path(coordinates[:12], results_part_i["GA_12"][0], "Best Roundtrip Route (GA, 12 Cities)")
    tsp.visualize_path(coordinates[:36], best_individual_part_ii, "Best Roundtrip Route (GA, Part II)")
