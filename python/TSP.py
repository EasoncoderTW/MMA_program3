import time
import matplotlib.pyplot as plt
from dataset import read_tsp_files
from scipy.spatial.distance import euclidean

import numpy as np
from tqdm import tqdm

def default_calculate_distances(num_cities=None, coordinates=None):
    # 計算所有城市之間的距離，並存入矩陣中
    distances = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            distances[i][j] = euclidean(coordinates[i], coordinates[j])
    return distances


# Abstract class for Traveling Salesman Problem
class TravelingSalesmanProblem:
    def __init__(self, coordinates, population_size, generations, calculate_distances=None):
        self.coordinates = coordinates
        self.distances = np.zeros((len(coordinates), len(coordinates)))
        self.population_size = population_size
        self.generations = generations
        self.calculate_distances = calculate_distances if calculate_distances is not None else default_calculate_distances
        self.num_cities = len(coordinates)  # Number of cities
        self.distances = self.calculate_distances(self.num_cities, coordinates)  # Calculate distances between cities
        self.path_history = []  # To store the history of paths

    def distance(self, path):
        # Calculate the total distance of the path
        total_distance = 0
        for i in range(len(path)):
            total_distance += self.distances[path[i]][path[(i + 1) % len(path)]]
        return total_distance

    def fitness_function(self, path):
        # Fitness function: inverse of the total distance
        total_distance = self.distance(path)
        if total_distance == 0:
            return float('inf')  # Avoid division by zero
        if total_distance == float('inf'):
            return 0
        return 1 / total_distance

    def run(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def runs(self, experiments=100):
        best_distance = float('inf')
        best_path = []
        best_fitness = 0
        average_distance = 0
        compute_time = 0

        best_history = []
        for run in tqdm(range(experiments), desc="Experiments"):
            start_time = time.time()
            current_best_path = self.run()  # Run the algorithm for one iteration
            end_time = time.time()

            # Assuming the best path and distance are obtained from the last iteration
            current_distance = self.distance(current_best_path)
            average_distance += current_distance
            compute_time += (end_time - start_time)

            if current_distance < best_distance:
                best_distance = current_distance
                best_path = current_best_path.copy()
                best_fitness = self.fitness_function(best_path)
                best_history = self.path_history.copy()  # Store the best path history

        average_distance /= experiments
        compute_time /= experiments

        return {
            "best_path": best_path,
            "best_distance": best_distance,
            "average_distance": average_distance,
            "compute_time": compute_time,
            "best_fitness": best_fitness,
            "best_history": best_history,
        }

    def visualize_path(self, path, title, savepath=None, show=False):
        plt.figure(figsize=(10, 6))
        dot_color = ['blue' if i % 2 == 0 else 'orange' for i in range(len(self.coordinates))]
        plt.scatter([c[0] for c in self.coordinates], [c[1] for c in self.coordinates], c=dot_color, marker='o', label='Cities')
        for i in range(len(path)):
            start = self.coordinates[path[i]]
            end = self.coordinates[path[(i + 1) % len(path)]]
            color = 'ro-' if path[i] % 2 == 0 else 'go-'
            plt.plot([start[0], end[0]], [start[1], end[1]], color)
            plt.text(start[0], start[1], str(path[i]), fontsize=10, color='red' if path[i] % 2 == 0 else 'green')  # Mark index on graph
        plt.title(title)
        plt.grid(True)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")

        if savepath:
            plt.savefig(f"{savepath}/{title}.png")

        if show:
            plt.show()

        plt.close()
