from dataset import read_tsp_files
import numpy as np
import matplotlib.pyplot as plt
import json

# Read history of paths from a json file
def read_history(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

# Visualize a GIF from the history of paths
def generate_history_animation(history, coordinates, output_filename):
    pass


# plot fittness history
def plot_fitness_history(history, output_filename):
    generations = [entry['generation'] for entry in history]
    best_distances = [entry['best_distance'] for entry in history]

    plt.figure(figsize=(10, 5))
    plt.plot(generations, best_distances, marker='o', linestyle='-', color='b')
    plt.title('Fitness History')
    plt.xlabel('Generation')
    plt.ylabel('Best Distance')
    plt.grid()
    plt.savefig(output_filename)
    plt.close()
