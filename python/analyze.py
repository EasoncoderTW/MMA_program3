from dataset import read_tsp_files
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm
import pandas as pd

# Read history of paths from a json file
def read_history(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

# Visualize a GIF from the history of paths
def generate_history_animation(history, coordinates, title, output_filename):
    from matplotlib.animation import FuncAnimation

    total_generations = len(history)

    history_prune = [history[0]]  # Start with the first entry in history
    for i in range(1,len(history)):
        same = True
        for j in range(len(history[i]['best_path'])):  # Updated to use 'history[i]'
            if history[i]['best_path'][j] != history[i-1]['best_path'][j]:  # Updated to use 'history[i]'
                same = False
                break
        if not same:
            history_prune.append(history[i])  # Updated to append 'history[i]'
    history = history_prune

    cities_count = len(history[0]['best_path'])
    coordinates = coordinates[:cities_count]  # Ensure coordinates match the number of cities in the path

    coordinates_x = [coord[0] for coord in coordinates]
    coordinates_y = [coord[1] for coord in coordinates]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(title)
    gapx = (max(coordinates_x) - min(coordinates_x)) * 0.1
    gapy = (max(coordinates_y) - min(coordinates_y)) * 0.1
    ax.set_xlim(min(coordinates_x) - gapx, max(coordinates_x) + gapx)
    ax.set_ylim(min(coordinates_y) - gapy, max(coordinates_y) + gapy)

    # Plot the cities
    ax.scatter(coordinates_x, coordinates_y, c='red', s=50)

    # Initialize the line that will be updated in the animation
    line, = ax.plot([], [], 'b-', lw=2)

    # text generation
    def generate_text(frame):
        best_distance = history[frame]['best_distance']
        generation = history[frame]['generation']
        return f"Generation: {generation}/{total_generations}\nBest Distance: {best_distance:.2f}"

    text = ax.text(0.05, 0.95, '', transform=ax.transAxes, ha='left', va='top', fontsize=12, color='black')

    def init():
        line.set_data([], [])
        text.set_text('')
        return line, text

    def update(frame):
        path = history[frame]['best_path']
        path_coordinates = [coordinates[i] for i in path] + [coordinates[path[0]]]  # Close the loop by adding the first city at the end
        path_coordinates_x = [coord[0] for coord in path_coordinates]
        path_coordinates_y = [coord[1] for coord in path_coordinates]
        line.set_data(path_coordinates_x, path_coordinates_y)
        text.set_text(generate_text(frame))
        return line, text

    ani = FuncAnimation(fig, update, frames=len(history), init_func=init, blit=True, repeat=False)
    ani.save(output_filename, writer="pillow", fps=2)
    plt.close(fig)


# plot fittness history
def plot_fitness_history(history, title, output_filename):
    generations = [entry['generation'] for entry in history]
    best_distances = [entry['best_distance'] for entry in history]

    plt.figure(figsize=(10, 5))
    plt.plot(generations, best_distances, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel('Generation')
    plt.ylabel('Best Distance')
    plt.grid()
    plt.savefig(output_filename)
    plt.close()

def analyze_tsp_history():
    # Load the TSP coordinates
    coordinates = list(read_tsp_files("../dataset").values())[0]  # Read the first TSP file's coordinates
    coordinates = [[c[0], c[1]] for c in coordinates]  # Convert to list of lists

    for filename in tqdm(os.listdir('../output/json')):
        history = read_history(os.path.join('../output/json', filename))
        plot_fitness_history(
            history,
            filename.replace('_best_history.json', '_cities'),
            os.path.join(
                '../output/figures',
                filename.replace('_best_history.json', '_distance_history.png')
            )
        )

        generate_history_animation(
            history,
            coordinates, filename.replace('_best_history.json', '_cities'),
            os.path.join(
                '../output/gif',
                filename.replace('_best_history.json', '_cities.gif')
            ))


def plot_statistics():
    part_1_csv = [f for f in os.listdir('../output/csv') if f.endswith('_part1_results.csv')]
    part_2_csv = [f for f in os.listdir('../output/csv') if f.endswith('_part2_results.csv')]

    part_1_df_list = [pd.read_csv(os.path.join('../output/csv', filename)) for filename in part_1_csv]
    part_2_df_list = [pd.read_csv(os.path.join('../output/csv', filename)) for filename in part_2_csv]

    part_1_df = pd.concat(part_1_df_list, ignore_index=True)
    part_2_df = pd.concat(part_2_df_list, ignore_index=True)

    # columns = [Algorithm,Cities,Best Distance,Best Fitness,Average Distance,Computation Time]
    # Plotting Part 1 Statistics
    # 1. Best Distance of 12,36,52 cities in different algorithms case
    # 2. Average Distance of 12,36,52 cities in different algorithms case
    # 3. Computation Time of 12,36,52 cities in different algorithms case

    metrics = ['Best Distance', 'Average Distance', 'Computation Time']
    filenames = ['best_distance_part1.png', 'average_distance_part1.png', 'computation_time_part1.png']
    y_labels = ['Best Distance', 'Average Distance', 'Computation Time (seconds)']
    titles = [
        'Best Distance for Different Algorithms and Cities',
        'Average Distance for Different Algorithms and Cities',
        'Computation Time for Different Algorithms and Cities'
    ]

    for metric, filename, y_label, title in zip(metrics, filenames, y_labels, titles):
        plt.figure(figsize=(12, 6))
        width = 0.25  # Width of each bar
        part_1_df = part_1_df.sort_values(by='Algorithm')  # Sort by Algorithm for better visibility
        algorithms = part_1_df['Algorithm'].unique()
        x = np.arange(len(algorithms))  # X positions for the bars

        colors = plt.get_cmap('tab10', len([52, 36, 12]))  # Use a colormap for better distinction

        for i, cities in enumerate([52, 36, 12]):  # inverse order for better visibility
            subset = part_1_df[part_1_df['Cities'] == cities]
            plt.bar(x + i * width, subset[metric], width=width, label=f'{metric} {cities} Cities', color=colors(i))

        plt.title(title)
        plt.xlabel('Algorithm')
        plt.ylabel(y_label)
        plt.xticks(x + width, algorithms, rotation=45)  # Adjust x-ticks to align with grouped bars
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'../output/figures/{filename}')
        plt.close()

    # Plotting Part 2 Statistics
    # 1. Best Distance of 36 cities in different algorithms case
    # 2. Average Distance of 36 cities in different algorithms case
    # 3. Computation Time of 36 cities in different algorithms case
    metrics = ['Best Distance', 'Average Distance', 'Computation Time']
    filenames = ['best_distance_part2.png', 'average_distance_part2.png', 'computation_time_part2.png']
    y_labels = ['Best Distance', 'Average Distance', 'Computation Time (seconds)']
    titles = [
        'Best Distance for Different Algorithms with 36 Cities',
        'Average Distance for Different Algorithms with 36 Cities',
        'Computation Time for Different Algorithms with 36 Cities'
    ]
    for metric, filename, y_label, title in zip(metrics, filenames, y_labels, titles):
        plt.figure(figsize=(12, 6))
        width = 0.25  # Width of each bar
        part_2_df = part_2_df.sort_values(by='Algorithm')  # Sort by Algorithm for better visibility
        algorithms = part_2_df['Algorithm'].unique()
        x = np.arange(len(algorithms))  # X positions for the bars

        colors = plt.get_cmap('tab10', len(part_2_df['Cities'].unique()))  # Use a colormap for better distinction

        for i, cities in enumerate(part_2_df['Cities'].unique()):  # Iterate through unique cities
            subset = part_2_df[part_2_df['Cities'] == cities]
            plt.bar(x + i * width, subset[metric], width=width, label=f'{metric} {cities} Cities', color=colors(i))

        plt.title(title)
        plt.xlabel('Algorithm')
        plt.ylabel(y_label)
        plt.xticks(x + width, algorithms, rotation=45)  # Adjust x-ticks to align with grouped bars
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'../output/figures/{filename}')
        plt.close()


if __name__ == "__main__":
    # analyze_tsp_history()
    plot_statistics()
    print("Analysis complete. Check the output figures for fitness history.")