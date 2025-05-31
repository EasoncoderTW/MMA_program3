from GA import GeneticAlgorithmTSP
from PSO import ParticleSwarmTSP
from ANT import AntColonyTSP
from dataset import read_tsp_files
import numpy as np
import pandas as pd

from scipy.spatial.distance import euclidean
def odd_calculate_distances(num_cities=None, coordinates=None):
    # 計算所有城市之間的距離，並存入矩陣中
    distances = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i % 2 == j % 2:  # 如果兩個城市都是奇數或都是偶數
                distances[i][j] = float('inf') # 奇數城市之間的距離設為無限大
            else:
                distances[i][j] = euclidean(coordinates[i], coordinates[j])  # 計算歐氏距離
    return distances


def execute_algorithms(coordinates, create_model, model_name, experiments):
    df = pd.DataFrame(columns=["Algorithm", "Cities", "Best Distance", "Best Fitness", "Average Distance", "Computation Time"])

    for cities_count in [12, 36, 52]:
        print(f"\nRunning algorithms for the first {cities_count} cities:")
        selected_coordinates = coordinates[:cities_count]

        # 初始化基因演算法並執行
        model = create_model(selected_coordinates)
        result = model.runs(experiments=experiments)  # 執行多次實驗以獲取平均值
        model.visualize_path(result["best_path"], title=f"{model_name} - {cities_count} Cities", savepath="../output", show=False)  # 可視化最佳路徑

        df.loc[len(df)] = {
            "Algorithm": model_name,
            "Cities": cities_count,
            "Best Distance": result["best_distance"],
            "Best Fitness": result["best_fitness"],
            "Average Distance": result["average_distance"],
            "Computation Time": result["compute_time"]
        }

    df.to_csv(f"../output/{model_name}_results.csv", index=False)

def create_ga_model(coordinates):
    return GeneticAlgorithmTSP(coordinates, population_size=50, generations=1000, mutation_rate=0.2)

def create_pso_model(coordinates):
    return ParticleSwarmTSP(coordinates, population_size=50, generations=1000, inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5)

def create_aco_model(coordinates):
    return AntColonyTSP(coordinates, num_ants=10, num_iterations=200, alpha=1, beta=3, evaporation_rate=0.5)

if __name__ == "__main__":
    # 從指定的資料夾讀取TSP文件並返回座標列表
    coordinates = list(read_tsp_files("../dataset").values())[0]  # 讀取第一個TSP文件的座標

    # Part I: Each team should execute at least two different metaheuristic algorithms and make the comparisons
    # by Best (shortest) Distance, Best fitness value, Average Distance, and Computation Time. Three cases
    # are examined: (i) the first 12 cities, (ii) the first 36 cities, and (iii) all 52 cities.

    models = {
        # "Genetic Algorithm": create_ga_model,
        # "Particle Swarm Optimization": create_pso_model,
        "Ant Colony Optimization": create_aco_model
    }
    for model_name, create_model in models.items():
        execute_algorithms(coordinates, create_model, model_name, experiments=10)
