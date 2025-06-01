from GA import GeneticAlgorithmTSP
from PSO import ParticleSwarmTSP
from ANT import AntColonyTSP
from dataset import read_tsp_files
import numpy as np
import pandas as pd
import json

from scipy.spatial.distance import euclidean
import random

SEED = 42  # For reproducibility
EXPERIMENTS = 10  # Number of experiments to run for each algorithm

random.seed(SEED)  # For reproducibility
np.random.seed(SEED)  # For reproducibility

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


def execute_algorithms(coordinates, create_model, model_name, experiments, cities_counts = [12, 36, 52]):
    df = pd.DataFrame(columns=["Algorithm", "Cities", "Best Distance", "Best Fitness", "Average Distance", "Computation Time"])

    for cities_count in cities_counts:
        print(f"\nRunning algorithms for the first {cities_count} cities:")
        selected_coordinates = coordinates[:cities_count]

        # 初始化基因演算法並執行
        model = create_model(selected_coordinates)
        result = model.runs(experiments=experiments)  # 執行多次實驗以獲取平均值
        model.visualize_path(result["best_path"], title=f"{model_name}_{cities_count}_cities", savepath="../output/images", show=False)  # 可視化最佳路徑

        df.loc[len(df)] = {
            "Algorithm": model_name,
            "Cities": cities_count,
            "Best Distance": result["best_distance"],
            "Best Fitness": result["best_fitness"],
            "Average Distance": result["average_distance"],
            "Computation Time": result["compute_time"],
        }

        # best_history to JSON
        with open(f"../output/json/{model_name}_{cities_count}_best_history.json", "w") as json_file:
            history = [
                {
                    "generation": i,
                    "best_distance": int(dist) if dist != float('inf') else None,
                    "best_path": [int(p) for p in path]
                } for i, (dist, path) in enumerate(model.path_history)
            ]
            json.dump(history, json_file)
            json_file.write("\n")

    df.to_csv(f"../output/csv/{model_name}_results.csv", index=False)

def create_ga_model(population_size=50, generations=1000, mutation_rate=0.2, calculate_distances=None):
    return lambda coordinates: GeneticAlgorithmTSP(coordinates, population_size, generations, mutation_rate, calculate_distances)

def create_pso_model(population_size=10000, generations=100, inertia_weight=1, cognitive_weight=2, social_weight=3, calculate_distances=None):
    return lambda coordinates: ParticleSwarmTSP(coordinates, population_size, generations, inertia_weight, cognitive_weight, social_weight, calculate_distances)

def create_aco_model(num_ants=10, num_iterations=200, alpha=1, beta=3, evaporation_rate=0.5, calculate_distances=None):
    return lambda coordinates: AntColonyTSP(coordinates, num_ants, num_iterations, alpha, beta, evaporation_rate, calculate_distances)

def part_1():
    # 從指定的資料夾讀取TSP文件並返回座標列表
    coordinates = list(read_tsp_files("../dataset").values())[0]  # 讀取第一個TSP文件的座標

    # Part I: Each team should execute at least two different metaheuristic algorithms and make the comparisons
    # by Best (shortest) Distance, Best fitness value, Average Distance, and Computation Time. Three cases
    # are examined: (i) the first 12 cities, (ii) the first 36 cities, and (iii) all 52 cities.

    models = {
        "Genetic_Algorithm": {
            0:create_ga_model(population_size=50, generations=1000, mutation_rate=0.2),
            1:create_ga_model(population_size=100, generations=500, mutation_rate=0.1)
        },
        "Particle_Swarm": {
            0:create_pso_model(population_size=10000, generations=100, inertia_weight=1, cognitive_weight=2, social_weight=3),
            1:create_pso_model(population_size=5000, generations=200, inertia_weight=0.5, cognitive_weight=1.5, social_weight=2)
        },
        "Ant_Colony": {
            0:create_aco_model(num_ants=10, num_iterations=200, alpha=1, beta=2, evaporation_rate=0.3),
            1:create_aco_model(num_ants=20, num_iterations=100, alpha=1, beta=2, evaporation_rate=0.3),
            2:create_aco_model(num_ants=10, num_iterations=200, alpha=1, beta=2, evaporation_rate=0.5),
        }
    }

    for model_name, model_case in models.items():
        for index, model in model_case.items():
            model_name_case = f"{model_name}_case{index+1}_part1"
            print(f"Executing {model_name_case}")
            execute_algorithms(coordinates, model, model_name_case, experiments=EXPERIMENTS, cities_counts=[12, 36, 52])  # 考慮12、36、52個城市

### part 2 class
class GATSP(GeneticAlgorithmTSP):
    def __init__(self, coordinates, population_size=50, generations=1000, mutation_rate=0.2, calculate_distances=None):
        super().__init__(coordinates, population_size, generations, mutation_rate, calculate_distances=odd_calculate_distances)

    def initialize_population(self):
        # 初始化種群：隨機生成初始路徑, 奇數城市之間無連接
        population = []
        for _ in range(self.population_size):
            route = list(range(self.num_cities))
            route_odd = [i for i in route if i % 2 == 1]  # 奇數城市
            route_even = [i for i in route if i % 2 == 0]  # 偶數城市
            np.random.shuffle(route_odd)  # 隨機打亂奇數城市
            np.random.shuffle(route_even)
            # 確保奇數城市之間無連接
            route = []
            for i in range(len(route_odd)):
                route.append(route_odd[i])
                if i < len(route_even):
                    route.append(route_even[i])
            if len(route_even) > len(route_odd):
                route.append(route_even[-1])
            population.append(route)
        return population

class PSOTSP(ParticleSwarmTSP):
    def __init__(self, coordinates, population_size=10000, generations=100, inertia_weight=1, cognitive_weight=2, social_weight=3, calculate_distances=None):
        super().__init__(coordinates, population_size, generations, inertia_weight, cognitive_weight, social_weight, calculate_distances=odd_calculate_distances)

    def initialize_particles(self):
        # 初始化粒子：隨機生成初始路徑，奇數城市之間無連接
        particles = []
        for _ in range(self.population_size):
            route = list(range(self.num_cities))
            route_odd = [i for i in route if i % 2 == 1]
            route_even = [i for i in route if i % 2 == 0]
            np.random.shuffle(route_odd)
            np.random.shuffle(route_even)
            # 確保奇數城市之間無連接
            route = []
            for i in range(len(route_odd)):
                route.append(route_odd[i])
                if i < len(route_even):
                    route.append(route_even[i])
            if len(route_even) > len(route_odd):
                route.append(route_even[-1])
            particles.append(route)
        # 初始化速度：每個速度為空的操作序列（尚未方向性）
        velocities = [[] for _ in range(self.population_size)]
        return particles, velocities

class ACOTSP(AntColonyTSP):
    def __init__(self, coordinates, num_ants=10, num_iterations=200, alpha=1, beta=3, evaporation_rate=0.5, calculate_distances=None):
        super().__init__(coordinates, num_ants, num_iterations, alpha, beta, evaporation_rate, calculate_distances=odd_calculate_distances)

    def initialize_pheromone(self):
        # 初始化信息素矩陣，奇數城市之間無連接
        self.pheromone = np.full((self.num_cities, self.num_cities), 1e-6)  # 初始信息素值
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i % 2 == j % 2:  # 如果兩個城市都是奇數或都是偶數
                    self.pheromone[i][j] = float('inf')  # 奇數城市之間的距離設為無限大

def create_ga_model_2(population_size=50, generations=1000, mutation_rate=0.2, calculate_distances=None):
    return lambda coordinates: GATSP(coordinates, population_size, generations, mutation_rate, calculate_distances)

def create_pso_model_2(population_size=10000, generations=100, inertia_weight=1, cognitive_weight=2, social_weight=3, calculate_distances=None):
    return lambda coordinates: PSOTSP(coordinates, population_size, generations, inertia_weight, cognitive_weight, social_weight, calculate_distances)

def create_aco_model_2(num_ants=10, num_iterations=200, alpha=1, beta=3, evaporation_rate=0.5, calculate_distances=None):
    return lambda coordinates: ACOTSP(coordinates, num_ants, num_iterations, alpha, beta, evaporation_rate, calculate_distances)



def part_2():
    # Part II: Only the first 36 cities are considered; determine the shortest route path if all the odd-numbered cities
    # are not connected. Note that the fitness function is different from that in Part I.

    coordinates = list(read_tsp_files("../dataset").values())[0]  # 讀取第一個TSP文件的座標

    num_cities = 36  # 只考慮前36個城市
    coordinates = coordinates[:num_cities]  # 只考慮前36個城市

    models = {
        "Genetic_Algorithm": {
            0:create_ga_model_2(population_size=50, generations=1500, mutation_rate=0.15, calculate_distances=odd_calculate_distances),
            1:create_ga_model_2(population_size=100, generations=1000, mutation_rate=0.1, calculate_distances=odd_calculate_distances)
        },
        "Particle_Swarm": {
            0:create_pso_model_2(population_size=8000, generations=150, inertia_weight=0.8, cognitive_weight=1.5, social_weight=2.5, calculate_distances=odd_calculate_distances),
            1:create_pso_model_2(population_size=6000, generations=250, inertia_weight=0.6, cognitive_weight=1.8, social_weight=3.2, calculate_distances=odd_calculate_distances)
        },
        "Ant_Colony": {
            0:create_aco_model_2(num_ants=15, num_iterations=300, alpha=1, beta=2.5, evaporation_rate=0.4, calculate_distances=odd_calculate_distances),
            1:create_aco_model_2(num_ants=25, num_iterations=200, alpha=1, beta=3, evaporation_rate=0.6, calculate_distances=odd_calculate_distances),
            2:create_aco_model_2(num_ants=20, num_iterations=400, alpha=1, beta=3.5, evaporation_rate=0.5, calculate_distances=odd_calculate_distances),
        }
    }

    for model_name, model_case in models.items():
        for index, model in model_case.items():
            model_name_case = f"{model_name}_case{index+1}_part2"
            print(f"Executing {model_name_case}")
            execute_algorithms(coordinates, model, model_name_case, experiments=EXPERIMENTS, cities_counts=[36,])  # 只考慮36個城市


def prepare_output_directory():
    import os
    output_dir = "../output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(os.path.join(output_dir, "images")):
        os.makedirs(os.path.join(output_dir, "images"))
    if not os.path.exists(os.path.join(output_dir, "json")):
        os.makedirs(os.path.join(output_dir, "json"))
    if not os.path.exists(os.path.join(output_dir, "csv")):
        os.makedirs(os.path.join(output_dir, "csv"))

if __name__ == "__main__":
    prepare_output_directory()  # 確保輸出目錄存在
    part_1()
    part_2()