import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm

class GeneticAlgorithm:
    def __init__(self, coordinates, population_size, generations, mutation_rate, num_slices=1):
        # 初始化基因演算法的參數
        self.coordinates = coordinates  # 城市的座標
        self.population_size = population_size  # 每代的族群大小
        self.generations = generations  # 演化的代數
        self.mutation_rate = mutation_rate  # 突變的機率
        self.num_cities = len(coordinates)  # 城市的數量
        self.distances = self.calculate_distances()  # 計算城市之間的距離矩陣
        self.num_slices = num_slices  # 基因切片數量

    def calculate_distances(self):
        # 計算所有城市之間的距離，並存入矩陣中
        distances = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                distances[i][j] = euclidean(self.coordinates[i], self.coordinates[j])
        return distances

    def fitness(self, path):
        # 計算路徑的適應度（總距離），適應度越低表示路徑越好
        return sum(self.distances[path[i], path[i + 1]] for i in range(len(path) - 1))

    def initialize_population(self):
        # 初始化族群，隨機生成多個城市排列（路徑）
        return [np.random.permutation(self.num_cities) for _ in range(self.population_size)]

    def select_parents(self, population):
        # 根據適應度選擇父母，適應度越高（距離越短），被選中的機率越大
        fitness_values = [1 / self.fitness(individual) for individual in population]
        probabilities = fitness_values / np.sum(fitness_values)
        indices = np.random.choice(range(self.population_size), size=2, p=probabilities)
        return [population[i] for i in indices]

    def crossover(self, parent1, parent2):
        # 執行交叉操作，生成子代
        child = [-1] * self.num_cities  # 初始化子代
        slices = sorted(np.random.choice(range(self.num_cities + 1), size=self.num_slices + 1, replace=False))
        for i in range(len(slices) - 1):
            start, end = slices[i], slices[i + 1]
            if i % 2 == 0:
                child[start:end] = parent1[start:end]  # 複製父母1的基因
            else:
                child[start:end] = parent2[start:end]  # 複製父母2的基因

        # 刪除重複的基因
        child = [gene for gene in child if gene != -1]  # 移除未填充的基因
        child = list(dict.fromkeys(child))  # 保留順序且不重複

        # 填補剩餘的基因，保留順序且不重複
        missing_genes = [gene for gene in parent1 if gene not in child]
        child.extend(missing_genes)  # 填補剩餘的基因

        return child

    def mutate(self, individual):
        # 執行突變操作，隨機交換兩個城市的位置
        if np.random.rand() < self.mutation_rate:
            i, j = np.random.choice(range(self.num_cities), size=2, replace=False)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    def run(self):
        # 主程式：執行基因演算法
        population = self.initialize_population()  # 初始化族群
        best_individual = None  # 最佳路徑
        best_fitness = float('inf')  # 最佳適應度（最低距離）
        for generation in tqdm(range(self.generations), desc="Running Genetic Algorithm"):
            # tqdm.write(f"Generation {generation}: Best Individual so far: {best_fitness}")
            new_population = []  # 新的族群
            for _ in range(self.population_size):
                parent1, parent2 = self.select_parents(population)  # 選擇父母
                child = self.crossover(parent1, parent2)  # 交叉生成子代
                child = self.mutate(child)  # 子代突變
                new_population.append(child)  # 加入新族群
            population = new_population  # 更新族群
            # 更新最佳路徑
            for individual in population:
                fitness = self.fitness(individual)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_individual = individual

        best_individual = [int(i) for i in best_individual]  # 將最佳路徑轉換為列表格式
        return best_individual, best_fitness  # 返回最佳路徑及其適應度

if __name__ == "__main__":
    from dataset import read_tsp_files
    # 從指定的資料夾讀取TSP文件並返回座標列表
    coordinates = list(read_tsp_files("../dataset").values())[0]  # 讀取第一個TSP文件的座標

    # 初始化基因演算法並執行
    ga = GeneticAlgorithm(coordinates, population_size=100, generations=100, mutation_rate=0.1, num_slices=4)
    best_individual, best_fitness = ga.run()
    best_individual = [int(i) for i in best_individual]  # 將最佳路徑轉換為列表格式
    print("Best Individual:", best_individual)  # 最佳路徑
    print("Best Fitness:", best_fitness)  # 最佳適應度（最低距離）