import numpy as np
from tqdm import tqdm
from TSP import TravelingSalesmanProblem

class GeneticAlgorithmTSP(TravelingSalesmanProblem):
    def __init__(self, coordinates, population_size, generations, mutation_rate, calculate_distances=None,num_elites=5, crossover_type="OX"):
        super().__init__(coordinates, population_size, generations, calculate_distances)
        self.mutation_rate = mutation_rate
        self.num_elites = num_elites
        self.crossover_type = crossover_type

    def initialize_population(self):
        return [np.random.permutation(self.num_cities) for _ in range(self.population_size)]

    def select_parents(self, population):
        fitness_values = [self.fitness_function(ind) for ind in population]
        if np.sum(fitness_values) == 0:
            indices = np.random.choice(range(self.population_size), size=2)
        else:
            probabilities = fitness_values / np.sum(fitness_values)
            indices = np.random.choice(range(self.population_size), size=2, p=probabilities)
        return [population[i] for i in indices]

    def crossover(self, parent1, parent2):
        if self.crossover_type == "PMX":
            return self._pmx_crossover(parent1, parent2)
        else:
            return self._ox_crossover(parent1, parent2)

    def _ox_crossover(self, parent1, parent2):
        start, end = sorted(np.random.choice(range(self.num_cities), 2, replace=False))
        child = [-1] * self.num_cities
        child[start:end+1] = parent1[start:end+1]
        fill_pos = (end + 1) % self.num_cities
        p2_idx = fill_pos
        while -1 in child:
            gene = parent2[p2_idx % self.num_cities]
            if gene not in child:
                child[fill_pos] = gene
                fill_pos = (fill_pos + 1) % self.num_cities
            p2_idx += 1
        return child

    def _pmx_crossover(self, parent1, parent2):
        start, end = sorted(np.random.choice(range(self.num_cities), 2, replace=False))
        child = [-1] * self.num_cities
        child[start:end+1] = parent1[start:end+1]
        for i in range(start, end+1):
            gene = parent2[i]
            if gene not in child:
                pos = i
                while True:
                    mapped_gene = parent1[pos]
                    pos = parent2.index(mapped_gene)
                    if child[pos] == -1:
                        break
                child[pos] = gene
        for i in range(self.num_cities):
            if child[i] == -1:
                child[i] = parent2[i]
        return child

    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            i, j = np.random.choice(range(self.num_cities), size=2, replace=False)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    def two_opt(self, route):
        best = route.copy()
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1:
                        continue
                    new_route = route[:i] + route[i:j][::-1] + route[j:]
                    if self.fitness_function(new_route) > self.fitness_function(best):
                        best = new_route
                        improved = True
            route = best
        return best

    def run(self):
        population = self.initialize_population()
        best_individual = []
        best_fitness = 0

        self.path_history = []  # 儲存路徑歷史
        for generation in tqdm(range(self.generations), desc="Running Genetic Algorithm", leave=False):
            population.sort(key=self.fitness_function, reverse=True)
            elites = population[:self.num_elites]

            new_population = elites.copy()
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents(population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

            for individual in population:
                fitness = self.fitness_function(individual)
                if best_fitness == float('inf') or fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()

            self.path_history.append((self.distance(best_individual),best_individual.copy()))  # 儲存當前最佳路徑

            # 動態調整突變率
            self.mutation_rate = max(0.01, self.mutation_rate * 0.99)

        # 最佳解使用 2-opt 微調
        best_individual = self.two_opt(best_individual)
        best_individual = [int(i) for i in best_individual]
        return best_individual

if __name__ == "__main__":
    from dataset import read_tsp_files
    # 從指定的資料夾讀取TSP文件並返回座標列表
    coordinates = list(read_tsp_files("../dataset").values())[0]  # 讀取第一個TSP文件的座標

    # 初始化基因演算法並執行
    ga = GeneticAlgorithmTSP(coordinates, population_size=50, generations=1000, mutation_rate=0.2)
    best_individual = ga.run()
    best_fitness = ga.fitness_function(best_individual)  # 計算最佳路徑的適應度
    print("Best Individual:", best_individual)  # 最佳路徑
    print("Best Fitness:", best_fitness)  # 最佳適應度（最低距離）
    ga.visualize_path(best_individual, title="Genetic Algorithm - Best Path", savepath=None, show=True)  # 可視化最佳路徑