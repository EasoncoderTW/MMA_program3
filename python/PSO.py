import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm

class ParticleSwarmOptimization:
    def __init__(self, coordinates, population_size, generations, inertia_weight, cognitive_weight, social_weight):
        self.coordinates = coordinates  # 城市的座標
        self.population_size = population_size  # 粒子數量
        self.generations = generations  # 演化的代數
        self.num_cities = len(coordinates)  # 城市的數量
        self.distances = self.calculate_distances()  # 計算城市之間的距離矩陣
        self.inertia_weight = inertia_weight  # 慣性權重
        self.cognitive_weight = cognitive_weight  # 認知權重
        self.social_weight = social_weight  # 社會權重

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

    def initialize_particles(self):
        # 初始化粒子的位置和速度
        particles = [np.random.permutation(self.num_cities) for _ in range(self.population_size)]
        velocities = [np.zeros(self.num_cities) for _ in range(self.population_size)]
        return particles, velocities

    def update_velocity(self, velocity, particle, personal_best, global_best):
        # 更新粒子的速度
        inertia = self.inertia_weight * velocity
        cognitive = self.cognitive_weight * np.random.rand(self.num_cities) * (personal_best - particle)
        social = self.social_weight * np.random.rand(self.num_cities) * (global_best - particle)
        new_velocity = inertia + cognitive + social
        return new_velocity

    def update_position(self, particle, velocity):
        # 更新粒子的位置
        new_particle = particle + velocity
        new_particle = np.argsort(new_particle)  # 將位置轉換為有效的城市排列
        return new_particle

    def run(self):
        # 主程式：執行粒子群演算法
        particles, velocities = self.initialize_particles()  # 初始化粒子
        personal_bests = particles.copy()  # 每個粒子的最佳位置
        personal_best_fitness = [self.fitness(p) for p in particles]  # 每個粒子的最佳適應度
        global_best = personal_bests[np.argmin(personal_best_fitness)]  # 全局最佳位置
        global_best_fitness = min(personal_best_fitness)  # 全局最佳適應度

        for generation in tqdm(range(self.generations), desc="Running PSO"):
            # tqdm.write(f"Generation {generation}: Best Fitness so far: {global_best_fitness}")
            for i in range(self.population_size):
                velocities[i] = self.update_velocity(velocities[i], particles[i], personal_bests[i], global_best)
                particles[i] = self.update_position(particles[i], velocities[i])
                fitness = self.fitness(particles[i])
                if fitness < personal_best_fitness[i]:
                    personal_bests[i] = particles[i]
                    personal_best_fitness[i] = fitness
                if fitness < global_best_fitness:
                    global_best = particles[i]
                    global_best_fitness = fitness

        global_best = [int(i) for i in global_best]  # 將最佳路徑轉換為列表格式
        return global_best, global_best_fitness  # 返回最佳路徑及其適應度

if __name__ == "__main__":
    from dataset import read_tsp_files
    # 從指定的資料夾讀取TSP文件並返回座標列表
    coordinates = list(read_tsp_files("../dataset").values())[0]  # 讀取第一個TSP文件的座標

    # 初始化粒子群演算法並執行
    pso = ParticleSwarmOptimization(coordinates, population_size=100, generations=100, inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5)
    best_individual, best_fitness = pso.run()
    best_individual = [int(i) for i in best_individual]  # 將最佳路徑轉換為列表格式
    print("Best Individual:", best_individual)  # 最佳路徑
    print("Best Fitness:", best_fitness)  # 最佳適應度（最低距離）
