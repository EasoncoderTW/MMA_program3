import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from dataset import read_tsp_files
from TSP import TravelingSalesmanProblem
import random

class ParticleSwarmTSP(TravelingSalesmanProblem):
    def __init__(self, coordinates, population_size, generations, inertia_weight, cognitive_weight, social_weight, calculate_distances=None):
        super().__init__(coordinates, population_size, generations, calculate_distances)
        s = (inertia_weight + cognitive_weight + social_weight)
        self.inertia_weight = inertia_weight / s  # 慣性權重
        self.cognitive_weight = cognitive_weight / s  # 認知權重
        self.social_weight = social_weight / s  # 社會權重

    def initialize_particles(self):
        # 初始化粒子的位置（城市排列）
        particles = [list(np.random.permutation(self.num_cities)) for _ in range(self.population_size)]

        # 初始化速度：每個速度為空的操作序列（尚未方向性）
        velocities = [[] for _ in range(self.population_size)]

        return particles, velocities

    def generate_swap_sequence(self, from_perm, to_perm):
        """產生一組 swap 操作，將 from_perm 轉換成 to_perm"""
        swaps = []
        perm = from_perm.copy()
        for i in range(len(perm)):
            if perm[i] != to_perm[i]:
                swap_index = list(perm).index(to_perm[i])
                swaps.append((i, swap_index))
                # 實際交換
                perm[i], perm[swap_index] = perm[swap_index], perm[i]
        return swaps


    def update_velocity(self, previous_velocity, particle, personal_best, global_best):
        cognitive_component = self.generate_swap_sequence(particle, personal_best)
        social_component = self.generate_swap_sequence(particle, global_best)

        # 隨機選取部分交換操作，模擬 inertia/cognitive/social 權重
        velocity = []

        # 模擬慣性（可保留部分前一代速度）
        inertia = random.sample(list(previous_velocity), int(self.inertia_weight * len(previous_velocity)))
        velocity.extend(inertia)

        cognitive = random.sample(list(cognitive_component), int(self.cognitive_weight * len(cognitive_component)))
        social = random.sample(list(social_component), int(self.social_weight * len(social_component)))

        velocity.extend(cognitive)
        velocity.extend(social)

        return velocity


    def update_position(self, particle, velocity):
        new_particle = particle.copy()
        for i, j in velocity:
            new_particle[i], new_particle[j] = new_particle[j], new_particle[i]
        return new_particle


    def run(self):
        # 主程式：執行粒子群演算法
        particles, velocities = self.initialize_particles()  # 初始化粒子
        personal_bests = particles.copy()  # 每個粒子的最佳位置
        personal_best_fitness = [self.fitness_function(p) for p in particles]  # 每個粒子的最佳適應度
        global_best = personal_bests[np.argmax(personal_best_fitness)]  # 全局最佳位置
        global_best_fitness = max(personal_best_fitness)  # 全局最佳適應度

        self.path_history = []  # 儲存路徑歷史
        for generation in tqdm(range(self.generations), desc="Running PSO", leave=False, position=1):
            # tqdm.write(f"Generation {generation}: Best Fitness so far: {global_best_fitness}")
            for i in range(self.population_size):
                velocities[i] = self.update_velocity(velocities[i], particles[i], personal_bests[i], global_best)
                particles[i] = self.update_position(particles[i], velocities[i])
                fitness = self.fitness_function(particles[i])  # 計算適應度
                if fitness > personal_best_fitness[i]:
                    personal_bests[i] = particles[i]
                    personal_best_fitness[i] = fitness
                if fitness > global_best_fitness:
                    global_best = particles[i]
                    global_best_fitness = fitness

            self.path_history.append((self.distance(global_best), global_best.copy()))  # 儲存當前全局最佳路徑

        global_best = [int(i) for i in global_best]  # 將最佳路徑轉換為列表格式
        return global_best  # 返回最佳路徑及其適應度

if __name__ == "__main__":
    from dataset import read_tsp_files
    # 從指定的資料夾讀取TSP文件並返回座標列表
    coordinates = list(read_tsp_files("../dataset").values())[0]  # 讀取第一個TSP文件的座標

    # 初始化粒子群演算法並執行
    pso = ParticleSwarmTSP(coordinates, population_size=10000, generations=100, inertia_weight=1, cognitive_weight=1, social_weight=3)
    best_individual = pso.run()
    best_fitness = pso.fitness_function(best_individual)
    best_individual = [int(i) for i in best_individual]  # 將最佳路徑轉換為列表格式
    print("Best Individual:", best_individual)  # 最佳路徑
    print("Best Fitness:", best_fitness)  # 最佳適應度（最低距離）
    pso.visualize_path(best_individual, title="PSO - Best Path", savepath=None, show=True)  # 可視化最佳路徑
