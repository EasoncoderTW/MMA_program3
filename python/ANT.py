import numpy as np
from tqdm import tqdm

from TSP import TravelingSalesmanProblem

class AntColonyTSP(TravelingSalesmanProblem):
    def __init__(self, coordinates, num_ants, num_iterations, alpha, beta, evaporation_rate, calculate_distances=None):
        super().__init__(coordinates, num_ants, num_iterations, calculate_distances)  # 初始化父類別
        # 初始化螞蟻演算法的參數
        self.num_ants = num_ants  # 每次迭代的螞蟻數量
        self.alpha = alpha  # 信息素的重要性指數
        self.beta = beta  # 距離的重要性指數
        self.evaporation_rate = evaporation_rate  # 信息素的揮發率
        self.pheromone = np.ones((self.num_cities, self.num_cities))  # 初始化信息素矩陣

    def run_init(self):
        # 初始化最佳路徑和距離
        self.best_path = None
        self.best_distance = float('inf')

    def run(self):
        # 主程式：執行螞蟻演算法
        best_path = None  # 最佳路徑
        best_distance = float('inf')  # 最佳距離（最低距離）
        for _ in tqdm(range(self.generations), desc="Running Ant Colony Optimization", leave=False, position=1):
            paths = self.construct_solutions()  # 建構所有螞蟻的解（路徑）
            self.update_pheromone(paths)  # 更新信息素
            for path in paths:
                distance = self.fitness_function(path)  # 計算路徑的距離
                if distance < best_distance:  # 更新最佳路徑
                    best_distance = distance
                    best_path = path

        best_path = [int(i) for i in best_path]  # 將最佳路徑轉換為列表格式
        return best_path  # 返回最佳路徑及其距離

    def construct_solutions(self):
        # 建構所有螞蟻的解（路徑）
        paths = []
        for _ in range(self.num_ants):
            path = [np.random.randint(self.num_cities)]  # 隨機選擇起始城市
            while len(path) < self.num_cities:
                probabilities, early_stop = self.calculate_probabilities(path[-1], path)  # 計算下一城市的選擇機率
                next_city = np.random.choice(range(self.num_cities), p=probabilities)  # 根據機率選擇下一城市
                path.append(next_city)
            paths.append(path)  # 加入螞蟻的路徑
        return paths

    def calculate_probabilities(self, current_city, visited):
        # 計算從當前城市到其他城市的選擇機率
        probabilities = []
        for next_city in range(self.num_cities):
            if next_city not in visited:  # 如果城市未被訪問過
                if self.distances[current_city][next_city] == float('inf'):  # 如果距離為無限大，則機率為0
                    probabilities.append(0)
                else:
                    # 計算信息素和距離的影響
                    probabilities.append(
                        (self.pheromone[current_city][next_city] ** self.alpha) *  # 信息素的影響
                        ((1 / self.distances[current_city][next_city]) ** self.beta)  # 距離的影響
                    )
            else:
                probabilities.append(0)  # 已訪問的城市機率為0
        probabilities = np.array(probabilities)
        p_sum = probabilities.sum()
        if p_sum == 0:
            return np.array([1/len(probabilities),] * self.num_cities), True
        return (probabilities / p_sum), False  # 正規化機率

    def update_pheromone(self, paths):
        # 更新信息素矩陣
        self.pheromone *= (1 - self.evaporation_rate)  # 信息素揮發
        for path in paths:
            distance = self.fitness_function(path)  # 計算路徑的距離
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i + 1]] += 1 / distance  # 根據路徑距離增加信息素

if __name__ == "__main__":
    from dataset import read_tsp_files
    # 從指定的資料夾讀取TSP文件並返回座標列表
    coordinates = list(read_tsp_files("../dataset").values())[0]  # 讀取第一個TSP文件的座標

    # 初始化螞蟻演算法並執行
    aco = AntColonyTSP(coordinates, num_ants=10, num_iterations=200, alpha=1, beta=3, evaporation_rate=0.5)
    # aco = AntColonyTSP(coordinates, num_ants=10, num_iterations=200, alpha=1, beta=3, evaporation_rate=0.5)
    best_path = aco.run()
    best_distance = aco.fitness_function(best_path)  # 計算最佳路徑的距離
    best_path = [int(i) for i in best_path]  # 將最佳路徑轉換為列表格式
    print("Best Path:", best_path)  # 最佳路徑
    print("Best Distance:", best_distance)  # 最佳距離（最低距離）
    print([p%2 for p in best_path])  # 印出最佳路徑的城市編號（奇數城市）

    aco.visualize_path(best_path, title="AntColonyTSP", savepath=None, show=True)  # 可視化最佳路徑