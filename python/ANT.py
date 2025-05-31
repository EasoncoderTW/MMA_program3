import numpy as np
from scipy.spatial.distance import euclidean

from tqdm import tqdm

class AntColonyOptimization:
    def __init__(self, coordinates, num_ants, num_iterations, alpha, beta, evaporation_rate):
        # 初始化螞蟻演算法的參數
        self.coordinates = coordinates  # 城市的座標
        self.num_ants = num_ants  # 每次迭代的螞蟻數量
        self.num_iterations = num_iterations  # 演算法的迭代次數
        self.alpha = alpha  # 信息素的重要性指數
        self.beta = beta  # 距離的重要性指數
        self.evaporation_rate = evaporation_rate  # 信息素的揮發率
        self.num_cities = len(coordinates)  # 城市的數量
        self.pheromone = np.ones((self.num_cities, self.num_cities))  # 初始化信息素矩陣
        self.distances = self.calculate_distances()  # 計算城市之間的距離矩陣

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

    def run(self):
        # 主程式：執行螞蟻演算法
        best_path = None  # 最佳路徑
        best_distance = float('inf')  # 最佳距離（最低距離）
        for _ in tqdm(range(self.num_iterations), desc="Running Ant Colony Optimization"):
            # print(f"Iteration {_ + 1}/{self.num_iterations}, best distance: {best_distance}")  # 顯示當前迭代次數
            paths = self.construct_solutions()  # 建構所有螞蟻的解（路徑）
            self.update_pheromone(paths)  # 更新信息素
            for path in paths:
                distance = self.fitness(path)  # 計算路徑的距離
                if distance < best_distance:  # 更新最佳路徑
                    best_distance = distance
                    best_path = path

        best_path = [int(i) for i in best_path]  # 將最佳路徑轉換為列表格式
        return best_path, best_distance  # 返回最佳路徑及其距離

    def construct_solutions(self):
        # 建構所有螞蟻的解（路徑）
        paths = []
        for _ in range(self.num_ants):
            path = [np.random.randint(self.num_cities)]  # 隨機選擇起始城市
            while len(path) < self.num_cities:
                probabilities = self.calculate_probabilities(path[-1], path)  # 計算下一城市的選擇機率
                next_city = np.random.choice(range(self.num_cities), p=probabilities)  # 根據機率選擇下一城市
                path.append(next_city)
            paths.append(path)  # 加入螞蟻的路徑
        return paths

    def calculate_probabilities(self, current_city, visited):
        # 計算從當前城市到其他城市的選擇機率
        probabilities = []
        for next_city in range(self.num_cities):
            if next_city not in visited:  # 如果城市未被訪問過
                probabilities.append(
                    (self.pheromone[current_city][next_city] ** self.alpha) *  # 信息素的影響
                    ((1 / self.distances[current_city][next_city]) ** self.beta)  # 距離的影響
                )
            else:
                probabilities.append(0)  # 已訪問的城市機率為0
        probabilities = np.array(probabilities)
        return probabilities / probabilities.sum()  # 正規化機率

    def update_pheromone(self, paths):
        # 更新信息素矩陣
        self.pheromone *= (1 - self.evaporation_rate)  # 信息素揮發
        for path in paths:
            distance = self.fitness(path)  # 計算路徑的距離
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i + 1]] += 1 / distance  # 根據路徑距離增加信息素

if __name__ == "__main__":
    from dataset import read_tsp_files
    # 從指定的資料夾讀取TSP文件並返回座標列表
    coordinates = list(read_tsp_files("../dataset").values())[0]  # 讀取第一個TSP文件的座標

    # 初始化螞蟻演算法並執行
    aco = AntColonyOptimization(coordinates, num_ants=20, num_iterations=200, alpha=1, beta=2, evaporation_rate=0.8)
    best_path, best_distance = aco.run()
    best_path = [int(i) for i in best_path]  # 將最佳路徑轉換為列表格式
    print("Best Path:", best_path)  # 最佳路徑
    print("Best Distance:", best_distance)  # 最佳距離（最低距離）