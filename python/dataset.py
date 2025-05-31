import glob
import os

def read_tsp_files(directory_path):
    tsp_files = glob.glob(os.path.join(directory_path, "*.tsp"))
    datasets = {}

    for tsp_file in tsp_files:
        with open(tsp_file, 'r') as file:
            coordinates = []
            reading_coords = False
            for line in file:
                line = line.strip()
                if line == "NODE_COORD_SECTION":
                    reading_coords = True
                    continue
                if reading_coords:
                    if line == "EOF":
                        break
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            x, y = float(parts[1]), float(parts[2])
                            coordinates.append((x, y))
                        except ValueError:
                            pass
            datasets[os.path.basename(tsp_file)] = coordinates

    return datasets

if __name__ == "__main__":
    directory_path = "../dataset"
    tsp_data = read_tsp_files(directory_path)
    for filename, coords in tsp_data.items():
        print(f"{filename}: {coords}")