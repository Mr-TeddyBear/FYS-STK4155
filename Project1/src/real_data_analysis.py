import numpy as np
from imageio import imread


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Import self written code
from frankefunction import FrankeFunction, PlotFrankeFunction
from utilFunctions import MSE, R2, create_X
from regressionMethods import OLS
from bootstrap import run_bootstrap
from k_fold import run_kfold


def run_realdata_analasys(filename):

    terrain_data = imread(filename)

    def plot_terrain(data):
        plt.figure()
        plt.title("Terrain data Norway 1")
        plt.imshow(data, cmap="gray")
        plt.xlabel("X")
        plt.ylabel("Y")
    # plt.show()

    # model complexity
    n = 5

    terrain_data_slice = terrain_data[400:800, 400:800]

    plot_terrain(terrain_data_slice)
    plt.show()

    def create_XY(terrain, norm=True):
        if norm:
            terrain = (terrain-np.min(terrain)) / \
                (np.max(terrain) - np.min(terrain))

        x = np.linspace(0, 1, terrain.shape[0])
        y = np.linspace(0, 1, terrain.shape[1])

        return x, y, terrain

    x, y, z = create_XY(terrain_data_slice)

    X = create_X(x, y, n, "lin")


if __name__ == "__main__":
    run_realdata_analasys('SRTM_data_Norway_1.tif')
