import numpy as np
from imageio import imread


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Import self written code
from frankefunction import FrankeFunction, PlotFrankeFunction
from utilFunctions import MSE, R2, create_X
from regressionMethods import OLS, RIDGE
from bootstrap import run_bootstrap
from k_fold import run_kfold, k_fold_validation_ridge


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


def model_complexity_bootstrap(X, z, n, n_boot=100, model=OLS):
    """
    Runs bootstrap on models from 1 to n_complexity.
    """
    n_comp_storage = np.empty([n, 3])

    train_X, test_X, train_Y, test_Y = train_test_split(
        X, z, test_size=0.2)

    tmp = run_bootstrap(
        n_boot, train_X, train_Y, model, test_X, test_Y)
    n_comp_storage[n, 0] = tmp["error"]
    n_comp_storage[n, 1] = tmp["bias"]
    n_comp_storage[n, 2] = tmp["variance"]
    print(f"Completd model complexity {n}")

    return n_comp_storage


def model_complexity_tradeoff(X, z, n, model=OLS):
    """
    Calculates MSE of test and train data for model complexity
    from 1 to n_complexity. Also generates data.
    """
    mse_test = np.zeros(n)
    mse_train = np.zeros(n)

    train_X, test_X, train_Y, test_Y = train_test_split(
        X, z, test_size=0.2)

    beta = model(X, z)

    tilde_test = test_X @ beta
    tilde_train = train_X @ beta

    mse_test[n] = MSE(test_Y, tilde_test)
    mse_train[n] = MSE(train_Y, tilde_train)

    return mse_train, mse_test


def model_complexity_tradeoff_k_fold(X, z, n, model=OLS):
    mse = run_kfold(X, z, nfold=5, model=OLS)[0]
    return mse


def ridge_bootstrap_and_kfold(X, z, n, lam, n_boot=100, model=RIDGE):
    """
    Bootstrap
    """

    train_X, test_X, train_Y, test_Y = train_test_split(
        X, z, test_size=0.2)

    tmp = run_bootstrap_ridge(
        n_boot, train_X, train_Y, model, l, test_X, test_Y)
    """
    K-fold
    """

    mse_test,  mse_train = np.mean(k_fold_validation_ridge(
        X, z, 10, RIDGE, lam))

    return n_comp, n_lambda, mse_train, mse_test, [tmp['error'], tmp['bias'], tmp['variance']]


if __name__ == "__main__":
    run_realdata_analasys('SRTM_data_Norway_1.tif')
