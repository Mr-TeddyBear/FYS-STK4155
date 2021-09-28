import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Import self written code
from frankefunction import FrankeFunction, PlotFrankeFunction
from utilFunctions import MSE, R2, create_X
from regressionMethods import OLS
from bootstrap import run_bootstrap


def model_complexity_bootstrap(N=100, n_comlexity=10, n_boot=100, model=OLS):
    """
    Runs bootstrap on models from 1 to n_complexity.
    """
    n_comp_storage = np.empty([n_comlexity, 3])
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)

    for n in range(n_comlexity):
        X = create_X(x, y, n=n+1, method="squared")
        z = FrankeFunction(x, y)

        train_X, test_X, train_Y, test_Y = train_test_split(
            X, z, test_size=0.2)

        tmp = run_bootstrap(
            n_boot, train_X, train_Y, model, test_X, test_Y)
        n_comp_storage[n, 0] = tmp["error"]
        n_comp_storage[n, 1] = tmp["bias"]
        n_comp_storage[n, 2] = tmp["variance"]
        print(f"Completd model complexity {n}")

    return np.linspace(1, n_comlexity, n_comlexity, endpoint=True), n_comp_storage


def model_complexity_tradeoff(n_comlexity=10, N=100, model=OLS):
    """
    Calculates MSE of test and train data for model complexity
    from 1 to n_complexity. Also generates data.
    """
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    mse_test = np.zeros(n_comlexity)
    mse_train = np.zeros(n_comlexity)
    for n in range(n_comlexity):
        X = create_X(x, y, n=n+1, method="squared")
        z = FrankeFunction(x, y)

        train_X, test_X, train_Y, test_Y = train_test_split(
            X, z, test_size=0.2)

        beta = model(X, z)

        tilde_test = test_X @ beta
        tilde_train = train_X @ beta

        mse_test[n] = MSE(test_Y, tilde_test)
        mse_train[n] = MSE(train_Y, tilde_train)

    return n_comlexity, mse_train, mse_test
