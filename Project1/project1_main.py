import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Import self written code
from frankefunction import FrankeFunction, PlotFrankeFunction
from utilFunctions import MSE, R2, create_X
from regressionMethods import OLS
from bootstrap import run_bootstrap


if __name__ == "__main__":
    np.random.seed(1859)
    n = 5
    N = 100
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    #print(np.shape(x), np.shape(y))
    X = create_X(x, y, n=n, method="squared")
    z = FrankeFunction(x, y)
    #print(np.shape(X), np.shape(z))

    train_X, test_X, train_Y, test_Y = train_test_split(X, z, test_size=0.2)

    ols_fit = OLS(train_X, train_Y)

    print(np.shape(train_X))
    print(np.shape(train_Y))
    print(np.shape(ols_fit))

    n_boot = 100
    bootstrap_output = run_bootstrap(
        n_boot, train_X, train_Y, OLS, test_X, test_Y)
