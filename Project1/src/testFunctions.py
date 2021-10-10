import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Import self written code
from frankefunction import FrankeFunction, PlotFrankeFunction
from utilFunctions import MSE, R2, create_X
from regressionMethods import OLS, RIDGE
from bootstrap import run_bootstrap, run_bootstrap_ridge
from k_fold import run_kfold, k_fold_validation, k_fold_validation_ridge


def model_complexity_bootstrap(N=100, n_comlexity=10, n_boot=100, model=OLS):
    """
    Runs bootstrap on models from 1 to n_complexity.
    """
    n_comp_storage = np.empty([n_comlexity, 3])
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)

    for n in range(n_comlexity):
        X = create_X(x, y, n=n+1, method="lin")
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


def model_complexity_tradeoff(x, y, n_comlexity=10, N=100, model=OLS):
    """
    Calculates MSE of test and train data for model complexity
    from 1 to n_complexity. Also generates data.
    """
    mse_test = np.zeros(n_comlexity)
    mse_train = np.zeros(n_comlexity)
    for n in range(n_comlexity):
        X = create_X(x, y, n=n+1, method="lin")
        z = FrankeFunction(x, y)

        train_X, test_X, train_Y, test_Y = train_test_split(
            X, z, test_size=0.2)

        beta = model(X, z)

        tilde_test = test_X @ beta
        tilde_train = train_X @ beta

        mse_test[n] = MSE(test_Y, tilde_test)
        mse_train[n] = MSE(train_Y, tilde_train)

    return n_comlexity, mse_train, mse_test


def model_complexity_tradeoff_k_fold(x, y, n_complexity=10, N=100, model=OLS):
    mse = np.zeros(n_complexity)

    for n in range(n_complexity):
        X = create_X(x, y, n=n+1, method="lin")
        z = FrankeFunction(x, y)

        mse[n] = run_kfold(X, z, nfold=5, model=OLS)[0]

    return mse, n_complexity


def sklearn_kfold(x, y, n_complexity=10, N=100, model=OLS):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    mse = np.zeros(n_complexity)

    for n in range(n_complexity):
        X = create_X(x, y, n=n+1, method="lin")
        z = FrankeFunction(x, y)

        m = LinearRegression(X, z, fit_intercept=True)

        mse[n] = np.mean(-(cross_val_score(m, X, y,
                         scoring='neg_mean_squared_error', cv=5)))

    return mse, n_complexity


def ridge_bootstrap_and_kfold(x, y, n_comp=10, n_lambda=10, N=100, n_boot=100, model=RIDGE):
    mse_test = np.zeros([n_comp, n_lambda])
    mse_train = np.zeros([n_comp, n_lambda])
    n_comp_storage = np.empty([n_comp, n_lambda, 3])

    for l in range(1, n_lambda+1):
        for n in range(n_comp):
            """
            Bootstrap
            """
            X = create_X(x, y, n=n+1, method="lin")
            z = FrankeFunction(x, y)

            train_X, test_X, train_Y, test_Y = train_test_split(
                X, z, test_size=0.2)

            tmp = run_bootstrap_ridge(
                n_boot, train_X, train_Y, model, l, test_X, test_Y)
            n_comp_storage[n, l-1, 0] = tmp["error"]
            n_comp_storage[n, l-1, 1] = tmp["bias"]
            n_comp_storage[n, l-1, 2] = tmp["variance"]

            """
            K-fold
            """

            mse_test[n, l-1],  mse_train[n, l-1] = [np.mean(i) for i in k_fold_validation_ridge(
                X, z, 10, RIDGE, l)]

    return n_comp, n_lambda, mse_train, mse_test, n_comp_storage
