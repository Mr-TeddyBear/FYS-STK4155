import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Import self written code
from frankefunction import FrankeFunction, PlotFrankeFunction
from utilFunctions import MSE, R2, create_X
from regressionMethods import OLS
from bootstrap import run_bootstrap
from k_fold import run_kfold
from testFunctions import *


if __name__ == "__main__":
    np.random.seed(1859)
    n = 2
    N = 100
    level = 1
    x = np.random.uniform(0, 1, N) + np.random.normal(0, 0.1, N)*level
    y = np.random.uniform(0, 1, N) + np.random.normal(0, 0.1, N)*level
    #print(np.shape(x), np.shape(y))
    X = create_X(x, y, n=n, method="squared")
    z = FrankeFunction(x, y)
#    print(np.shape(X), np.shape(z))

    train_X, test_X, train_Y, test_Y = train_test_split(X, z, test_size=0.2)

    ols_fit = OLS(train_X, train_Y)

#    comp_n, complexity_boot = model_complexity_bootstrap(
#        n_comlexity=8, n_boot=10)

    #plt.plot(comp_n, complexity_boot[:, 0])

#    plt.plot(comp_n, complexity_boot[:, 1])
#    plt.plot(comp_n, complexity_boot[:, 2])

    """
    Calculate and plot bias-variance on test and train data, using OLS
    """
    """
    n_max_complex, mse_train, mse_test = model_complexity_tradeoff(
        n_comlexity=20)
    n = np.linspace(1, n_max_complex, n_max_complex, endpoint=True, dtype=int)
    plt.plot(n, mse_test, "-o")
    plt.plot(n, mse_train, "-o")
    plt.legend(["Test", "Train"])
    plt.show()
    """

    """
    K-fold corss-validation
    """

    a, n = model_complexity_tradeoff_k_fold(x, y)

    n = np.linspace(1, n, n, endpoint=True, dtype=int)
    plt.plot(n, a)

    #b, _ = sklearn_kfold(x, y)
    #plt.plot(n, b)
    # print(b)
    #plt.legend(["self", "sklearn"])

    n_c, n_l, ridge_k_mse_train, ridge_k_mse_test, ridge_bootstrap = ridge_bootstrap_and_kfold(
        x, y)
    plt.figure()

    plt.plot(np.min(ridge_k_mse_test, axis=1), '-o')
    plt.title("Lambda MSE K-fold")
    plt.legend(["Test"])
    plt.figure()
    
    plt.plot(np.min(ridge_bootstrap[:,:,0], axis=1), '-o')
    plt.title("Lambda MSE Bootstrap")
    plt.figure()


    leg = []
    for i in n:
        plt.plot(ridge_bootstrap[i-1,:,2], '-o')
        leg.append(str(i))
    plt.legend(leg)
    plt.title("Bootstrap variance")
    plt.figure()

    for i in n:
        plt.plot(i, np.mean(ridge_bootstrap[:,i-1,1]), '-o')
        leg.append(str(i))
    plt.legend(leg)
    plt.title("Bootstrap bias")
    plt.figure()

    for i in n:
        plt.plot(i, np.mean(ridge_bootstrap[:,i-1,1]), 'g-o')
        plt.plot(i, np.mean(ridge_bootstrap[:,i-1,2]), 'y-o')
    plt.title("mean Bootstrap bias vs variance")

    bootstrap_best_error_complexity, bootstrap_best_error_lambda = np.where( ridge_bootstrap[:,:,0] == np.amin(ridge_bootstrap[:,:,0]))
    kfold_best_error_complexity, kfold_best_error_lambda  = np.where(ridge_k_mse_test ==  np.amin(ridge_k_mse_test))
    print(f"Best error from bootstrap, complexity: {bootstrap_best_error_complexity}, lambda: {bootstrap_best_error_lambda}")
    print(f"Best error from K-fold  complexity: {kfold_best_error_complexity}, lambda: {kfold_best_error_lambda}")   

    plt.show()
