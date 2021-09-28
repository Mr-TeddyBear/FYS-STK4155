import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Import self written code
from frankefunction import FrankeFunction, PlotFrankeFunction
from utilFunctions import MSE, R2, create_X
from regressionMethods import OLS
from bootstrap import run_bootstrap
from testFunctions import model_complexity_bootstrap, model_complexity_tradeoff


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

#    comp_n, complexity_boot = model_complexity_bootstrap(
#        n_comlexity=8, n_boot=10)

    #plt.plot(comp_n, complexity_boot[:, 0])

# plt.plot(comp_n, complexity_boot[:, 1])
#    plt.plot(comp_n, complexity_boot[:, 2])

    n_max_complex, mse_train, mse_test = model_complexity_tradeoff(
        n_comlexity=20)
    n = np.linspace(1, n_max_complex, n_max_complex, endpoint=True, dtype=int)
    plt.plot(n, mse_test, "-o")
    plt.plot(n, mse_train, "-o")
    plt.legend(["Test", "Train"])
    plt.show()
