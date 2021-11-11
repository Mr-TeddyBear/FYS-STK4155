import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

from src.gradientDecent import gradientDecent
from src.regressionMethods import OLS
from src.utilFunctions import create_X, R2, MSE
from src.frankefunction import FrankeFunction


def main():
    np.random.seed(1804)
    n = 2
    N = 100
    level = 0

    x = np.random.uniform(0, 1, N) + np.random.normal(0, 0.1, N)*level
    y = np.random.uniform(0, 1, N) + np.random.normal(0, 0.1, N)*level

    X = create_X(x, y, n)
    z = FrankeFunction(x, y)

    # Simple OLS
    train_X, test_X, train_Y, test_Y = train_test_split(X, z, test_size=0.2)

    theta_linreg = np.linalg.inv(train_X.T @ train_X) @ (train_X.T @ train_Y)
    epochs, M, t0, t1 = 50, 5, 5, 50
    sgd = gradientDecent(epochs, M, t0, t1)

    theta = np.random.randn(2, 1)

    theta = sgd.calculateGradient(OLS, train_X, train_Y, theta)

    print(f"Simple OLS: {theta_linreg}")
    print(f"SGD theta: {theta}")

    ypred = test_X.dot(theta_linreg)
    ypred2 = test_Y.dot(theta)

    print(f"OLS MSE: {MSE(theta_linreg, test_Y)}")
    print(f"SGD MSE: {MSE(theta, test_Y)}")

    # plt.plot(xnew, ypred, "ro-")
    # plt.plot(xnew, ypred2, "ro-")
    # plt.show()


if __name__ == "__main__":
    main()
