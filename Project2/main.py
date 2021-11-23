import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression

from src.gradientDecent import gradientDecent
from src.regressionMethods import OLS
from src.utilFunctions import create_X, R2, MSE
from src.frankefunction import FrankeFunction
from src.FFNN import FFNNetwork

from tqdm import tqdm

import seaborn as sea


def run_SGD():
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

    theta = np.random.randn(X.shape[1])
    theta = sgd.calculateGradient(OLS, train_X, train_Y, theta)

    print(f"Simple OLS: {theta_linreg}")
    print(f"SGD theta: {theta}")

    ypred = test_X.dot(theta_linreg)
    ypred2 = test_X @ theta

    print(f"OLS MSE: {MSE(ypred, test_Y)}")
    print(f"SGD MSE: {MSE(ypred2, test_Y)}")

    # plt.plot(xnew, ypred, "ro-")
    # plt.plot(xnew, ypred2, "ro-")
    # plt.show()

    skSGD = SGDRegressor()

    skSGD.fit(train_X, train_Y)

    mseSKlearn = mean_squared_error(test_Y, skSGD.predict(test_X))

    print(f"SKlearn MSE: {mseSKlearn}")

    epochs = range(60, 301, 20)
    M = range(20, 80, 10)
    degree = range(5, 15)
    t0, t1 = 5, 500

    MSE_result = np.empty([len(epochs), len(M)])

    for i, e in tqdm(enumerate(epochs)):
        for j, m in enumerate(M):
            theta = np.random.rand(X.shape[1])
            sgd = gradientDecent(e, m, t0, t1)
            B = sgd.calculateGradient(OLS, train_X, train_Y, theta)
            pred = test_X.dot(B)
            mse_tmp = MSE(pred, test_Y)
#            print(mse_tmp, e, m)
            MSE_result[i, j] = mse_tmp

    sea.heatmap(MSE_result, robust=True)
    plt.show()


def run_FFNN_grad():
    np.random.seed(1804)
    n = 2
    N = 100
    level = 0.2

    x = np.random.uniform(0, 1, N) + np.random.normal(0, 0.1, N)*level
    y = np.random.uniform(0, 1, N) + np.random.normal(0, 0.1, N)*level

    X = create_X(x, y, n)
    z = FrankeFunction(x, y)

    #z = np.array(z)
    print(z.shape, X.shape)

    train_X, test_X, train_Y, test_Y = train_test_split(X, z, test_size=0.2)

    print(train_Y.shape)
    epochs = range(60, 301, 40)
    M = range(20, 80, 10)
    degree = range(5, 15)
    t0, t1 = 5, 500

    acc_score = np.empty([len(epochs), len(M)])

    for i, e in enumerate(epochs):
        for j, m in enumerate(M):
            layers = (train_X.shape[0], 100, 100)
            network = FFNNetwork(train_X, train_Y, test_X, test_Y, layers)
            network.train(n_batches=m, n_epochs=e)
            accuracy = network.accuracy(test_X, test_Y)

            skNetwork = MLPRegressor(hidden_layer_sizes=layers[1:], activation='logistic', solver="sgd", alpha=0.01, batch_size=int(
                train_X.shape[0] // m), learning_rate='constant', learning_rate_init=0.01)
            skNetwork.fit(train_X, train_Y)
            skScore = skNetwork.score(test_X, test_Y)

            print("Accuracy ", accuracy, skScore)

            acc_score[i, j] = accuracy
    print(np.min(acc_score), np.argmin(acc_score))
    sea.heatmap(acc_score, robust=True)
    plt.show()


def run_FFNN_skdata():
    X, y = make_regression(n_samples=200, random_state=1)
    train_X, test_X, train_Y, test_Y = train_test_split(X, y,
                                                        random_state=1)

    print(train_Y.shape)
    epochs = range(60, 301, 40)
    M = range(20, 80, 10)
    degree = range(5, 15)
    t0, t1 = 5, 500

    print(train_X.shape)

    acc_score = np.empty([len(epochs), len(M)])

    for i, e in enumerate(epochs):
        for j, m in enumerate(M):
            layers = (train_X.shape[1], 100, 100)
            network = FFNNetwork(train_X, train_Y, test_X, test_Y, layers)
            network.train(n_batches=m, n_epochs=e)
            accuracy = network.accuracy(test_X, test_Y)

            skNetwork = MLPRegressor(hidden_layer_sizes=layers[1:], activation='logistic', solver="sgd", alpha=0.01, batch_size=int(
                train_X.shape[0] // m), learning_rate='constant', learning_rate_init=0.01)
            skNetwork.fit(train_X, train_Y)
            skScore = skNetwork.score(test_X, test_Y)

            print("Accuracy ", accuracy, skScore)

            acc_score[i, j] = accuracy
    print(np.min(acc_score), np.argmin(acc_score))
    sea.heatmap(acc_score, robust=True)
    plt.show()


if __name__ == "__main__":
    run_FFNN_skdata()
    run_FFNN_grad()
    run_SGD()
