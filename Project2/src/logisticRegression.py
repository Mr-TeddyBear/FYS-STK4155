from src.gradientDecent import gradientDecent
from src.utilFunctions import MSE
import numpy as np


class logisticRegression():

    def __init__(self, epochs: int, M: int, t0: int, t1: int):
        self.epochs = epochs
        self.M = M
        self.batch_size = int(epochs//M)
        self.t0 = t0
        self.t1 = t1

    def _regressionStep(self, X, y, beta):
        y_mod_step = np.dot(X, beta)
        p = self.sigmoid(y_mod_step)

        deriv_step = -np.dot(X.T, (y - p))/self.batch_size

        return deriv_step

    def sigmoid(self, X):
        """
        Sigmoid activation function
        """
        return 1/(1 + np.exp(-X))

    def _SGD(self, X, y):
        SGD_rutine = gradientDecent(self.epochs, self.M, self.t0, self.t1)
        self.theta = SGD_rutine.calculateGradient(
            self._regressionStep, X, y, self.theta)

    def fit(self, X, y):
        self.theta = np.random.rand(X.shape[1], 1)
        self._SGD(X, y)

    def predict(self, X):
        return np.dot(X, self.theta)

    def score(self, X, y):
        y_pred = np.dot(X, self.theta)
        return MSE(y_pred, y)
