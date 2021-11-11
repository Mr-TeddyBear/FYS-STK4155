import numpy as np
from sklearn.utils import resample


class gradientDecent():
    """
    Class for calcualting the stokastic gradient decent(SGD)


    -- Keyword arguments --
    epochs -- number of epochs
    m -- number of minibatches
    M -- size of minibatches
    """

    def __init__(self, epochs: int, M: int, t0: int, t1: int):
        self.epochs = epochs
        self.M = M
        self.m = int(epochs/M)
        self.t0 = t0
        self.t1 = t1

    def updateInitialConditions(self, epoch: int, m: int, M: int):
        self.epochs = epoch
        self.M = M
        self.m = int(epoch/M)

    def learning_rate(self, t=1):
        return self.t0/(self.t1 + t)

    def calculateGradient(self, regMethod, X: np.ndarray, y: np.ndarray, theta: float):
        """
        Simple SGD without momentum
        """
        for epoch in range(self.epochs):
            for i in range(self.m):
                Xb, yb = resample(X, y, replace=False, n_samples=self.M)
                print(Xb.shape, yb.shape)
                grad = regMethod(Xb, yb, theta)
                l_rate = self.learning_rate(epoch*self.m+i)
                theta = theta - l_rate*grad
        return theta

    @property
    def getParamaters(self):
        str_out = f"Number of epochs: {self.epochs}\n" + \
                  f"Size of minibatches: {self.M}\n" + \
                  f"t0, t1: {self.t0} {self.t1}"
        print(str_out)
