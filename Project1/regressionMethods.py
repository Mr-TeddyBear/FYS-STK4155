import numpy as np


def OLS(X, y):
    olsBeta = (np.linalg.inv(X.T @ X) @ X.T) @ y
    return olsBeta
