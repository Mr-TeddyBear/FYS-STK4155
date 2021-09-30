import numpy as np


def OLS(X, y):
    """
    Simple ordinary least squres method using matix inversion.
    Returns beta matrx for OLS
    """
    print("OLS shapes", X.shape, y.shape)
    print((X.T @ X))
    olsBeta = (np.linalg.inv(X.T @ X) @ X.T) @ y
    return olsBeta
