import numpy as np


def OLS(X, y):
    """
    Simple ordinary least squres method using matix inversion.
    Returns beta matrx for OLS
    """
    olsBeta = (np.linalg.inv(X.T @ X) @ X.T) @ y
    return olsBeta


def RIDGE(X, y, lam):
    ridgeBeta = (np.linalg.inv(
        (X.T @ X) + (np.identity(X.shape[1]) * lam)) @ X.T) @ y
    return ridgeBeta
