import numpy as np


def OLS(X, y):
    """
    Simple ordinary least squres method using matix inversion.
    Returns beta matrx for OLS
    """
    olsBeta = (np.linalg.inv(X.T @ X) @ X.T) @ y
    return olsBeta
