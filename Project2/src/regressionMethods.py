import numpy as np


def OLS(X, y, theta, p=0):
    """
    Simple ordinary least squres method using matix inversion.
    Returns beta matrx for OLS

    -- Keyword arguments --
    X -- X array/ design matrix
    y -- y array of known outcomes
    theta -- gradient?
    p -- penalty paramater, default is not penalty
    """
    olsBeta = X.T @ (X @ theta - y) + (theta * p)
    return olsBeta


def RIDGE(X, y, lam):
    ridgeBeta = (np.linalg.inv(
        (X.T @ X) + (np.identity(X.shape[1]) * lam)) @ X.T) @ y
    return ridgeBeta
