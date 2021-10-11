import numpy as np


def scale(x):
    return (x - np.mean(x))/np.std(x)


def create_X(x, y, n, method="lin"):
    """
    Creates the beta matrix by combining x and y,
    the complxity of the matrix defined by n

    Copied from lecture notes week35
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)		# Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:, q+k] = (x**(i-k))*(y**k)

    return X


def _lin(x, y, X, n):
    """
    Combination function for create_X
    """
    for i in range(1, n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:, q+k] = (x**(i-k))*(y**k)
    return X


def _squared(x, y, X, n):
    """
    Combination function for create_X
    """
    k = 0  # counts how many rows in the matrix we have filled out
    for j in range(0, n + 1):
        for i in range(0, n - j + 1):
            X[:, k] = x**i * y**j
            k += 1
    return X


def MSE(y_data, y_model):
    return 1/len(y_data) * np.sum((y_data-y_model)**2)


def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
