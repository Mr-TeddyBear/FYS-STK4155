from sklearn.utils import resample
import numpy as np


def _bootstrap(n_bootstrap_runs, x, y, x_test, y_test, model):
    y_pred = np.empty((y_test.shape[0], n_bootstrap_runs))
    for i in range(n_bootstrap_runs):
        x_, y_ = resample(x, y)
        y_pred[:, i] = (x_test @ model(x_, y_))

    return y_pred


def error_bias_variance(y_test, y_pred):
    error = np.zeros(y_pred.shape[1])
    for i in range(y_pred.shape[1]):
        error[i] = (np.mean((y_test - y_pred[:, i])
                            ** 2, keepdims=True))
    error = np.mean(error)
    bias = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True))**2)
    variance = np.mean(np.var(y_pred, axis=1, keepdims=True))
    return {"error": error, "bias": bias, "variance": variance}


def run_bootstrap(n_bootstrap_runs, x, y, model, x_test, y_test):
    y_pred = _bootstrap(n_bootstrap_runs, x, y, x_test, y_test, model)
    return error_bias_variance(y_test, y_pred)
