from sklearn.utils import resample
import numpy as np
from utilFunctions import MSE, R2


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


def k_fold_validation(x, y, fold_length, n_folds, model):
    print("n folds", n_folds)
    mse_fold = np.empty(n_folds)
    r2_fold = np.empty(n_folds)
    for i in range(0, n_folds-1):
        print(f"Running fold {i}")
        trainX = np.ones(x.shape[0], dtype=bool)
        trainY = np.ones(y.shape, dtype=bool)

        trainX[i*fold_length:(i+1)*fold_length] = False
        trainY[i*fold_length:(i+1)*fold_length] = False

        x_train = x[trainX]
        y_train = y[trainY]

        x_test = x[np.invert(trainX)]
        y_test = y[np.invert(trainY)]

        print("shapes", x_train.shape, y_train.shape)

        beta = model(x_train, y_train)
        predict = x_test @ beta

        mse_fold[i] = MSE(y_test, predict)
        r2_fold[i] = R2(y_test, predict)

    trainX = np.ones(x.shape[0], dtype=bool)
    trainY = np.ones(y.shape, dtype=bool)

    trainX[-fold_length:] = False
    trainY[-fold_length:] = False

    x_train = x[trainX]
    y_train = y[trainY]

    x_test = x[np.invert(trainX)]
    y_test = y[np.invert(trainY)]

    beta = model(x_train, y_train)
    predict = x_test @ beta

    mse_fold[-1] = MSE(y_test, predict)
    r2_fold[-1] = R2(y_test, predict)

    return mse_fold, r2_fold


def leave_one_out_validation(x, y, model):
    n_folds = x.shape[1]
    mse_fold = np.empty(n_folds)
    r2_fold = np.empty(n_folds)
    for i in range(0, n_folds+1):
        train = np.ones(x.shape, dtype=bool)
        train[i] = False
        x_train = x[train]
        y_train = y[train]

        x_test = x[np.invert(train)]
        y_test = y[np.invert(train)]

        beta = model(x_train, y_train)
        predict = x_test @ beta

        mse_fold[i] = MSE(y_test, predict)
        r2_fold[i] = R2(y_test, predict)

    return mse_fold, r2_fold


def run_kfold(x, y, model, nfold=5):
    fold_size = int(x.shape[0]/5)
    if (x.shape[0] <= nfold):
        mse, r2 = leave_one_out_validation(x, y, model)
    else:
        # do k-fold validation
        mse, r2 = k_fold_validation(
            x, y, fold_size, n_folds=nfold, model=model)
    return np.mean(mse), np.mean(r2)
