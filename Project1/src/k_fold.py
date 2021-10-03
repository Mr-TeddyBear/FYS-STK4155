from sklearn.utils import resample
import numpy as np
from utilFunctions import MSE, R2
def k_fold_validation(x, y, n_folds, model):
    #print("n folds", n_folds)
    fold_length = int(x.shape[0] // n_folds)
    mse_fold = np.empty(n_folds)
    r2_fold = np.empty(n_folds)
    for i in range(0, n_folds-1):
        #print(f"Running fold {i}")
        trainX = np.ones(x.shape[0], dtype=bool)
        trainY = np.ones(y.shape, dtype=bool)

        trainX[i*fold_length:(i+1)*fold_length] = False
        trainY[i*fold_length:(i+1)*fold_length] = False

        x_train = x[trainX]
        y_train = y[trainY]

        x_test = x[np.invert(trainX)]
        y_test = y[np.invert(trainY)]

        #print("shapes", x_train.shape, y_train.shape)

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

def k_fold_validation_ridge(x, y, n_folds, model, l):
    fold_length = int(x.shape[0] // n_folds)
    mse_fold = np.empty(n_folds)
    r2_fold = np.empty(n_folds)
    for i in range(0, n_folds-1):
#        print(f"Running fold {i}")
        trainX = np.ones(x.shape[0], dtype=bool)
        trainY = np.ones(y.shape, dtype=bool)

        trainX[i*fold_length:(i+1)*fold_length] = False
        trainY[i*fold_length:(i+1)*fold_length] = False

        x_train = x[trainX]
        y_train = y[trainY]

        x_test = x[np.invert(trainX)]
        y_test = y[np.invert(trainY)]

        #print("shapes", x_train.shape, y_train.shape)

        beta = model(x_train, y_train, l)
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

    beta = model(x_train, y_train, l)
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
            x, y, n_folds=nfold, model=model)
    return np.mean(mse), np.mean(r2)
