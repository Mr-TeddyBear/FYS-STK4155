from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Import self written code
from frankefunction import FrankeFunction, PlotFrankeFunction
from utilFunctions import MSE, R2, create_X
from regressionMethods import OLS
from bootstrap import run_bootstrap
from k_fold import run_kfold
from testFunctions import *


def main():
    # Path("figures").mkdir(parents=True, exist_ok=True)
    np.random.seed(1859)
    n = 2
    N = 100
    level = 2
    x = np.random.uniform(0, 1, N) + np.random.normal(0, 0.1, N)*level
    y = np.random.uniform(0, 1, N) + np.random.normal(0, 0.1, N)*level
    #print(np.shape(x), np.shape(y))
    X = create_X(x, y, n=n, method="squared")
    z = FrankeFunction(x, y)
    print(np.shape(X), np.shape(z))
    input()
    train_X, test_X, train_Y, test_Y = train_test_split(X, z, test_size=0.2)

    ols_fit = OLS(train_X, train_Y)

    comp_n, complexity_boot = model_complexity_bootstrap(
        n_comlexity=20, n_boot=10)

    #plt.plot(comp_n, complexity_boot[:, 0], '-o')

    plt.plot(np.log10(complexity_boot[:, 1]), '-o')
    plt.plot(np.log10(complexity_boot[:, 2]), '-o')
    plt.legend(["Bias", "Variance"])
    plt.title("Bias variance tradeoff, using OLS bootstrap, log10")
    # plt.show()
    plt.savefig("./figures/ols_bias_variance_tradeoff.pdf", dpi=600)
    plt.figure()

    """
    Calculate and plot bias-variance on test and train data, using OLS
    """
    n_max_complex, mse_train, mse_test = model_complexity_tradeoff(x, y,
                                                                   n_comlexity=20)
    n = np.linspace(1, n_max_complex, n_max_complex, endpoint=True, dtype=int)
    plt.plot(np.log10(mse_test), "-o")
    plt.plot(np.log10(mse_train), "-o")
    plt.legend(["Test", "Train"])
    plt.title("Test vs Train data, OLS log10-log10")
    plt.savefig("./figures/ols_test_vs_train_error.pdf", dpi=600)
    plt.figure()
#    plt.show()

    """
    K-fold corss-validation
    """

    a, n = model_complexity_tradeoff_k_fold(x, y, n_complexity=20)
    n = 10
    n = np.linspace(1, n, n, endpoint=True, dtype=int)
    plt.plot(np.log10(a), '-o')
    plt.plot(np.log10(mse_test), '-o')
    plt.legend(["Kfold mse", "MSE"])
    plt.title("Model complexity tradeoff OLS")
    plt.savefig("./figures/ols_kfold_vs_mse.pdf", dpi=600)

    #b, _ = sklearn_kfold(x, y)
    #plt.plot(n, b)
    # print(b)
    #plt.legend(["self", "sklearn"])

    n_c, n_l, ridge_k_mse_train, ridge_k_mse_test, ridge_bootstrap = ridge_bootstrap_and_kfold(
        x, y)
    plt.figure()

    plt.plot(np.min(ridge_k_mse_test, axis=1), '-o')
    plt.title("Lambda MSE K-fold, Ridge")
    plt.legend(["Test"])
    plt.savefig("./figures/ridge_lamge_mse_kfold.pdf", dpi=600)
    plt.figure()

    plt.plot(np.min(ridge_bootstrap[:, :, 0], axis=1), '-o')
    plt.title("Lambda MSE Bootstrap, Ridge")
    plt.savefig("./figures/ridge_lambda_bootstrap.pdf", dpi=600)
    plt.figure()

    mean_ridge_boot = []
    for i in n:
        mean_ridge_boot.append(np.mean(ridge_bootstrap[i-1, :, 2]))
    plt.plot(np.log10(mean_ridge_boot), '-o')
    plt.title("Mean bootstrap variance, Ridge")
    plt.savefig("./figures/ridge_bootstrap_variance.pdf", dpi=600)
    plt.figure()

    mean_bootstrap_bias = []
    for i in n:
        mean_bootstrap_bias.append(np.mean(ridge_bootstrap[:, i-1, 1]))
    plt.plot(mean_bootstrap_bias, '-o')
    plt.title("Mean bootstrap bias, Ridge")
    plt.savefig("./figures/ridge_bootstrap_bias.pdf", dpi=600)
    plt.figure()

    plt.plot(np.log10(mean_ridge_boot), '-o')
    plt.plot(np.log10(mean_bootstrap_bias), '-o')
    plt.legend(["variance", "bias"])
    plt.title("mean Bootstrap bias vs variance, Ridge, log10")
    plt.savefig("./figures/ridge_bias_vs_variance.pdf", dpi=600)

    bootstrap_best_error_complexity, bootstrap_best_error_lambda = np.where(
        ridge_bootstrap[:, :, 0] == np.amin(ridge_bootstrap[:, :, 0]))
    kfold_best_error_complexity, kfold_best_error_lambda = np.where(
        ridge_k_mse_test == np.amin(ridge_k_mse_test))
    print(
        f"Best error from bootstrap, complexity: {bootstrap_best_error_complexity}, lambda: {bootstrap_best_error_lambda}")
    print(
        f"Best error from K-fold  complexity: {kfold_best_error_complexity}, lambda: {kfold_best_error_lambda}")

    alpha = 10**np.arange(-2, 4, 1, dtype=float)
    n = np.arange(0, len(alpha), 1)
    n_c, n_l, lasso_k_mse_train, lasso_k_mse_test, lasso_bootstrap = lasso_bootstrap_and_kfold(
        x, y, alpha)
    plt.figure()

    plt.plot(np.min(lasso_k_mse_test, axis=1), '-o')
    plt.title("Alpha MSE K-fold, lasso")
    plt.legend(["Test"])
    plt.savefig("./figures/lasso_lambda_kfold.pdf", dpi=600)
    plt.figure()

    plt.plot(np.min(lasso_bootstrap[:, :, 0], axis=1), '-o')
    plt.title("Aplha MSE Bootstrap, lasso")
    plt.savefig("./figures/lasso_alpha_bootstrap.pdf", dpi=600)
    plt.figure()

    mean_lasso_boot = []
    for i in n:
        mean_lasso_boot.append(np.mean(lasso_bootstrap[i-1, :, 2]))
    plt.plot(mean_lasso_boot, '-o')
    plt.title("Mean bootstrap variance, lasso")
    plt.savefig("./figures/lasso_bootstrap_variance.pdf", dpi=600)
    plt.figure()

    mean_bootstrap_bias = []
    for i in n:
        mean_bootstrap_bias.append(np.mean(lasso_bootstrap[:, i-1, 1]))
    plt.plot(mean_bootstrap_bias, '-o')
    plt.title("Mean bootstrap bias, lasso")
    plt.savefig("./figures/lasso_bootstrap_bias.pdf", dpi=600)
    plt.figure()

    plt.plot(mean_lasso_boot, '-o')
    plt.plot(mean_bootstrap_bias, '-o')
    plt.legend(["variance", "bias"])
    plt.savefig("./figures/lasso_bias_vs_variance.pdf", dpi=600)
    plt.title("mean Bootstrap bias vs variance, lasso")

    bootstrap_best_error_complexity, bootstrap_best_error_lambda = np.where(
        lasso_bootstrap[:, :, 0] == np.amin(lasso_bootstrap[:, :, 0]))
    kfold_best_error_complexity, kfold_best_error_lambda = np.where(
        lasso_k_mse_test == np.amin(lasso_k_mse_test))
    print(
        f"Best error from bootstrap, complexity: {bootstrap_best_error_complexity}, alpha: {bootstrap_best_error_lambda}")
    print(
        f"Best error from K-fold  complexity: {kfold_best_error_complexity}, alpha: {kfold_best_error_lambda}")

    plt.show()


if __name__ == "__main__":
    main()
