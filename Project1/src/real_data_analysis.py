import numpy as np
from imageio import imread


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Import self written code
from frankefunction import FrankeFunction, PlotFrankeFunction
from utilFunctions import MSE, R2, create_X
from regressionMethods import OLS, RIDGE
from bootstrap import run_bootstrap, run_bootstrap_ridge
from k_fold import run_kfold, k_fold_validation_ridge


def run_realdata_analasys(filename):

    terrain_data = imread(filename)

    def plot_terrain(data):
        plt.figure()
        plt.title("Terrain data Norway 1")
        plt.imshow(data, cmap="gray")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.figure()
                                                                                                                                
    # model complexity
    n = 2
    # Ridge lambda paramater
    lam = 10**(np.arange(1,2,1,dtype=float))

    terrain_data_slice = terrain_data[400:800, 400:800]

    plot_terrain(terrain_data_slice)
    #plt.show()

    def create_XY(terrain, norm=True):
        if norm:
            terrain = (terrain-np.min(terrain)) / \
                (np.max(terrain) - np.min(terrain))

        x = np.linspace(0, 1, terrain.shape[0])
        y = np.linspace(0, 1, terrain.shape[1])

        x, y = np.meshgrid(x,y)

        return x, y, terrain

    x, y, z = create_XY(terrain_data_slice)

    z = z.ravel()

    ols_bootstrap = np.empty([n,3])
    ols_mse_tradeoff = np.empty([n,2])
    ols_kfold_tradeoff = np.empty(n)

    ridge_bootstrap = np.empty([n,len(lam),3])
    ridge_kfold = np.empty([n,len(lam),2])
    
    for comp in range(1, n+1):
        print(f"Complexity {comp}")
        X = create_X(x, y, comp, "lin")
        

        """
        OLS
        """

        ols_bootstrap[comp-1] = model_complexity_bootstrap(X, z)
        ols_mse_tradeoff[comp-1] = model_complexity_tradeoff(X, z)
        ols_kfold_tradeoff[comp-1] = model_complexity_tradeoff_k_fold(X, z)


        """
        Ridge
        """
        
        ridge_bootstrap_kfold = []
        for i,para in enumerate(lam):
            print(f"    Lambda {para}")
            tmp = ridge_bootstrap_and_kfold(X, z, para)
            print(comp, para)
            #Kfold mse error            
            ridge_kfold[comp-1, i] = tmp["kfold"]
            ridge_bootstrap[comp-1, i] = tmp["bootstrap"]
            

        
 
    plt.plot(np.log10(ols_bootstrap[:,0]), '-o')
    plt.plot(np.log10(ols_mse_tradeoff[:,1]), '-o')
    plt.plot(np.log10(ols_kfold_tradeoff), '-o')
    plt.legend(["bootstrap_error", "MSError", "kfold_error"])
    plt.title("MSError with increasing model complexity")
    plt.figure()

    plt.plot(ols_bootstrap[:,1], '-o')
    plt.plot(ols_bootstrap[:,2], '-o')
    plt.figure()    

    print(ridge_kfold)

    for i in range(len(ridge_kfold)):
        plt.plot(ridge_kfold[i,:,1])
        plt.plot(ridge_bootstrap[i,:,0])
        plt.title(f"Lambda paramater for {i+1} complexity")
        plt.legend(["Ridge", "Bootstrap"])
        plt.figure()

    plt.show()

def model_complexity_bootstrap(X, z, n_boot=100, model=OLS):
    """
    Runs bootstrap on models from 1 to n_complexity.
    """
    n_comp_storage = np.empty([3])

    train_X, test_X, train_Y, test_Y = train_test_split(
        X, z, test_size=0.2)

    tmp = run_bootstrap(
        n_boot, train_X, train_Y, model, test_X, test_Y)
    n_comp_storage[0] = tmp["error"]
    n_comp_storage[1] = tmp["bias"]
    n_comp_storage[2] = tmp["variance"]

    return np.asarray(n_comp_storage)


def model_complexity_tradeoff(X, z, model=OLS):
    """
    Calculates MSE of test and train data for model complexity
    from 1 to n_complexity. Also generates data.
    """

    train_X, test_X, train_Y, test_Y = train_test_split(
        X, z, test_size=0.2)

    beta = model(X, z)

    tilde_test = test_X @ beta
    tilde_train = train_X @ beta

    mse_test = MSE(test_Y, tilde_test)
    mse_train = MSE(train_Y, tilde_train)

    return np.array([mse_train, mse_test])


def model_complexity_tradeoff_k_fold(X, z, model=OLS):
    mse = run_kfold(X, z, nfold=5, model=OLS)[0]
    return mse


def ridge_bootstrap_and_kfold(X, z, lam, n_boot=100, model=RIDGE):
    """
    Bootstrap
    """

    train_X, test_X, train_Y, test_Y = train_test_split(
        X, z, test_size=0.2)

    tmp = run_bootstrap_ridge(
        n_boot, train_X, train_Y, model, lam, test_X, test_Y)
    """
    K-fold
    """

    tmp = np.array([tmp['error'], tmp['bias'], tmp['variance']]) 

    mse_test,  mse_train = k_fold_validation_ridge(X, z, 10, RIDGE, lam)

    kfold = np.array([np.mean(mse_train), np.mean(mse_test)])

    return { "kfold": kfold, "bootstrap": tmp}


if __name__ == "__main__":
    run_realdata_analasys('SRTM_data_Norway_1.tif')
