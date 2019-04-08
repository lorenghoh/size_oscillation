import os, glob, warnings

import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

# Scikit
from sklearn import linear_model as lm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

# Plotting
import lib_plot as plib
import seaborn as sns

def detect_outliers(regressor, x, y, rho=0.98):
    if regressor == None:
        regressor = lm.RidgeCV()
    
    reg_name = regressor.__class__.__name__
    if reg_name == 'RANSACRegressor':
        outlier = find_RANSAC_outlier(regressor, x, y)
        is_outlier = outlier
        return compress_inliers(is_outlier, x, y)
    elif reg_name == 'HuberRegressor':
        outlier = find_huber_outlier(regressor, x, y)
        is_outlier = outlier
        return compress_inliers(is_outlier, x, y)
    else:
        try:
            # Initial fit using linear regressor
            regressor.fit(x[:, None], y)
            y_reg = regressor.predict(x[:, None])

            is_outlier = np.zeros(len(x), dtype=bool)
            is_outlier[np.argsort(np.abs(y - y_reg))[-1]] = True
        except:
            raise TypeError("Regressor not compatible")

    x_ma = np.ma.masked_array(x, is_outlier)
    y_ma = np.ma.masked_array(y, is_outlier)
    
    max_iter = int(len(x) * 0.20) # 20% max outliers
    for i in range(max_iter):
        # Remove outlier and re-run regression
        corr = get_linear_corr(regressor, x_ma, y_ma)
        # print(corr)

        # Stop at threshold correlation score (rho)
        # Otherwise, filter next outlier
        if corr >= rho:
            break
        else:
            x_ma, y_ma, outlier = find_linear_outlier(regressor, x_ma, y_ma)
    if corr <= 0.9:
        warnings.warn("Warning: data cannot be fully linearlized")
    
    return x_ma.compressed(), y_ma.compressed(), x_ma.mask

def get_linear_corr(regressor, x_ma, y_ma):
    x_ = x_ma.compressed()
    y_ = y_ma.compressed()

    regressor.fit(x_[:, None], y_)
    return regressor.score(x_[:, None], y_)

def find_linear_outlier(regressor, x_ma, y_ma):
    regressor.fit(x_ma.compressed()[:, None], y_ma.compressed())
    y_reg = regressor.predict(x_ma.data[:, None])
    y_reg = np.ma.masked_array(y_reg, mask=y_ma.mask)

    outlier = np.abs(y_ma - y_reg).argsort(endwith=False)[-1]
    x_ma.mask[outlier] = True
    y_ma.mask[outlier] = True

    return x_ma, y_ma, outlier

def compress_inliers(is_outlier, x, y):
    x_ = np.compress(~is_outlier, x)
    y_ = np.compress(~is_outlier, y)
    return x_, y_, is_outlier

def find_RANSAC_outlier(regressor, x, y):
    regressor.fit(x[:, None], y)
    return ~regressor.inlier_mask_

def find_huber_outlier(regressor, x, y):
    regressor.fit(x[:, None], y)
    return regressor.outliers_

def plot_outliers(df):
    #---- Plotting 
    fig = plib.init_plot((8, 6))

    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1)

        # Define grid
        X = np.logspace(0, np.log10(df.max()), 50)
        
        log_kde = KernelDensity(bandwidth=1).fit(df.counts[:, None])
        kde = log_kde.score_samples(X[:, None]) / np.log(10)

        # Filter outliers
        try:
            if i == 1:
                regressor = None
            elif i == 2:
                regressor = lm.RANSACRegressor()
            elif i == 3:
                regressor = lm.HuberRegressor()
            else:
                raise ValueError
        except ValueError:
            ax.plot(np.log10(X[:, None]), kde[:], 'k-*', lw=0.5)
            X_, Y_ = np.log10(X), kde
        else:
            X_, Y_, mask_ = detect_outliers(
                                    regressor,
                                    np.log10(X),
                                    kde
                                )

            # Print in/outliers
            ax.plot(np.log10(X[:, None]), kde[:], 'k-', lw=0.5)
            ax.plot(np.log10(X[~mask_, None]), kde[~mask_], '*', c='k')
            ax.plot(np.log10(X[mask_, None]), kde[mask_], '*', c='r')

        # RANSAC
        rs_estimator = lm.RANSACRegressor()
        rs_estimator.fit(X_[:, None], Y_)
        y_rs = rs_estimator.estimator_.predict(np.log10(X)[:, None])
        ax.plot(np.log10(X), y_rs, '-', label='RANSAC')

        # Theil Sen regression
        ts_model = lm.TheilSenRegressor(n_jobs=16)
        ts_model.fit(X_[:, None], Y_)
        y_ts = ts_model.predict(np.log10(X)[:, None])
        ax.plot(np.log10(X), y_ts, '--', label='Theil-Sen')

        # Huber regression
        hb_model = lm.HuberRegressor()
        hb_model.fit(X_[:, None], Y_)
        y_ts = hb_model.predict(np.log10(X)[:, None])
        ax.plot(np.log10(X), y_ts, '-.', label='Huber')

        # Labels
        ax.legend()
        if i == 0:
            ax.set_title("No Outliers")
        elif i == 1:
            ax.set_title("Linear Detection")
        elif i == 2:
            ax.set_title("RANSAC Outliers")
        elif i == 3:
            ax.set_title("Huber Outliers")
        else:
            raise("Subplot id not recognized")
        ax.set_xlim((0, 3.5))
        ax.set_ylim((-6.5, 0))

        ax.set_xlabel(r'$\log_{10}$ Counts')
        ax.set_ylabel(r'$\log_{10}$ Normalized Density')
    
    file_name = f'../png/outlier_detection.png'
    plib.save_fig(fig, file_name)

def main():
    src = '/users/loh/nodeSSD/repos/size_oscillation'
    case = 'BOMEX_12HR' # Case name 
    c_type = 'cloud' # Type of clouds (core or cloud)
    time = np.random.randint(540) # Pick a random time

    df = f'{src}/{case}_2d_{c_type}_counts_{time:03d}.pq'
    df = pq.read_pandas(df).to_pandas()

    plot_outliers(df)

if __name__ == '__main__':
    main()
