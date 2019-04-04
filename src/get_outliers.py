import os, glob

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

def detect_outliers(regressor, x, y):
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
    
    max_iter = 15
    for i in range(max_iter):
        # Remove outlier and re-run regression
        x_ = x_ma.compressed()
        y_ = y_ma.compressed()

        regressor.fit(x_[:, None], y_)
        corr = regressor.score(x_[:, None], y_)
        print(corr)

        # Stop at threshold correlation score
        # Otherwise, filter next outlier
        if np.isclose(corr, 0.99, atol=1e-3):
            break
        else:
            outlier = find_linear_outlier(regressor, x_ma, y_ma)
            x_ma.mask[outlier] = True
            y_ma.mask[outlier] = True
    if corr <= 0.975:
        raise ValueError("Given data cannot be linearized")
    
    return x_, y_

def find_linear_outlier(regressor, x_ma, y_ma):
    x_ = x_ma.compressed()
    y_ = y_ma.compressed()

    regressor.fit(x_[:, None], y_)
    y_reg = regressor.predict(x_ma.data[:, None])
    y_reg = np.ma.masked_array(y_reg, mask=y_ma.mask)

    # Sort by outliers (Requires Numpy 1.16)
    outlier = np.abs(y_ma - y_reg).argsort(endwith=False)[-1]
    return outlier

def compress_inliers(is_outlier, x, y):
    x_ = np.compress(~is_outlier, x)
    y_ = np.compress(~is_outlier, y)
    return x_, y_

def find_RANSAC_outlier(regressor, x, y):
    regressor.fit(x[:, None], y)
    return ~regressor.inlier_mask_

def find_huber_outlier(regressor, x, y):
    regressor.fit(x[:, None], y)
    return regressor.outliers_

def main():
    src = '/users/loh/nodeSSD/repos/size_oscillation'
    case = 'BOMEX_12HR' # Case name 
    c_type = 'cloud' # Type of clouds (core or cloud)
    time = np.random.randint(540) # Pick a random time

    df = f'{src}/{case}_2d_{c_type}_counts_{time:03d}.pq'
    df = pq.read_pandas(df).to_pandas()

    # TODO: 4-pane plots for different schemes
    #---- Plotting 
    fig = plib.init_plot((5, 4))
    ax = fig.add_subplot(111)

    # Define grid
    X_ = np.logspace(0, np.log10(df.max()), 50)
    
    log_kde = KernelDensity(bandwidth=1).fit(df.counts[:, None])
    kde = log_kde.score_samples(X_[:, None]) / np.log(10)

    ax.plot(np.log10(X_[:, None]), kde, '-+')

    # Filter outliers
    X_, Y_ = detect_outliers(
        None,
        np.log10(X_),
        kde)

    # Ridge regression
    reg_model = lm.RidgeCV()
    reg_model.fit(X_[:, None], Y_[:])
    y_reg = reg_model.predict(X_[:, None])
    ax.plot(X_[:], y_reg, '-', c='k',
            label='Linear Regression')

    # RANSAC
    rs_estimator = lm.RANSACRegressor()
    rs_estimator.fit(X_[:, None], Y_)
    y_rs = rs_estimator.estimator_.predict(X_[:, None])
    ax.plot(X_, y_rs, '-',
            label='RANSAC Estimator')

    # Theil Sen regression
    ts_model = lm.TheilSenRegressor(n_jobs=16)
    ts_model.fit(X_[:, None], Y_)
    y_ts = ts_model.predict(X_[:, None])
    ax.plot(X_, y_ts, '--',
            label='Theil-Sen Regression')

    # Huber regression
    hb_model = lm.HuberRegressor(fit_intercept=False)
    hb_model.fit(X_[:, None], Y_)
    y_ts = hb_model.predict(X_[:, None])
    ax.plot(X_, y_ts, '-.',
            label='Huber Regression')

    # Labels
    ax.legend()
    ax.set_xlim((0, 3.5))
    ax.set_ylim((-6.5, 0))

    ax.set_xlabel(r'$\log_{10}$ Counts')
    ax.set_ylabel(r'$\log_{10}$ Normalized Density')
    
    file_name = f'../png/outlier_detection.png'
    plib.save_fig(fig, file_name)

if __name__ == '__main__':
    main()
