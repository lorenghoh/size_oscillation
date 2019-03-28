import os, glob

import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit
from sklearn import linear_model as lm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def detect_outliers(regressor, x, y):
    # Initial regression
    regressor.fit(x[:, None], y)
    y_ts = regressor.predict(x[:, None])

    # Remove first five bins
    x = np.delete(x, [0, 1, 2, 3, 4])
    y = np.delete(y, [0, 1, 2, 3, 4])
    y_ts = np.delete(y_ts, [0, 1, 2, 3, 4])

    # Maximum number of outliers (10%)
    max_iter = np.int(0.1 * 50)

    # Sort by outliers
    outlier_list = np.argsort(np.abs(y - y_ts))[::-1][:max_iter]
    
    outliers = []
    for i in range(max_iter):
        # Update outlier list
        outliers += [outlier_list[i]]

        # Remove outlier and re-run regression
        X_ = np.delete(x, outliers)
        Y_ = np.delete(y, outliers)

        regressor.fit(X_[:, None], Y_)
        corr = regressor.score(X_[:, None], Y_)
        # Stop at threshold correlation score
        if corr >= 0.975:
            break

    return X_, Y_

def main():
    src = '/users/loh/nodeSSD/repos/size_oscillation'

    for time in np.arange(0, 540, 5):
        #---- Plotting 
        fig = plt.figure(1, figsize=(5, 4))
        fig.clf()
        sns.set_context('paper')
        sns.set_style('ticks', 
            {
                'axes.grid': False,
                'axes.linewidth': '0.75',
                'grid.color': '0.75',
                'grid.linestyle': u':',
                'legend.frameon': True,
            })
        plt.rc('text', usetex=True)
        plt.rc('font', family='Serif')

        ax = plt.subplot(1, 1, 1)

        case = 'BOMEX_BOWL' # Case name 
        c_type = 'core' # Type of clouds (core or cloud)

        df = f'{src}/{case}_2d_{c_type}_counts_{time:03d}.pq'
        df = pq.read_pandas(df).to_pandas()

        # Define grid
        X_ = np.logspace(0, np.log10(df.max()), 50)
        
        log_kde = KernelDensity(bandwidth=1).fit(df.counts[:, None])
        kde = log_kde.score_samples(X_[:, None]) / np.log(10)

        plt.plot(np.log10(X_[:, None]), kde, '-+')

        # Filter outliers
        # reg_model = lm.TheilSenRegressor(n_jobs=16)
        reg_model = lm.HuberRegressor(fit_intercept=False)
        X_, Y_ = detect_outliers(
            reg_model,
            np.log10(X_),
            kde)

        # Ridge regression
        reg_model = lm.RidgeCV()
        reg_model.fit(X_[:, None], Y_[:])
        y_reg = reg_model.predict(X_[:, None])
        plt.plot(X_[:], y_reg, '--',
                label='Linear Regression')

        # Theil Sen regression
        ts_model = lm.TheilSenRegressor(n_jobs=16)
        ts_model.fit(X_[:, None], Y_)
        y_ts = ts_model.predict(X_[:, None])
        plt.plot(X_, y_ts, '-', c='k',
                label='Theil-Sen Regression')

        # Huber regression
        hb_model = lm.HuberRegressor(fit_intercept=False)
        hb_model.fit(X_[:, None], Y_)
        y_ts = hb_model.predict(X_[:, None])
        plt.plot(X_, y_ts, '-.', c='b',
                label='Huber Regression')

        print(reg_model.coef_, ts_model.coef_, hb_model.coef_)

        # Labels
        plt.legend()
        plt.xlim((0, 3.5))
        plt.ylim((-6.5, 0))

        plt.xlabel(r'$\log_{10}$ Counts')
        plt.ylabel(r'$\log_{10}$ Normalized Density')
        
        plt.tight_layout(pad=0.5)
        figfile = f'../png/temp/cloud_size_dist_{time:03d}.png'
        print('\t Writing figure to {}...'.format(figfile))
        plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                    facecolor='w', transparent=True)

if __name__ == '__main__':
    main()
