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

def calc_slope(counts):
    # Define grid
    X_ = np.logspace(0, np.log10(counts.max()), 50)

    # grid = GridSearchCV(
    #     KernelDensity(),
    #     {'bandwidth': np.linspace(1, 50, 50)},
    #     n_jobs=16)
    # grid.fit(df.counts[:, None])
    # print(grid.best_params_)
    
    log_kde = KernelDensity(bandwidth=2).fit(counts[:, None])
    kde = log_kde.score_samples(X_[:, None]) / np.log(10)
    
    # Initial regression
    ts_model = lm.TheilSenRegressor(n_jobs=16)
    ts_model.fit(np.log10(X_[:, None]), kde)
    y_ts = ts_model.predict(np.log10(X_[:, None]))

    # Sort by outliers
    outlier_list = np.argsort(kde - y_ts)

    # Maximum number of outliers (20%)
    max_iter = np.int(0.2 * 50)
    outliers = []
    for i in range(max_iter):
        # Update outlier list
        outliers += [outlier_list[i]]

        # Remove outlier and re-run regression
        X_new = np.delete(X_, outliers)
        kde_new = np.delete(kde, outliers)

        ts_model = lm.TheilSenRegressor(n_jobs=16)
        ts_model.fit(np.log10(X_new[:, None]), kde_new)
        
        corr = ts_model.score(np.log10(X_new[:, None]), kde_new)
        if corr >= 0.975:
            break

    # Ridge regression
    lr_model = lm.RidgeCV()
    lr_model.fit(np.log10(X_new[:, None]), kde_new[:])

    # Theil Sen regression
    ts_model = lm.TheilSenRegressor(n_jobs=16)
    ts_model.fit(np.log10(X_new[:, None]), kde_new)

    # Huber regression
    hb_model = lm.HuberRegressor(fit_intercept=False)
    hb_model.fit(np.log10(X_new[:, None]), kde_new)

    return lr_model.coef_, ts_model.coef_, hb_model.coef_

def main():
    src = '/users/loh/nodeSSD/repos/size_oscillation'

    slopes = []
    cols = ['lr', 'ts', 'hb']
    for time in np.arange(0, 540):
        pq_in = f'{src}/2d_cloud_counts_{time:03d}.pq'
        df = pq.read_pandas(pq_in).to_pandas()

        slopes += [calc_slope(df.counts)]
        df_out = pd.DataFrame(slopes, columns=cols)
        df_out.to_parquet(f'../pq/cloud_size_dist_slope.pq')

if __name__ == '__main__':
    main()
