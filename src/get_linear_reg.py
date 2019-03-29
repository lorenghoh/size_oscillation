import os, glob

import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

# Scikitgit 
from sklearn import linear_model as lm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from get_outliers import detect_outliers

def calc_slope(counts):
    # Define grid
    X_ = np.logspace(0, np.log10(counts.max()), 50)

    # grid = GridSearchCV(
    #     KernelDensity(),
    #     {'bandwidth': np.linspace(1, 50, 50)},
    #     n_jobs=16)
    # grid.fit(df.counts[:, None])
    # print(grid.best_params_)
    
    log_kde = KernelDensity(bandwidth=1).fit(counts[:, None])
    kde = log_kde.score_samples(X_[:, None]) / np.log(10)

    # Filter outliers
    # reg_model = lm.TheilSenRegressor(n_jobs=16)
    reg_model = lm.HuberRegressor(fit_intercept=False)
    X_, Y_ = detect_outliers(
        reg_model,
        np.log10(X_),
        kde)

    # Ridge regression
    lr_model = lm.RidgeCV()
    lr_model.fit(X_[:, None], Y_[:])

    # Theil Sen regression
    ts_model = lm.TheilSenRegressor(n_jobs=16)
    ts_model.fit(X_[:, None], Y_)

    # Huber regression
    hb_model = lm.HuberRegressor(fit_intercept=False)
    hb_model.fit(X_[:, None], Y_)

    return lr_model.coef_, ts_model.coef_, hb_model.coef_

def main():
    src = '/users/loh/nodeSSD/repos/size_oscillation'

    # TODO: Re-do cloud slopes
    c_type = 'core'
    case = 'BOMEX_BOWL'

    slopes = []
    cols = ['lr', 'ts', 'hb']
    for time in tqdm(np.arange(0, 540)):
        pq_in = f'{src}/{case}_2d_{c_type}_counts_{time:03d}.pq'
        df = pq.read_pandas(pq_in).to_pandas()

        # Supress output
        # slopes += [calc_slope(df.counts)]
        lr, ts, hb = calc_slope(df.counts)
        slopes += [(lr, ts, hb)]
        tqdm.write(
            f'{time:3d}, lr = {lr[0]:.3f}, ts = {ts[0]:.3f}, hb = {hb[0]:.3f}'
        )
    df_out = pd.DataFrame(slopes, columns=cols)
    df_out.to_parquet(f'../pq/{case}_{c_type}_size_dist_slope.pq')

if __name__ == '__main__':
    main()
