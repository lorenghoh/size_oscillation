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
    X_ = np.linspace(1, np.max(counts), 100)

    # grid = GridSearchCV(
    #     KernelDensity(),
    #     {'bandwidth': np.linspace(1, 50, 50)},
    #     n_jobs=16)
    # grid.fit(df.counts[:, None])
    # print(grid.best_params_)

    # TODO: examine the impact of this parameter
    counts = counts[counts > 1] 
    
    log_kde = KernelDensity(bandwidth=20).fit(counts[:, None])
    kde = np.log10(np.exp(log_kde.score_samples(X_[:, None])))

    # Theil Sen regression
    ts_model = lm.TheilSenRegressor(n_jobs=16)
    ts_model.fit(np.log10(X_[:, None]), kde)

    return ts_model.coef_

def main():
    src = '/users/loh/nodeSSD/repos/size_oscillation'

    slopes = []
    for time in np.arange(0, 540):
        pq_in = f'{src}/2d_cloud_counts_{time:03d}.pq'
        df = pq.read_pandas(pq_in).to_pandas()

        slopes += [calc_slope(df.counts)]
        df_out = pd.DataFrame(slopes, columns=['slope'])
        df_out.to_parquet(f'../pq/cloud_size_dist_slope.pq')

if __name__ == '__main__':
    main()
