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

import get_linear_reg

def plot_kde():
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
        df = pq.read_pandas(f'{src}/2d_cloud_counts_{time:03d}.pq').to_pandas()

        # Define grid
        X_ = np.linspace(1, df.max(), 100)

        counts = df.counts[df.counts > 1]
        
        log_kde = KernelDensity(bandwidth=20).fit(counts[:, None])
        kde = np.log10(np.exp(log_kde.score_samples(X_[:, None])))

        plt.plot(np.log10(X_), kde, '-+')

        # Ridge regression
        reg_model = lm.RidgeCV()
        reg_model.fit(np.log10(X_[:, None]), kde[:])

        y_reg = reg_model.predict(np.log10(X_)[:, None])
        plt.plot(np.log10(X_[:]), y_reg, '--',
                label='Linear Regression')

        # Theil Sen regression
        ts_model = lm.TheilSenRegressor(n_jobs=16)
        ts_model.fit(np.log10(X_[:, None]), kde)
        y_ts = ts_model.predict(np.log10(X_)[:, None])
        plt.plot(np.log10(X_), y_ts, '-', c='k',
                label='Theil-Sen Regression')

        # Huber regression
        hb_model = lm.HuberRegressor(fit_intercept=False)
        hb_model.fit(np.log10(X_[:, None]), kde)
        y_ts = hb_model.predict(np.log10(X_)[:, None])
        plt.plot(np.log10(X_), y_ts, '-.', c='b',
                label='Huber Regression')

        print(reg_model.coef_, ts_model.coef_, hb_model.coef_)

        # Labels
        plt.legend()
        plt.ylim(top=-1, bottom=-6)

        plt.xlabel(r'$\log_{10}$ Counts')
        plt.ylabel(r'$\log_{10}$ Normalized Density')
        
        plt.tight_layout(pad=0.5)
        # figfile = '../png/{}_BOMEX_12HR.png'.format(os.path.splitext(__file__)[0])
        figfile = f'../png/temp/cloud_size_dist_{time:03d}.png'
        print('\t Writing figure to {}...'.format(figfile))
        plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                    facecolor='w', transparent=True)

if __name__ == '__main__':
    plot_kde()
