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
    X_ = np.linspace(5, df.max(), 400)

    # grid = GridSearchCV(
    #     KernelDensity(),
    #     {'bandwidth': np.linspace(1, 50, 50)},
    #     n_jobs=16)
    # grid.fit(df.counts[:, None])
    # print(grid.best_params_)

    counts = np.array(df.counts[df.counts > 5])
    
    log_kde = KernelDensity(bandwidth=15).fit(counts[:, None])
    kde = np.log10(np.exp(log_kde.score_samples(X_[:, None])))

    plt.plot(np.log10(X_), kde, '-+')

    # Ridge regression
    reg_model = lm.RidgeCV()
    reg_model.fit(np.log10(X_[100:, None]), kde[100:])

    y_reg = reg_model.predict(np.log10(X_)[100:, None])
    plt.plot(np.log10(X_[100:]), y_reg, '--',
            label='Linear Regression')

    # Theil Sen regression
    ts_model = lm.TheilSenRegressor(n_jobs=16)
    ts_model.fit(np.log10(X_[:, None]), kde)
    y_ts = ts_model.predict(np.log10(X_)[:, None])
    plt.plot(np.log10(X_), y_ts, '-', c='k',
            label='Theil-Sen Regression')

    print(reg_model.coef_, ts_model.coef_)

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
        X_ = np.linspace(5, df.max(), 200)

        # grid = GridSearchCV(
        #     KernelDensity(),
        #     {'bandwidth': np.linspace(1, 50, 50)},
        #     n_jobs=16)
        # grid.fit(df.counts[:, None])
        # print(grid.best_params_)

        counts = np.array(df.counts[df.counts > 5])
        
        log_kde = KernelDensity(bandwidth=10).fit(counts[:, None])
        kde = np.log10(np.exp(log_kde.score_samples(X_[:, None])))

        plt.plot(np.log10(X_), kde, '-+')

        # Ridge regression
        reg_model = lm.RidgeCV()
        reg_model.fit(np.log10(X_[50:, None]), kde[50:])

        y_reg = reg_model.predict(np.log10(X_)[50:, None])
        plt.plot(np.log10(X_[50:]), y_reg, '--',
                label='Linear Regression')

        # Theil Sen regression
        ts_model = lm.TheilSenRegressor(n_jobs=16)
        ts_model.fit(np.log10(X_[:, None]), kde)
        y_ts = ts_model.predict(np.log10(X_)[:, None])
        plt.plot(np.log10(X_), y_ts, '-', c='k',
                label='Theil-Sen Regression')

        print(reg_model.coef_, ts_model.coef_)

        # Labels
        plt.legend()
        ax.set_xscale('log')
        plt.ylim(top=-1, bottom=-6)

        plt.xlabel(r'$\log_{10}$ Area')
        plt.ylabel(r'$\log_{10}$ Normalized Density')
        
        plt.tight_layout(pad=0.5)
        # figfile = '../png/{}_BOMEX_12HR.png'.format(os.path.splitext(__file__)[0])
        figfile = f'../png/temp/cloud_size_dist_{time:03d}.png'
        print('\t Writing figure to {}...'.format(figfile))
        plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                    facecolor='w', transparent=True)

if __name__ == '__main__':
    plot_kde()
