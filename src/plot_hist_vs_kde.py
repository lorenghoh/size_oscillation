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
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

def plot_kde():
    src = '/users/loh/nodeSSD/repos/size_oscillation'

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
    df = pq.read_pandas(f'{src}/2d_cloud_counts_470.pq').to_pandas()

    # Define grid for Kernel Density Estimate
    X_ = np.linspace(1, df.max(), 400)
    bins = np.linspace(1, df.max(), 80)

    log_kde = KernelDensity(bandwidth=40).fit(df.counts[:, None])
    kde = np.exp(log_kde.score_samples(X_[:, None]))

    plt.plot(X_, kde, '-+', label='Gaussian KDE')
    plt.hist(df.counts, bins=bins, log=True,
             histtype=u'step', density=True, label='Histogram')

    # Labels
    plt.legend()
    ax.set_xscale('log')

    ax.set_xlabel(r'Area')
    ax.set_ylabel(r'Normalized Density')
    
    plt.tight_layout(pad=0.5)
    figfile = '../png/{}_BOMEX_BOWL.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                facecolor='w', transparent=True)

if __name__ == '__main__':
    plot_kde()
