import os, glob

import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import linear_model as lm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, \
                                            ExpSineSquared, RationalQuadratic

def main():
    df = pq.read_pandas('../pq/cloud_size_dist_slope.pq').to_pandas()

    kernel = 1.0 * RBF(length_scale_bounds=(1e3, 1e5)) \
            + 1.0 * WhiteKernel(noise_level=1) \
            + 1.0 * ExpSineSquared(
                length_scale_bounds=(0.1, 10),   
                periodicity=80, 
                periodicity_bounds=(70, 90)) \
            + 1.0 * ExpSineSquared(
                length_scale_bounds=(0.1, 10), 
                periodicity=40, 
                periodicity_bounds=(30, 50))
    gp = GaussianProcessRegressor(kernel=kernel,
                                  normalize_y=False,
                                  n_restarts_optimizer=15)

    X_ = np.arange(540)
    gp.fit(X_[:, None], np.array(df.slope))

    #---- Plotting 
    fig = plt.figure(1, figsize=(10, 4))
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

    plt.plot(X_, df.slope, '--*')
    
    X = np.linspace(1, 540, 540)
    y_mean, y_std = gp.predict(X[:, None], return_std=True)
    plt.plot(X, y_mean, 'k', lw=1, zorder=9)
    plt.fill_between(X, y_mean - y_std, y_mean + y_std,
                     alpha=0.2, color='k')
    print(gp.kernel_)
    print(gp.log_marginal_likelihood_value_)

    y_samples = gp.sample_y(X[:, None], 10)
    plt.plot(X, y_samples, lw=0.5, alpha=0.2)

    plt.xlabel(r'Time [min]')
    plt.ylabel(r'Slope $b$')
    
    plt.tight_layout(pad=0.5)
    figfile = '../png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                facecolor='w', transparent=True)

if __name__ == '__main__':
    main()