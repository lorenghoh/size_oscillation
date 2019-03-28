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
from sklearn.gaussian_process import kernels as kern

def main():
    case = 'BOMEX_BOWL' # Case name 
    c_type = 'core' # Type of clouds (core or cloud)
    reg_model = 'ts' # See get_linear_reg.py for choices

    df = pq.read_pandas(f'../pq/{case}_{c_type}_size_dist_slope.pq')
    df = df.to_pandas()

    # kernel = 1.0 * kern.RBF(length_scale_bounds=(1e2, 1e4)) \
    kernel = 1.0 * kern.() \
            + 1.0 * kern.WhiteKernel(noise_level=1) \
            + 1.0 * kern.ExpSineSquared(
                        length_scale=1,
                        periodicity=80, 
                        periodicity_bounds=(70, 90)
                    ) \
            # + 1.0 * kern.ExpSineSquared(
            #             length_scale=1,
            #             periodicity=40, 
            #             periodicity_bounds=(30, 50)
            #         )
    gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5)

    X_ = np.arange(540)
    gp.fit(X_[:, None], np.array(df[f'{reg_model}']))

    #---- Plotting 
    fig = plt.figure(1, figsize=(12, 4))
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

    plt.plot(X_, df[f'{reg_model}'], '--*', zorder=9)
    
    X = np.linspace(1, 720, 2880)
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