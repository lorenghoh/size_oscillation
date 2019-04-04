import os, glob

import numpy as np
import numba as nb
import joblib as jb
import pandas as pd
import pyarrow.parquet as pq 

import lib_plot as plib

from sklearn import linear_model as lm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as kern

def main():
    case = 'BOMEX_BOWL' # Case name 
    c_type = 'cloud' # Type of clouds (core or cloud)
    reg_model = 'ts' # See get_linear_reg.py for choices

    file_name = f'../pq/{case}_{c_type}_size_dist_slope.pq'
    df = pq.read_pandas(file_name).to_pandas()

    kernel = kern.RBF(length_scale_bounds=(1e1, 1e4)) \
            + 1.0 * kern.WhiteKernel(noise_level=1) \
            + 1.0 * kern.ExpSineSquared(
                        length_scale=1e2,
                        length_scale_bounds=(1e1, 1e4),
                        periodicity=80,
                        periodicity_bounds=(65, 95)
                    ) \
            # + 1.0 * kern.ExpSineSquared(
            #             length_scale=100,
            #             length_scale_bounds=(1, 1e3),
            #             periodicity=40, 
            #             periodicity_bounds=(25, 55)
            #         )
    gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5)

    X_ = np.arange(540)
    gp.fit(X_[:, None], np.array(df[f'{reg_model}']))

    #---- Plotting 
    fig = plib.init_plot()
    ax = fig.add_subplot(111)

    ax.plot(X_, df[f'{reg_model}'], '--*', zorder=9)
    
    X = np.linspace(1, 540, 540) # Training scope
    # X = np.linspace(1, 720, 2880) # Predictions
    y_mean, y_std = gp.predict(X[:, None], return_std=True)
    ax.plot(X, y_mean, 'k', lw=1, zorder=9)
    ax.fill_between(X, y_mean - y_std, y_mean + y_std,
                     alpha=0.2, color='k')
    print(gp.kernel_)
    print(gp.log_marginal_likelihood_value_)

    y_samples = gp.sample_y(X[:, None], 10)
    ax.plot(X, y_samples, lw=0.5, alpha=0.2)

    # Plot labels
    ax.set_xlabel(r'Time [min]')
    ax.set_ylabel(r'Slope $b$')
    
    file_name = f'../png/{os.path.splitext(__file__)[0]}.png'
    plib.save_fig(fig, file_name)

if __name__ == '__main__':
    main()