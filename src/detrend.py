import os, glob

import numpy as np
import numba as nb
import joblib as jb
import pandas as pd
import pyarrow.parquet as pq 

import lib_plot as plib

from statsmodels.nonparametric import smoothers_lowess as sl

from sklearn import linear_model as lm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as kern

"""
Detrend timeseries, using Kernel regression
"""
def detrend():
    pass

def main():
    case = 'BOMEX_SWAMP' # Case name 
    c_type = 'cloud' # Type of clouds (core or cloud)
    reg_model = 'ts' # See get_linear_reg.py for choices

    file_name = f'../pq/{case}_{c_type}_size_dist_slope.pq'
    df = pq.read_pandas(file_name).to_pandas()

    X_ = np.arange(720)
    Y_ = np.array(df[f'{reg_model}'])
    Y_ls = sl.lowess(Y_, X_, return_sorted=False, frac=.3)

    #---- Plotting 
    fig = plib.init_plot()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(X_, Y_, '--*', zorder=9)
    ax1.plot(X_, Y_ls, '-')

    ax2.plot(X_, Y_ - Y_ls, '--*')

    # Plot labels
    ax1.set_xlabel(r'Time [min]')
    ax1.set_ylabel(r'Slope $b$')
    
    file_name = f'../png/{os.path.splitext(__file__)[0]}.png'
    plib.save_fig(fig, file_name)

if __name__ == '__main__':
    main()