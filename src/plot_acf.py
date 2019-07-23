import os, glob

import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq

import lib_plot as plib
import seaborn as sns

import statsmodels.tsa.stattools as st
import statsmodels.graphics.tsaplots as tp

def main():
    case = 'BOMEX_12HR' # Case name 
    c_type = 'core_2d'

    #---- Plotting 
    fig = plib.init_plot(figsize=(6, 8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    x = np.arange(0, 540)

    # Plot timeseries
    file_name = f'../pq/{case}_{c_type}_size_dist_slope.pq'
    df = pq.read_pandas(file_name).to_pandas()

    # ax.plot(x, -df.lr, 'k-', lw=0.75,
    #         label='Linear Regression')
    # ax.plot(x, -df.ts, '-*', lw=0.75,
    #         label='Theil-Sen Regression')
    # ax.plot(x, -df.hb, '--', lw=0.75,
    #         label='Huber Regression')

    ds = pd.Series(np.array(df.rs, dtype=np.float32))
    tp.plot_acf(ds, ax=ax2, lags=90)
    tp.plot_pacf(ds, ax=ax1, lags=90)

    file_name = f'../png/{os.path.splitext(__file__)[0]}.png'
    plib.save_fig(fig, file_name)

if __name__ == '__main__':
    main()