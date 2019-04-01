import os, glob

import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

import lib_plot as plib
import seaborn as sns

def main():
    case = 'BOMEX_BOWL' # Case name 
    c_type = 'cloud'

    #---- Plotting 
    fig = plib.init_plot()
    ax = fig.add_subplot(111)
    
    x = np.arange(0, 540)

    # Plot timeseries
    file_name = f'../pq/{case}_{c_type}_size_dist_slope.pq'
    df = pq.read_pandas(file_name).to_pandas()

    # ax.plot(x, -df.lr, 'k-', lw=0.75,
    #         label='Linear Regression')
    ax.plot(x, -df.ts, '-*', lw=0.75,
            label='Theil-Sen Regression')
    # ax.plot(x, -df.hb, '--', lw=0.75,
    #         label='Huber Regression')

    ax.set_xlabel(r'Time [min]')
    ax.set_ylabel(r'Slope $b$')
    ax.legend()

    # Vars on the second scale
    file_name = f'../pq/{case}_vars.pq'
    df2 = pq.read_pandas(file_name).to_pandas()

    ax2 = ax.twinx()
    ax2.plot()

    ax2.plot(x, df2.qp_ave, 'k-', lw=0.75)

    ax2.set_ylabel(r'Ave. Precip. [g/kg]')
    
    file_name = f'../png/{os.path.splitext(__file__)[0]}.png'
    plib.save_fig(fig, file_name)

if __name__ == '__main__':
    main()