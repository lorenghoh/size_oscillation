import os, glob

import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

import lib_plot as plib

def main():
    case = 'BOMEX_BOWL' # Case name 
    c_type = 'cloud'

    df = pq.read_pandas(f'../pq/{case}_{c_type}_size_dist_slope.pq')
    df = df.to_pandas()
    
    x = np.arange(0, 540)

    #---- Plotting 
    fig = plib.init_plot()
    ax = fig.add_subplot(111)
    
    ax.plot(x, df.lr, 'k-', lw=0.75,
            label='Linear Regression')
    ax.plot(x, df.ts, '-*', lw=0.75,
            label='Theil-Sen Regression')
    ax.plot(x, df.hb, '--', lw=0.75,
            label='Huber Regression')

    ax.set_xlabel(r'Time [min]')
    ax.set_ylabel(r'Slope $b$')

    ax.legend()
    
    file_name = f'../png/{os.path.splitext(__file__)[0]}.png'
    plib.save_fig(fig, file_name)

if __name__ == '__main__':
    main()