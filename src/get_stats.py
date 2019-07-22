import os, glob

import numpy as np
import numba as nb
import xarray as xr
import pandas as pd
import pyarrow.parquet as pq 

from tqdm import tqdm

import lib_plot as plib
import seaborn as sns

# Read netCDF output for domain properties and print them into DataFrame objects

def get_stats(stat_file_n, target):
    with xr.open_dataset(stat_file_n, autoclose=True) as ds:
        return ds[target][:]

if __name__ == '__main__':
    # TODO: read from model_config.json dict
    case = 'BOMEX_24HR'

    src = f'/Howard16TB/data/loh/{case}/OUT_STAT/'

    stat_file_n = src + 'BOMEX_2880x768x128_25m_25m_1s.nc'
    target = 'CAPE'

    ds = get_stats(stat_file_n, target)

    #---- Plotting 
    fig = plib.init_plot()
    ax = fig.add_subplot(111)

    ts = np.arange(len(ds)) / 60
    # ax.plot(ts, np.sum(ds[:], axis=1))
    ax.plot(ts, ds)

    # Labels
    # ax.legend()

    ax.set_xlabel(r'Time [hr]')
    ax.set_ylabel(r'CAPE [J/kg]')
    
    file_name = f'../png/{case}_stat_{target}'
    plib.save_fig(fig, file_name)