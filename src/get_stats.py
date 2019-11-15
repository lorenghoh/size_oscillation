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
    case = 'BOMEX_SWAMP'
    src = f'/Howard16TB/data/loh/'

    stat_file_n = f"{src}/{case}/BOMEX_1536x512x128_25m_1s_ent_ocean_50m.nc"

    CAPE = get_stats(stat_file_n, 'CAPE')
    LWP = get_stats(stat_file_n, 'LWP')

    #---- Plotting 
    fig = plib.init_plot()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ts = np.arange(len(CAPE)) / 60
    ax1.plot(ts, CAPE, '-', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2.plot(ts, LWP, '*-')
    ax2.tick_params(axis='y')

    # Labels
    # ax.legend()

    ax1.set_xlabel(r'Time [hr]')
    ax1.set_ylabel(r'CAPE [g/m2]')
    ax2.set_ylabel(r'LWP [W/m2]')
    
    file_name = f'../png/{case}_stat_CAPE.png'
    plib.save_fig(fig, file_name)