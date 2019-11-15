import os, sys, glob

import numpy as np
import numba as nb

import dask.array as da
import dask.dataframe as dd
from dask.delayed import delayed

import xarray as xr
import pandas as pd
import pyarrow.parquet as pq

import scipy.ndimage.measurements as measure
import scipy.ndimage.morphology as morph

from tqdm import tqdm
from pathlib import Path

import lib.calcs as calcs
import lib.index as index

def sample_conditional_field(ds):
    """
    Define conditional fields

    Return
    ------
    c0_fld: cloud field (QN > 0)
    c1_fld: core field (QN > 0, W > 0, B > 0)
    c2_fld: plume (tracer-dependant)
    """
    th_v = calcs.theta_v(
        ds['p'][:] * 100,
        ds['TABS'][:],
        ds['QV'][:] / 1e3,
        ds['QN'][:] / 1e3,
        ds['QP'][:] / 1e3)

    buoy = (th_v > np.mean(th_v, axis=(1, 2)))

    c0_fld = (ds['QN'] > 0)
    c1_fld = (buoy & c0_fld)

    # Define plume based on tracer fields (computationally intensive)
    # tr_field = ds['TR01'][:]
    # tr_ave = np.nanmean(tr_field, axis=(1, 2))
    # tr_std = np.std(tr_field, axis=(1, 2))
    # tr_min = .05 * np.cumsum(tr_std) / (np.arange(len(tr_std))+1)

    # c2_fld = (tr_field > \
    #             np.max(np.array([tr_ave + tr_std, tr_min]), 0)[:, None, None])

    return c0_fld, c1_fld

def cluster_clouds():
    """
    Clustering the cloud volumes/projections from the netcdf4 dataset.

    There are a few different ways to cluster cloud/core volumes. The possible
    choices are: 2d cluster (no clustering), ~~3d_volume (full cloud volume)~~,
    3d_projection (projections of 3D volumes), 2d_projection (projections of the
    entire cloud field), etc.

    Clustering options:
    - 2d: 2D cluster (horizontal slices)
    - projection: 3D projection

    Return
    ------
    Parquet files containing the coordinates
    """
    # TODO: Use ModelConfig class for model parameters
    case_name = 'BOMEX_SWAMP'

    src = Path(f'/Howard16TB/data/loh/{case_name}/variables')
    dest = '/users/loh/nodeSSD/temp'

    def write_clusters(ds, src, dest):
        bin_st = morph.generate_binary_structure(3, 2)
        bin_st[0] = 0
        bin_st[-1] = 0

        c0_fld, c1_fld = sample_conditional_field(ds)

        df = pd.DataFrame(columns = ['coord', 'cid', 'type'])
        for item in [0, 1]:
            c_field = locals()[f'c{item}_fld']

            c_label, n_features = measure.label(c_field, structure=bin_st)
            c_label = c_label.ravel() # Sparse array

            # Extract indices
            c_index = np.arange(len(c_label))
            c_index = c_index[c_label > 0]
            c_label = c_label[c_label > 0]

            if item == 0:
                c_type = np.ones(len(c_label), dtype=int)
            elif item == 1:
                c_type = np.zeros(len(c_label), dtype=int)
            else:
                raise TypeError
            
            df_ = pd.DataFrame.from_dict(
                    {
                        'coord': c_index,
                        'cid': c_label,
                        'type': c_type,
                    })
            df = pd.concat([df, df_])
        
        file_name = f"{dest}/{case_name}_{time:04d}.pq"
        df.to_parquet(file_name)
        
        tqdm.write(f"Written {file_name}")
    
    fl = sorted(src.glob('*'))
    for time, item in enumerate(tqdm(fl)):
        fname = item.as_posix()
        if item.suffix == '.nc':
            with xr.open_dataset(fname) as ds:
                try:
                    ds = ds.squeeze('time')
                except:
                    pass
        elif item.suffix == '.zarr':
            with xr.open_zarr(fname) as ds:
                try:
                    ds = ds.squeeze('time')
                except:
                    pass
        else:
            print("Error: File type not recognized.")
            raise Exception
        
        write_clusters(ds, src, dest)

if __name__ == '__main__':
    cluster_clouds()
