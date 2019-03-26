import os, sys, glob

import numpy as np
import numba as nb
import xarray as xr
import pandas as pd

import pyarrow.parquet as pq
import scipy.ndimage.measurements as measure
import scipy.ndimage.morphology as morph

import var_calcs as vc

def sample_conditional_field(ds):
    th_v = vc.theta_v(
        ds['p'][:]*100,
        ds['TABS'][:],
        ds['QV'][:]/1e3,
        ds['QN'][:]/1e3,
        ds['QP'][:]/1e3)
        
    buoy_field = (th_v > np.mean(th_v, axis=(1, 2)))

    cloud_field = (ds['QN'] > 0)
    core_field = buoy_field & cloud_field

    return core_field, cloud_field

@nb.jit(['i8[:](i4[:], i4)'], nogil=True, nopython=True)
def count_features(labels, n_features):
    c_ = np.zeros(n_features, dtype=np.int_)

    for i in range(len(labels)):
        if  labels[i]> 0:
            c_[labels[i]] += 1
    return c_

def cluster_clouds():
    # TODO: read from model_config.json dict
    src = '/Howard16TB/data/loh/BOMEX_12HR/variables'
    dest = '/users/loh/nodeSSD/repos/size_oscillation'

    bin_struct = morph.generate_binary_structure(3, 1)
    # Label 2D structures 
    bin_struct[0] = 0
    bin_struct[-1] = 0

    nc_list = sorted(glob.glob(f'{src}/*.nc'))
    for time, item in enumerate(nc_list):
        with xr.open_dataset(item, autoclose=True) as ds:
            try:
                ds = ds.squeeze('time')
            except:
                pass

            core_field, cloud_field = sample_conditional_field(ds)
            for item in ['core', 'cloud']:
                c_field = locals()[f'{item}_field']

                label, n_features = measure.label(c_field, structure=bin_struct)
                label = label.ravel()

                f_counts = count_features(label, n_features)
                df = pd.DataFrame(f_counts, columns=['counts'])

                file_name = f'{dest}/2d_{item}_counts_{time:03d}.pq'
                df.to_parquet(file_name)

                print(f"Written {file_name}")

if __name__ == '__main__':
    cluster_clouds()
