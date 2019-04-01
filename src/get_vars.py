import os, glob

import numpy as np
import numba as nb
import xarray as xr
import pandas as pd
import pyarrow.parquet as pq 

from tqdm import tqdm

# Read netCDF output for domain properties and print them into DataFrame objects

def main():
    # TODO: read from model_config.json dict
    case = 'BOMEX_BOWL'

    src = f'/Howard16TB/data/loh/{case}/variables'
    dest = '/users/loh/nodeSSD/repos/size_oscillation'

    # Initialize lists
    qp_tot_list = []
    qp_sum_list = []

    nc_list = sorted(glob.glob(f'{src}/*.nc'))
    for time, item in enumerate(tqdm(nc_list[:540])):
        with xr.open_dataset(item, autoclose=True) as ds:
            try:
                ds = ds.squeeze('time')
            except:
                pass

        qp_tot = np.sum(np.array(ds.QP))
        qp_tot_list += [qp_tot]
        qp_sum_list += [qp_tot / len(ds.x) / len(ds.y)]

    dataset = {
        'qp_tot': qp_tot_list,
        'qp_sum': qp_sum_list}
    df = pd.DataFrame.from_dict(dataset)

    file_name = f'{dest}/{case}_vars.pq'
    df.to_parquet(file_name)

if __name__ == '__main__':
    main()