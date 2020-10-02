import numpy as np

import pyarrow.parquet as pq

from pathlib import Path
from tqdm import tqdm

import lib.config
import case


config = lib.config.read_config()
pwd = Path(config['pwd'])
src = Path(config['case']) / 'clusters'

model = case.Model()


def get_pq(pq_path):
    df = pq.read_pandas(pq_path).to_pandas()
    df['z'] = df.coord // (model.nx * model.ny)

    return df


def cloud_dz(pq_path):
    df = get_pq(pq_path)
    df = df.groupby(['cid', 'z']).size().reset_index(name='counts')

    return df.counts.value_counts().values


def cloud_projection(pq_path):
    df = get_pq(pq_path)
    # TODO: Implement 2D projection of clouds

    return df


def cloud_volume(pq_path):
    df = get_pq(pq_path)

    return df.cid.value_counts().values


if __name__ == '__main__':
    cluster_list = sorted(src.glob('*'))
