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


def get_pq(pq_path, ctype='cloud'):
    df = pq.read_pandas(pq_path).to_pandas()

    if ctype == 'cloud':
        df = df[df.type == 0]
    elif ctype == 'core':
        df = df[df.type == 1]
    else:
        raise TypeError("Sample type not recognized")

    df['z'] = df.coord // (model.nx * model.ny)

    return df


def cloud_dz(pq_path, ctype='cloud'):
    df = get_pq(pq_path, ctype)
    df = df.groupby(['cid', 'z']).size().reset_index(name='counts')

    df.counts = df.counts * model.dx * model.dy
    return df.counts.value_counts().to_numpy()


def cloud_projection(pq_path):
    df = get_pq(pq_path)
    # TODO: Implement 2D projection of clouds

    return df


def cloud_volume(pq_path):
    df = get_pq(pq_path)

    return df.cid.value_counts()
