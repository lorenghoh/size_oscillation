import numpy as np

import pyarrow.parquet as pq

from pathlib import Path
from tqdm import tqdm

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

import lib.config
import lib.case


config = lib.config.read_config()
pwd = Path(config['pwd'])
src = Path(config['case']) / 'clusters'

model = lib.case.Model()


def rank(samples):
    rank, size = np.unique(samples, return_index=True)

    return rank, size


def kde(samples):
    x_grid = np.log10(np.logspace(0, np.log10(np.max(samples)), 50))

    log_kde = KernelDensity(bandwidth=1).fit(np.log(samples[:, None]))
    kde = log_kde.score_samples(x_grid[:, None]) / np.log(10)

    return x_grid, kde


if __name__ == '__main__':
    cluster_list = sorted(src.glob('*'))
