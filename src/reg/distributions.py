import numpy as np

from pathlib import Path

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from lib.config import read_config
from lib.case import Model


config = read_config()
pwd = Path(config['pwd'])
src = Path(config['case']) / 'clusters'

model = Model()


def rank(samples, log=True):
    rank, size = np.unique(samples, return_index=True)

    if log:
        mask = (rank > 0) & (size > 0)
        return np.log10(rank[mask]), np.log10(size[mask])

    return rank, size


def kde(samples, cv=True):
    x_grid = np.log10(np.logspace(3, np.log10(np.max(samples)), 50))
    y_data = np.log10(samples[:, None])

    if cv:
        # use grid search cross-validation to optimize the bandwidth
        params = {"bandwidth": np.linspace(1e-2, 1, 100)}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(y_data)

        bw = grid.best_estimator_.bandwidth
        log_kde = grid.best_estimator_
    else:
        # Place a safe bet for bandwidth
        log_kde = KernelDensity(bandwidth=1).fit(y_data)

    kde10 = log_kde.score_samples(x_grid[:, None]) / np.log(10)

    return x_grid, kde10, bw
