import warnings

import numpy as np
import numba as nb
import pandas as pd

# Scikit
from sklearn import linear_model as lm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

# Plotting
import lib.plot as plib
import seaborn as sns


def detect_outliers(regressor, x, y, rho=0.98):
    if regressor is None:
        regressor = lm.RidgeCV()

    reg_name = regressor.__class__.__name__

    if reg_name == "RANSACRegressor":
        outlier = find_RANSAC_outlier(regressor, x, y)
        is_outlier = outlier

        return compress_inliers(is_outlier, x, y)
    elif reg_name == "HuberRegressor":
        outlier = find_huber_outlier(regressor, x, y)
        is_outlier = outlier

        return compress_inliers(is_outlier, x, y)
    else:
        try:
            # Initial fit using linear regressor
            regressor.fit(x[:, None], y)
            y_reg = regressor.predict(x[:, None])

            is_outlier = np.zeros(len(x), dtype=bool)
            is_outlier[np.argsort(np.abs(y - y_reg))[-1]] = True
        except Exception:
            raise TypeError("Regressor not compatible")

    x_ma = np.ma.masked_array(x, is_outlier)
    y_ma = np.ma.masked_array(y, is_outlier)

    max_iter = int(len(x) * 0.20)  # 20% max outliers

    for i in range(max_iter):
        # Remove outlier and re-run regression
        corr = get_linear_corr(regressor, x_ma, y_ma)
        # print(corr)

        # Stop at threshold correlation score (rho)
        # Otherwise, filter next outlier
        if corr >= rho:
            break
        else:
            x_ma, y_ma, outlier = find_linear_outlier(regressor, x_ma, y_ma)

    if corr <= 0.9:
        warnings.warn("Warning: data cannot be fully linearlized")

    return x_ma.compressed(), y_ma.compressed(), x_ma.mask


def get_linear_corr(regressor, x_ma, y_ma):
    x_ = x_ma.compressed()
    y_ = y_ma.compressed()

    regressor.fit(x_[:, None], y_)

    return regressor.score(x_[:, None], y_)


def find_linear_outlier(regressor, x_ma, y_ma):
    regressor.fit(x_ma.compressed()[:, None], y_ma.compressed())
    y_reg = regressor.predict(x_ma.data[:, None])
    y_reg = np.ma.masked_array(y_reg, mask=y_ma.mask)

    outlier = np.abs(y_ma - y_reg).argsort(endwith=False)[-1]
    x_ma.mask[outlier] = True
    y_ma.mask[outlier] = True

    return x_ma, y_ma, outlier


def compress_inliers(is_outlier, x, y):
    x_ = np.compress(~is_outlier, x)
    y_ = np.compress(~is_outlier, y)

    return x_, y_, is_outlier


def find_RANSAC_outlier(regressor, x, y):
    regressor.fit(x[:, None], y)

    return ~regressor.inlier_mask_


def find_huber_outlier(regressor, x, y):
    regressor.fit(x[:, None], y)

    return regressor.outliers_
