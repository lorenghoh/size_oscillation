import numpy as np

from pathlib import Path

# Scikit-learn
from sklearn import linear_model as lm
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from scipy import optimize
from scipy import interpolate

from reg.outliers import detect_outliers

import lib.config


config = lib.config.read_config()
pwd = Path(config['pwd'])


def piecewise_linear(x, y, max_n=2, return_reg=False):
    tree = DecisionTreeRegressor(max_leaf_nodes=max_n)
    tree.fit(x[:, None], np.gradient(y))
    dys_dt = tree.predict(x[:, None]).flatten()

    pt_in = np.unique(dys_dt, return_index=True)[-1]
    pt_in = np.append(pt_in, len(dys_dt))

    # Find the longest segment for regression
    pt_b = np.argmax(pt_in[1:] - pt_in[:-1])
    x_seg = x[pt_in[pt_b]:pt_in[pt_b + 1]]
    y_seg = y[pt_in[pt_b]:pt_in[pt_b + 1]]

    # Regression
    reg = lm.TheilSenRegressor()
    reg.fit(x_seg[:, None], y_seg)

    if return_reg:
        return reg

    return reg.coef_.item()


def regressor(x, y, model='RidgeCV', remove_outlier=False, return_reg=False):
    if remove_outlier:
        x, y, mask = detect_outliers(None, x, y)

    if model == 'RidgeCV':
        reg = lm.RidgeCV()
    elif model == 'RANSAC':
        reg = lm.RANSACRegressor()
    elif model == 'TheilSen':
        reg = lm.TheilSenRegressor()
    else:
        raise ValueError("Regression model not found.")
    reg.fit(x[:, None], y)

    if return_reg:
        return reg

    return reg.coef_.item()


if __name__ == '__main__':
    pass
