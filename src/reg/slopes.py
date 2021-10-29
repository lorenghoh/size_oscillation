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
pwd = Path(config["pwd"])


def piecewise_linear(x, y, max_n=2, return_reg=False):
    tree = DecisionTreeRegressor(criterion="friedman_mse", max_leaf_nodes=max_n)
    tree.fit(x[:, None], np.gradient(y))
    dys_dt = tree.predict(x[:, None]).flatten()

    _, ind, c = np.unique(dys_dt, return_index=True, return_counts=True)
    pt_in = ind[np.argsort(c)][::-1]
    pt_in = np.append(pt_in, len(dys_dt))

    # Find the longest segment for regression
    x_seg = x[pt_in[0] : np.min(pt_in[pt_in > pt_in[0]])]
    y_seg = y[pt_in[0] : np.min(pt_in[pt_in > pt_in[0]])]

    # Regression
    reg = lm.TheilSenRegressor()
    reg.fit(x_seg[:, None], y_seg)

    if return_reg:
        return reg

    return reg.coef_.item()


def linreg(x, y, model="RANSAC", remove_outlier=False, return_reg=False):
    if remove_outlier:
        x, y, mask = detect_outliers(None, x, y)

    if model == "RidgeCV":
        reg = lm.RidgeCV()
    elif model == "RANSAC":
        reg = lm.RANSACRegressor()
    elif model == "TheilSen":
        reg = lm.TheilSenRegressor()
    else:
        raise ValueError("Regression model not found.")
    reg.fit(x[:, None], y)

    if return_reg:
        return reg

    return reg.coef_.item()


if __name__ == "__main__":
    pass
