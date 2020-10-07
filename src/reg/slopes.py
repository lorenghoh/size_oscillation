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


def regressor(x, y, model='RidgeCV', remove_outlier=False):
    if remove_outlier:
        x, y, mask = detect_outliers(None, x, y)

    if model == 'RidgeCV':
        reg = lm.RidgeCV()
    elif model == 'RANSAC':
        reg = lm.RANSACRegressor()
    elif model == 'TheilSen':
        reg = lm.TheilSenRegressor()
    elif model == 'Piecewise':
        tree = DecisionTreeRegressor(max_leaf_nodes=2)
        tree.fit(x[:, None], np.gradient(y[:None]))
        dys_dt = tree.predict(x[:, None]).flatten()

        # Point of inflection. We are assuming two piecewise-linear
        # curves, so the first index will be the point of inflection
        pt_in = np.unique(dys_dt, return_index=True)[-1][0]

        # Regression
        linreg = lm.RidgeCV()
        linreg.fit(x[pt_in:, None], y[pt_in:])

        return linreg.coef_.item
    else:
        raise ValueError("Regression model not found.")
    reg.fit(x[:, None], y)

    return reg.coef_.item()


if __name__ == '__main__':
    pass
