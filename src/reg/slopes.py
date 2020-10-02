import numpy as np

from pathlib import Path

# Scikit-learn
from sklearn import linear_model as lm
from sklearn.model_selection import GridSearchCV

from get_outliers import detect_outliers

import lib.config


config = lib.config.read_config()
pwd = Path(config['pwd'])


def regressor(x, y, model='RidgeCV', remove_outlier=True):
    if remove_outlier:
        x, y, mask = detect_outliers(None, np.log10(x), np.log10(y))

    if model == 'RidgeCV':
        reg = lm.RidgeCV()
    elif model == 'RANSAC':
        reg = lm.RANSACRegressor()
    elif model == 'TheilSen':
        reg = lm.TheilSenRegressor()
    else:
        raise ValueError("Regression model not found.")
    reg.fit(x[:, None], y)

    return reg.coef_


if __name__ == '__main__':
    pass
