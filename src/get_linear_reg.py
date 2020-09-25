import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Scikit-learn
from sklearn import linear_model as lm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from get_outliers import detect_outliers
from pathlib import Path

import lib.config

config = lib.config.read_config()
pwd = config['pwd']


def calc_slope(counts):
    # Define grid
    X_ = np.logspace(0, np.log10(counts.max()), 50)

    log_kde = KernelDensity(bandwidth=1).fit(counts[:, None])
    kde = log_kde.score_samples(X_[:, None]) / np.log(10)

    # Filter outliers
    X_, Y_, mask = detect_outliers(None, np.log10(X_), kde)

    # Ridge regression
    lr_model = lm.RidgeCV()
    lr_model.fit(X_[:, None], Y_)

    # RANSAC regression
    rs_estimator = lm.RANSACRegressor()
    rs_estimator.fit(X_[:, None], Y_)

    # Theil Sen regression
    ts_model = lm.TheilSenRegressor(n_jobs=16)
    ts_model.fit(X_[:, None], Y_)

    coefs = (lr_model.coef_, rs_estimator.estimator_.coef_, ts_model.coef_)
    return coefs


def main():
    src = Path(config['data'])

    slopes = []
    cols = ["lr", "rs", "ts"]
    for item in tqdm(sorted(src.glob('*.pq'))):
        df = pq.read_pandas(item).to_pandas()

        df["z"] = df.coord // (1536 * 512)
        df = df.groupby(["cid", "z"]).size().reset_index(name="counts")

        lr, rs, ts = calc_slope(df.counts)
        slopes += [(lr, rs, ts)]
        tqdm.write(
            f"lr = {lr[0]:.3f}, "
            f"rs = {rs[0]:.3f}, "
            f"ts = {ts[0]:.3f}"
        )
    df_out = pd.DataFrame(slopes, columns=cols)
    df_out.to_parquet(f"{src}/size_dist_slope.pq")


if __name__ == "__main__":
    main()
