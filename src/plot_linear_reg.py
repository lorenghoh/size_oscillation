import numpy as np
import pyarrow.parquet as pq

# Scikit
from sklearn import linear_model as lm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from get_outliers import detect_outliers

from tqdm import tqdm
from pathlib import Path

import lib.plot
import lib.config
import seaborn as sns


config = lib.config.read_config()


def plot_kde():
    src = Path(config["data"]) / "pq"

    # Plotting
    fig = lib.plot.init_plot((5, 4))
    ax = fig.add_subplot(111)

    df = pq.read_pandas(f"{src}/cloud_cluster_0575.pq").to_pandas()

    df["z"] = df.coord // (1536 * 512)
    df = df.groupby(["cid", "z"]).size().reset_index(name="counts")

    # Define grid
    X_ = np.logspace(0, np.log10(df.counts.max()), 50)

    log_kde = KernelDensity(bandwidth=1).fit(df.counts[:, None])
    kde = log_kde.score_samples(X_[:, None]) / np.log(10)

    # ax.plot(np.log10(X_), kde, "-+")

    # Filter outliers
    reg_model = lm.RANSACRegressor()
    X_, Y_, _ = detect_outliers(reg_model, np.log10(X_), kde)

    ax.plot(X_, np.gradient(np.exp(Y_)), "-+")

    # # Ridge regression
    # reg_model = lm.RidgeCV()
    # reg_model.fit(X_[:, None], Y_)

    # y_reg = reg_model.predict(X_[:, None])
    # ax.plot(X_, y_reg, "--", label="Linear Regression")

    # # RANSAC
    # rs_estimator = lm.RANSACRegressor()
    # rs_estimator.fit(X_[:, None], Y_)
    # y_rs = rs_estimator.estimator_.predict(X_[:, None])
    # ax.plot(X_, y_rs, "-*", c="g", label="RANSAN Regression")

    # # Theil Sen regression
    # ts_model = lm.TheilSenRegressor(n_jobs=16)
    # ts_model.fit(X_[:, None], Y_)
    # y_ts = ts_model.predict(X_[:, None])
    # ax.plot(X_, y_ts, "-", c="k", label="Theil-Sen Regression")

    # # Huber regression
    # hb_model = lm.HuberRegressor(fit_intercept=False)
    # hb_model.fit(X_[:, None], Y_)
    # y_ts = hb_model.predict(X_[:, None])
    # ax.plot(X_, y_ts, "-.", c="b", label="Huber Regression")

    # # Output slopes
    # print(
    #     f"{reg_model.coef_[0]:.3f}, \n"
    #     f"{rs_estimator.estimator_.coef_[0]:.3f}, \n"
    #     f"{ts_model.coef_[0]:.3f}, \n"
    #     f"{hb_model.coef_[0]:.3f} \n"
    # )

    # Labels
    # ax.legend()
    # ax.set_xlim((0, 3.5))
    # ax.set_ylim((-6, 0))

    ax.set_xlabel(r"$\log_{10}$ Counts")
    ax.set_ylabel(r"$\log_{10}$ Normalized Density")

    lib.plot.save_fig(fig, Path(__file__))


if __name__ == "__main__":
    plot_kde()
