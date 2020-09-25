import numpy as np
import pandas as pd
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


def plot_kde_grad():
    src = Path(config["data"]) / "pq"

    # Plotting
    fig = lib.plot.init_plot((5, 4))
    ax = fig.add_subplot(111)

    for time in np.arange(800, 1200, 30):
        df = pq.read_pandas(f"{src}/cloud_cluster_{time:04d}.pq").to_pandas()

        df["z"] = df.coord // (1536 * 512)
        df = df.groupby(["cid", "z"]).size().reset_index(name="counts")

        # Define grid
        X_ = np.logspace(0, 3.5, 80)

        log_kde = KernelDensity(bandwidth=1).fit(df.counts[:, None])
        kde = log_kde.score_samples(X_[:, None]) / np.log(10)

        # Filter outliers
        reg_model = lm.RANSACRegressor()
        X_, Y_, _ = detect_outliers(reg_model, np.log10(X_), kde)
        Y_ = np.gradient(np.exp(Y_))
        print(np.min(Y_))

        ax.plot(X_, Y_, "-+", label=f"{time:03d}")

    ax.legend()
    ax.set_xlabel(r"$\log_{10}$ Counts")
    ax.set_ylabel(r"$\log_{10}$ Normalized Density")

    lib.plot.save_fig(fig, Path(__file__))


def get_kde_grad_timeseries(df_path):
    src = Path(config["data"])

    pq_list = list((src / "pq").glob('*.pq'))
    slopes = []
    for item in tqdm(pq_list):
        df = pq.read_pandas(item).to_pandas()

        df["z"] = df.coord // (1536 * 512)
        df = df.groupby(["cid", "z"]).size().reset_index(name="counts")

        # Define grid
        X_ = np.logspace(0, 3.5, 80)

        log_kde = KernelDensity(bandwidth=1).fit(df.counts[:, None])
        kde = log_kde.score_samples(X_[:, None]) / np.log(10)

        # Filter outliers
        reg_model = lm.RANSACRegressor()
        X_, Y_, _ = detect_outliers(reg_model, np.log10(X_), kde)
        Y_ = np.gradient(np.exp(Y_))

        slopes += [np.min(Y_)]

    df_out = pd.DataFrame(slopes, columns=['min_grad'])
    df_out.to_parquet(df_path)


def plot_kde_grad_timeseries(flag=True):
    df_path = Path(config["data"]) / "kde_grade_timeseries.pq"

    if flag:
        get_kde_grad_timeseries(df_path)
    df = pq.read_pandas(df_path).to_pandas()

    # Plotting
    fig = lib.plot.init_plot((12, 4))
    ax = fig.add_subplot(111)

    ax.plot(df.min_grad, '-+')

    ax.legend()
    ax.set_xlabel(r"$\log_{10}$ Counts")
    ax.set_ylabel(r"$\log_{10}$ Normalized Density")

    lib.plot.save_fig(fig, Path(__file__))


if __name__ == "__main__":
    plot_kde_grad()
