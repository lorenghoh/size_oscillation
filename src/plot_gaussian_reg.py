import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn import linear_model as lm

# from statsmodels.nonparametric import smoothers_lowess as sl
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as kern

from pathlib import Path

import lib.plot
import lib.config

from matplotlib.colors import LogNorm


config = lib.config.read_config()


def main():
    src = Path(config["data"])

    # file_name = f"{src}/size_dist_slope.pq"
    # reg_model = "lr"  # See get_linear_reg.py for choices

    file_name = f"{src}/kde_grade_timeseries.pq"
    reg_model = "min_grad"

    df = pq.read_pandas(file_name).to_pandas()

    kern_noise = kern.WhiteKernel(noise_level=1e-1)
    kern_periodic = kern.ExpSineSquared(
        length_scale=1e2,
        length_scale_bounds=(10, 1e3),
        periodicity=45,
        periodicity_bounds=(5, 60),
    )
    kernel = kern_noise + 1.0 * kern_periodic + 1.0 * kern.RBF()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

    X = np.arange(720)
    y = np.array(df[f"{reg_model}"])[720:]
    # y = y - sl.lowess(y, X, return_sorted=False, frac=0.3)

    # ---- Random sampling
    # TODO: Use scikit-learn's CV module
    # https://scikit-learn.org/stable/modules/cross_validation.html
    xx = np.arange(0, 720)
    np.random.shuffle(xx)
    x_tr, _ = xx[:360], xx[360:]
    # x_tr = xx

    gp.fit(x_tr[:, None], y[x_tr])
    # gp.fit(X[:, None], y)
    # print(gp.kernel_)
    # for item in gp.kernel_.hyperparameters:
    #     print(item)
    # print(gp.kernel_.theta)

    # ---- Plotting
    fig = lib.plot.init_plot(figsize=(12, 4))
    # fig = lib.plot.init_plot(figsize=(5, 4))
    ax = fig.add_subplot(111)

    # Plot LML landscape
    # theta0 = np.logspace(0, 4, 49)
    # theta1 = np.logspace(0, 5, 50)
    # Theta0, Theta1 = np.meshgrid(theta0, theta1)
    # LML = [
    #     [
    #         gp.log_marginal_likelihood(
    #             np.log(
    #                 [
    #                     np.exp(gp.kernel_.theta[0]),
    #                     np.exp(gp.kernel_.theta[1]),
    #                     Theta0[i, j],
    #                     Theta1[i, j],
    #                 ]
    #             )
    #         )
    #         for i in range(Theta0.shape[0])
    #     ]
    #     for j in range(Theta0.shape[1])
    # ]
    # LML = np.array(LML).T

    # vmin, vmax = (LML).min(), (LML).max()
    # print(vmin, vmax)
    # level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 50), decimals=1)
    # im = ax.contour(
    #     np.log10(Theta0),
    #     np.log10(Theta1),
    #     LML,
    #     levels=level,
    #     norm=LogNorm(vmin=vmin, vmax=vmax),
    #     linewidths=0.75,
    # )
    # fig.colorbar(im, ax=ax)

    # ax.set_xlabel("Length-scale")
    # ax.set_ylabel("Periodicity")

    X = np.linspace(0, 720, 720)
    ax.plot(X, y, "--*", zorder=9)

    y_mean, y_std = gp.predict(X[:, None], return_std=True)
    ax.plot(X[:], y_mean[:], "k", lw=1, zorder=8)
    ax.plot(x_tr, y[x_tr], "r+", zorder=9)
    ax.fill_between(X, y_mean - y_std, y_mean + y_std, alpha=0.2, color="k")
    print(gp.kernel_)
    print(gp.log_marginal_likelihood_value_ / len(X))

    y_samples = gp.sample_y(X[:, None], 10)
    ax.plot(X, y_samples, lw=0.5, alpha=0.2)

    # Plot labels
    ax.set_xlabel(r"Time [min]")
    ax.set_ylabel(r"Slope $b$")

    lib.plot.save_fig(fig, Path(__file__))


if __name__ == "__main__":
    main()
