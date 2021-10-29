import warnings
import numpy as np

from pathlib import Path

from sklearn import linear_model as lm
from sklearn.tree import DecisionTreeRegressor

from scipy import optimize
from scipy import interpolate

# Set-up local (and temporary) sys.path for import
# All scripts for calculations and plots need this
from context import add_path

add_path(Path(".").resolve())

try:
    import lib.plot as pl
    import lib.config

    from reg.outliers import detect_outliers
    from reg.samples import cloud_dz as sample
    from reg.distributions import kde as distribution
    from reg.slopes import piecewise_linear as slope
except Exception:
    raise Exception("Issue with dynamic import")

config = lib.config.read_config()
pwd = Path(config["pwd"])
src = Path(config["case"]) / "clusters"


def plot_piecewise_linear(x, y, bw=None):
    # ---- Plotting
    fig = pl.init_plot((6, 8))

    # Subplot 1: decision tree
    ax = fig.add_subplot(211)
    dy = np.gradient(y)

    tree = DecisionTreeRegressor(max_leaf_nodes=2)
    tree.fit(x[:, None], dy)
    dys_dt = tree.predict(x[:, None])

    clist = ['C2', 'C3']
    colors = iter(clist)

    ax.plot(x, dy, ".", ms=8, color='C0')
    for item in np.unique(dys_dt):
        mask = (dys_dt == item)

        c = next(colors)
        ax.plot(x[mask], dys_dt[mask], "-", color=c, lw=2)
        ax.axvspan(x[mask][0], x[mask][-1], color=c, alpha=0.2)

    ax.set_xlabel(r"$\log_{10}$ Cloud Size", fontsize=12)
    ax.set_ylabel(r"$\partial \log_{10}$ Normalized Density", fontsize=12)

    # Subplot 2: slope
    ax = fig.add_subplot(212)

    linreg = slope(x, y, max_n=2, return_reg=True)
    y_rs = linreg.predict(x[:, None])

    print(f"Linear reg slope: {linreg.coef_[0]}")

    if bw is not None:
        print(f"Bandwidth: {bw}")

    ax.plot(x, y, ".", ms=8)
    ax.plot(x, y_rs, "-")

    colors = iter(clist)
    for item in np.unique(dys_dt):
        mask = (dys_dt == item)
        c = next(colors)

        ax.axvspan(x[mask][0], x[mask][-1], color=c, alpha=0.2)

    ax.set_xlabel(r"$\log_{10}$ Cloud Size", fontsize=12)
    ax.set_ylabel(r"$\log_{10}$ Normalized Density", fontsize=12)

    file_name = Path(f"{pwd}/png/piecewise_linear_fit.png")
    pl.save_fig(fig, file_name)


def main():
    cluster_list = sorted(src.glob("*.pq"))
    cluster = cluster_list[360]

    samples = sample(cluster, ctype='cloud')
    x, y, bw = distribution(samples)

    plot_piecewise_linear(x, y, bw)


if __name__ == "__main__":
    main()
