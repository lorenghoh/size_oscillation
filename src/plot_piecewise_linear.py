import warnings
import numpy as np

from pathlib import Path
import seaborn as sns

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


def plot_piecewise_linear(x, y):
    # ---- Plotting
    fig = pl.init_plot((6, 8))

    # Subplot 1: decision tree
    ax = fig.add_subplot(211)
    dy = np.gradient(y)

    tree = DecisionTreeRegressor(max_leaf_nodes=2)
    tree.fit(x[:, None], dy)
    dys_dt = tree.predict(x[:, None])

    colors = iter(["C0", "C1"])

    ax.plot(x, dy, ".", ms=8)
    for item in np.unique(dys_dt):
        mask = dys_dt == item
        ax.plot(x[mask], dys_dt[mask], "-", color="C1", lw=2)
        ax.axvspan(x[mask][0], x[mask][-1], color=next(colors), alpha=0.2)

    ax.set_xlabel(r"$\log_{10}$ Cloud Size", fontsize=12)
    ax.set_ylabel(r"$\delta \log_{10}$ Normalized Density", fontsize=12)

    # Subplot 2: slope
    ax = fig.add_subplot(212)

    linreg = slope(x, y, max_n=2, return_reg=True)
    y_rs = linreg.predict(x[:, None])

    print(linreg.coef_)

    ax.plot(x, y, ".", ms=8)
    ax.plot(x, y_rs, "-")

    ax.set_xlabel(r"$\log_{10}$ Cloud Size", fontsize=12)
    ax.set_ylabel(r"$\log_{10}$ Normalized Density", fontsize=12)

    file_name = Path(f"{pwd}/png/piecewise_linear_fit.png")
    pl.save_fig(fig, file_name)


def main():
    cluster_list = sorted(src.glob("*.pq"))
    cluster = cluster_list[-360]

    samples = sample(cluster)
    x, y = distribution(samples)

    plot_piecewise_linear(x, y)


if __name__ == "__main__":
    main()
