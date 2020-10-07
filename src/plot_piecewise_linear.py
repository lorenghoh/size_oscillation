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
add_path(Path('.').resolve())

try:
    import lib.plot as plib
    import lib.config

    from reg.outliers import detect_outliers
    from reg.samples import cloud_dz as sample
    from reg.distributions import kde as distribution
except Exception:
    raise Exception("Issue with dynamic import")

config = lib.config.read_config()
pwd = Path(config['pwd'])
src = Path(config['case']) / 'clusters'


def plot_piecewise_linear(x, y):
    # ---- Plotting
    fig = plib.init_plot((6, 4))
    ax = fig.add_subplot(111)

    tree = DecisionTreeRegressor(max_leaf_nodes=2)
    tree.fit(x[:, None], np.gradient(y[:None]))
    dys_dt = tree.predict(x[:, None]).flatten()

    # Point of inflection. We are assuming two piecewise-linear
    # curves, so the first index will be the point of inflection
    pt_in = np.unique(dys_dt, return_index=True)[-1][0]

    # Regression
    linreg = lm.TheilSenRegressor()
    linreg.fit(x[pt_in:, None], y[pt_in:])
    y_rs = linreg.predict(x[:, None])

    print(linreg.coef_)

    ax.plot(x, y, '*')
    ax.plot(x, y_rs, "-", label="RANSAC")

    ax.set_xlabel(r"$\log_{10}$ R")
    ax.set_ylabel(r"$\log_{10}$ S")

    file_name = Path(f"{pwd}/png/piecewise_linear_fit.png")
    plib.save_fig(fig, file_name)


def main():
    cluster_list = sorted(src.glob('*.pq'))
    cluster = cluster_list[-720]

    samples = sample(cluster)
    x, y = distribution(samples)

    mask = (y > -10)
    x = x[mask]
    y = y[mask]

    plot_piecewise_linear(x, y)


if __name__ == "__main__":
    main()
