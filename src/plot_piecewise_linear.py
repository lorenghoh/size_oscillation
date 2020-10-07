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
    from reg.distributions import rank as distribution
    from reg.slopes import piecewise_linear as slope
except Exception:
    raise Exception("Issue with dynamic import")

config = lib.config.read_config()
pwd = Path(config['pwd'])
src = Path(config['case']) / 'clusters'


def plot_piecewise_linear(x, y):
    # ---- Plotting
    fig = plib.init_plot((6, 4))
    ax = fig.add_subplot(111)

    linreg = slope(x, y, return_reg=True)
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

    plot_piecewise_linear(x, y)


if __name__ == "__main__":
    main()
