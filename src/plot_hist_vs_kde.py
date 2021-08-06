import warnings
import numpy as np

from pathlib import Path
import seaborn as sns

from scipy import stats

# Set-up local (and temporary) sys.path for import
# All scripts for calculations and plots need this
from context import add_path
add_path(Path('.').resolve())

try:
    import lib.plot as plib
    import lib.config

    from reg.outliers import detect_outliers
    from reg.samples import cloud_dz as sample
    from reg.distributions import rank
    from reg.distributions import kde
    from reg.slopes import piecewise_linear as slope
except Exception:
    raise Exception("Issue with dynamic import")

config = lib.config.read_config()
pwd = Path(config['pwd'])
src = Path(config['case']) / 'clusters'


def plot_hist_kde(x, y, r, s, samples):
    # ---- Plotting
    fig = plib.init_plot((7, 6))
    ax = fig.add_subplot(111)

    ax.plot(x, y, '*')

    h = np.histogram(samples)

    ax.set_xlabel(r"$\log_{10}$ R", fontsize=12)
    ax.set_ylabel(r"Normalized $\log_{10}$ S", fontsize=12)

    file_name = Path(f"{pwd}/png/kde_vs_hist.png")
    plib.save_fig(fig, file_name)


def main():
    cluster_list = sorted(src.glob('*.pq'))
    cluster = cluster_list[-360]

    samples = sample(cluster)
    x, y = kde(samples)
    r, s = rank(samples)

    plot_hist_kde(x, y, r, s, samples)


if __name__ == "__main__":
    main()
