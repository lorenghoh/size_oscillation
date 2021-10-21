import numpy as np

from pathlib import Path
from sklearn.neighbors import KernelDensity

# Set-up local (and temporary) sys.path for import
# All scripts for calculations and plots need this
from context import add_path

add_path(Path(".").resolve())

try:
    import lib.plot as pl
    import lib.config

    from reg.outliers import detect_outliers
    from reg.samples import cloud_dz as sample
    from reg.distributions import rank
    from reg.distributions import kde
    from reg.slopes import piecewise_linear as slope
except Exception:
    raise Exception("Issue with dynamic import")

config = lib.config.read_config()
pwd = Path(config["pwd"])
src = Path(config["case"]) / "clusters"


def plot_hist_kde(samples):
    # ---- Plotting
    fig = pl.init_plot((7, 6))

    ax = fig.add_subplot(111)
    
    # KDE
    x = np.log10(np.logspace(3, np.log10(np.max(samples)), 30))

    log_kde = KernelDensity(bandwidth=0.4).fit(np.log10(samples[:, None]))
    y = log_kde.score_samples(x[:, None]) / np.log(10)

    ax.plot(x, y)

    samples = samples[samples > 1e3]
    bins = np.log10(np.logspace(3, np.log10(np.max(samples)), 9))

    hist, edges = np.histogram(np.log10(samples), bins=bins, density=True)
    hist = np.log(hist / np.sum(hist)) / np.log(10)

    depth = -5.75
    ax.plot([edges[0], edges[0]], [depth, hist[0]], 'k-', alpha=0.5)
    for i in range(len(edges) - 1):
        ax.plot([edges[i], edges[i + 1]], [hist[i], hist[i]], 'k-', alpha=0.5)
        ax.plot([edges[i + 1], edges[i + 1]], [depth, hist[i]], 'k-', alpha=0.5)
        
        # TODO: fill between the lines`

    ax.set_ylim([-2.5, 0])

    ax.set_xlabel(r"$\log_{10}$ Cloud Size", fontsize=12)
    ax.set_ylabel(r"$\log_{10}$ Normalized Density", fontsize=12)

    file_name = Path(f"{pwd}/png/kde_vs_hist.png")
    pl.save_fig(fig, file_name)


def main():
    cluster_list = sorted(src.glob("*.pq"))
    cluster = cluster_list[-360]

    samples = sample(cluster)
    plot_hist_kde(samples)


if __name__ == "__main__":
    main()
