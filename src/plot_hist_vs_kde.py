import numpy as np

from pathlib import Path

# Set-up local (and temporary) sys.path for import
# All scripts for calculations and plots need this
from context import add_path

add_path(Path(".").resolve())

try:
    import lib.plot as pl
    import lib.config

    from reg.samples import cloud_dz as sample
    from reg.distributions import kde
except Exception:
    raise Exception("Issue with dynamic import")

config = lib.config.read_config()
pwd = Path(config["pwd"])
src = Path(config["case"]) / "clusters"


def plot_hist_kde(samples):
    # ---- Plotting
    fig = pl.init_plot((6, 5))

    ax = fig.add_subplot(111)

    # KDE
    x, y, _ = kde(samples)
    y_data = np.log10(samples)

    ax.plot(x, y, "-", lw=3, color="C1")

    # Histogram
    samples = samples[samples > 1e3]
    bins = np.log10(np.logspace(3, np.log10(np.max(samples)), 9))

    hist, edges = np.histogram(y_data, bins=bins, density=True)
    hist = np.log(hist / 2) / np.log(10)

    y_end = -6
    for i in range(len(edges) - 1):
        x_f = np.array([edges[i], edges[i + 1]])
        y1_f = np.array([hist[i], hist[i]])
        y2_f = np.array([y_end, y_end])

        ax.fill_between(x_f, y1_f, y2_f, where=(y1_f > y2_f), color="C0", alpha=0.4)

    ax.set_ylim([-2.5, 0])

    ax.set_xlabel(r"$\log_{10}$ Cloud Size", fontsize=12)
    ax.set_ylabel(r"$\log_{10}$ Normalized Density", fontsize=12)

    file_name = Path(f"{pwd}/png/kde_vs_hist.png")
    pl.save_fig(fig, file_name)


def main():
    cluster_list = sorted(src.glob("*.pq"))
    cluster = cluster_list[720]

    samples = sample(cluster)
    plot_hist_kde(samples)


if __name__ == "__main__":
    main()
