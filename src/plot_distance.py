import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path

# Set-up local (and temporary) sys.path for import
# All scripts for calculations and plots need this
from context import add_path

add_path(Path(".").resolve())

try:
    import lib.config
    import lib.plot as pl
    import lib.distance as dist
    import lib.cluster as cl
except Exception:
    raise Exception("Issue with dynamic import")

config = lib.config.read_config()
pwd = Path(config["pwd"])
src = Path(config["case"]) / "clusters"


def plot_cent_dist():
    cluster_list = sorted(src.glob("*"))

    # m_dist = np.zeros((8, 1440))
    m_dist = np.zeros((30, 144))
    for i, item in enumerate(tqdm(cluster_list[0:1440:10])):
        df = pd.read_parquet(item)
        df = df[df.type == 1].copy()
        df = cl.coord_to_zxy_cols(df)

        cents = cl.find_centroid(df)
        arr = np.array([cents.x.values, cents.y.values])

        tril = np.tril(dist.calc_pdist_pb(arr, arr))
        tril = tril.ravel()

        for j, d in enumerate(np.arange(1, 30)):
            max_d = d * 1e3 / 25
            m_dist[j, i] = np.median(tril[(tril > 5) & (tril < max_d)])

    # ---- Plotting
    fig = pl.init_plot((12, 6))
    ax = fig.add_subplot(111)

    lsc = iter(["-", "--", ":", "-."])

    for i in np.arange(5, 16, 3):
        ax.plot(m_dist[i] - np.mean(m_dist[i]), next(lsc), label=f"{i+1} km")

    ax.set_title("Median Distance Between Centroids", fontsize=16)
    ax.set_xlabel("Time [Hours]")
    ax.set_ylabel("$\delta \tilde{d}$")
    ax.legend()

    file_name = Path(f"{pwd}/png/plot_distance.png")
    pl.save_fig(fig, file_name)



def plot_intercluster_dist():
    pass


if __name__ == "__main__":
    plot_cent_dist()
