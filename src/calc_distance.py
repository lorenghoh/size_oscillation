import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq

from pathlib import Path
from tqdm import tqdm

from scipy.spatial import distance

# Set-up local (and temporary) sys.path for import
# All scripts for calculations and plots need this
from context import add_path

add_path(Path(".").resolve())

try:
    import lib.config
except Exception:
    raise Exception("Issue with dynamic import")

config = lib.config.read_config()
pwd = Path(config["pwd"])
src = Path(config["case"]) / "clusters"


def coord_to_zxy_cols(df):
    index = df.coord.values

    # TODO: Now that this is vectorized, I can speed up the post-processing
    df["z"], df["y"], df["x"] = np.unravel_index(index, (192, 512, 1536))

    return df


@nb.jit(nopython=True, fastmath=True, parallel=True)
def calc_pdist(x, y):
    pdist = np.zeros((len(x), len(y)))

    for i in range(pdist.shape[0]):
        for j in range(pdist.shape[1]):
            dx = x[j] - x[i]
            dy = y[j] - y[i]

            pdist[i, j] = np.sqrt(dx ** 2 + dy ** 2)

    return pdist


@nb.jit(nopython=True, fastmath=True)
def _ad(a, b):
    return np.abs(a - b)


@nb.jit(nopython=True, fastmath=True, parallel=True)
def calc_pdist_pb(x, y, nx=1536, ny=512):
    pdist = np.zeros((len(x), len(y)))

    for i in range(pdist.shape[0]):
        for j in range(pdist.shape[1]):
            dx = _ad(x[j], x[i])
            dx_pb = np.minimum(dx, _ad(dx, nx))

            dy = _ad(y[j], y[i])
            dy_pb = np.minimum(dy, _ad(dy, ny))

            pdist[i, j] = np.sqrt(dx_pb ** 2 + dy_pb ** 2)

    return pdist


def find_centroid(df):
    # Centroid
    df = df[["cid", "x", "y"]].copy()

    df = df.groupby("cid").filter(lambda x: len(x) > 8)
    df = df.groupby("cid").agg(y=("y", "mean"), x=("x", "mean"))

    return df


if __name__ == "__main__":
    cluster_list = sorted(src.glob("*"))
    item = cluster_list[139]

    df = pd.read_parquet(item)

    df = df[df.type == 1].copy()  # Cloud core
    df = coord_to_zxy_cols((df))

    # Find 2D projections
    df = df.drop_duplicates(subset=["y", "x"])
    df = find_centroid(df)

    print(calc_pdist_pb(df.x.values, df.y.values))
