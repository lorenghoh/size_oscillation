import numpy as np
import pandas as pd

from pathlib import Path

# Set-up local (and temporary) sys.path for import
# All scripts for calculations and plots need this
from context import add_path

add_path(Path(".").resolve())

try:
    import lib.config
    import lib.distance as dist
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


def find_centroid(df):
    # Centroid
    df = df[["cid", "x", "y"]].copy()

    df = df.groupby("cid").filter(lambda x: len(x) > 8)
    df = df.groupby("cid").agg(y=("y", "mean"), x=("x", "mean"))

    return df


def find_pb_clusters(df, nx=1536, ny=512):
    """
    Re-index cloud ids returned from measure.label. Because the cloud
    domain is governed by doubly-periodic boundary conditions, a small
    number of clouds will cross the boundary. The function looks at x
    and y boundaries (x = 0 and y = 0) and see if cloud regions can be
    found on the other side of the domain (nx - 1 and ny - 1).

    Unfortunately, it is not yet too useful at this point since most of
    the utility functions do not work with periodic boundary conditions.

    Returns
    -------
    df : Pandas Dataframe
        Dataframe with updated cids

    """

    # First check x-axis boundaries
    x_cid = df[df.x == 0].cid.unique()
    for uid in x_cid:
        # List of y-positions at x-axis boundary
        _df = df[(df.cid == uid) & (df.x == 0)][["z", "y", "x"]]
        _df.loc[df.x == 0, "x"] = nx - 1

        _id = pd.merge(df, _df, on=["z", "y", "x"]).cid.unique()
        df.loc[df.cid.isin(_id), "cid"] = uid

    # Repeat for y-axis boundaries
    y_cid = df[df.y == 0].cid.unique()
    for uid in y_cid:
        # List of x-positions at y-axis boundary
        _df = df[(df.cid == uid) & (df.y == 0)][["z", "y", "x"]]
        _df.loc[df.x == 0, "y"] = ny - 1

        _id = pd.merge(df, _df, on=["z", "y", "x"]).cid.unique()
        df.loc[df.cid.isin(_id), "cid"] = uid

    return df


if __name__ == "__main__":
    cluster_list = sorted(src.glob("*"))
    item = cluster_list[139]

    df = pd.read_parquet(item)

    df = df[df.type == 1].copy()  # Cloud core
    df = coord_to_zxy_cols((df))

    # Find 2D projections
    df = df.drop_duplicates(subset=["y", "x"])
    df = find_centroid(find_pb_clusters(df))

    arr = np.array([df.x.values, df.y.values])
    tril = np.tril(dist.calc_pdist_pb(arr, arr))
    print(tril)
