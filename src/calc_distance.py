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


if __name__ == "__main__":
    cluster_list = sorted(src.glob("*"))
    item = cluster_list[139]

    df = pd.read_parquet(item)

    df = df[df.type == 1].copy()  # Cloud core
    df = coord_to_zxy_cols((df))

    # Find 2D projections
    df = df.drop_duplicates(subset=["y", "x"])
    df = find_centroid(df)

    arr = np.array([df.x.values, df.y.values])
    print(np.tril(dist.calc_pdist_pb(arr, arr)))
