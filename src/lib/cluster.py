import numpy as np


def coord_to_zxy_cols(df):
    """
    Unravels the raveled index (index -> z, y, x) and creates corresponding
    columns in the resulting Pandas DataFrame.

    """
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


def project_clouds(df):
    return df.drop_duplicates(subset=["y", "x"])
