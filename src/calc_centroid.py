import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pathlib import Path
from tqdm import tqdm

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

    # Model parameters from CGILS_S6 run
    nx, ny = 1728, 576

    df['z'] = index // (ny * nx)
    index = index % (ny * nx)

    df['y'] = index // nx
    df['x'] = index % nx

    return df


def main():
    cluster_list = sorted(src.glob("*"))
    item = cluster_list[139]

    df = pd.read_parquet(item)
    df = df[df.type == 1]  # Cloud core

    print(coord_to_zxy_cols(df))

    return

    df = pd.DataFrame(s_list, columns=["slope"])
    df.to_parquet(f"{pwd}/output/slope_CGILS_S6_COR_KDE_PIECEWISE.pq")


if __name__ == "__main__":
    main()
