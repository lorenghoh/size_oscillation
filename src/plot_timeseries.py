import numpy as np

from pathlib import Path
from pyarrow import parquet as pq

# Set-up local (and temporary) sys.path for import
# All scripts for calculations and plots need this
from context import add_path

add_path(Path(".").resolve())

try:
    import lib.plot as pl
    import lib.config
except Exception:
    raise Exception("Issue with dynamic import")

config = lib.config.read_config()
pwd = Path(config["pwd"])
src = Path(config["case"]) / "clusters"


def main():
    # ---- Plotting
    fig = pl.init_plot((12, 6))
    ax = fig.add_subplot(111)

    p = Path(f"{pwd}/output/slope_CGILS_S6_COR_KDE_PIECEWISE.pq")
    df = pq.read_pandas(p).to_pandas()

    y = df.slope.to_numpy()
    x = np.arange(len(y)) / 60

    ax.plot(x, y, "o-")

    ax.set_xlabel("Time [hours]", fontsize=12)
    ax.set_ylabel("b", fontsize=12)

    file_name = Path(f"{pwd}/png/timeseries.png")
    pl.save_fig(fig, file_name)


if __name__ == "__main__":
    main()
