import warnings
import numpy as np

from pathlib import Path
from pyarrow import parquet as pq

# Set-up local (and temporary) sys.path for import
# All scripts for calculations and plots need this
from context import add_path

add_path(Path(".").resolve())

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
pwd = Path(config["pwd"])
src = Path(config["case"]) / "clusters"


def main():
    # ---- Plotting
    fig = plib.init_plot((8, 3))
    ax = fig.add_subplot(111)

    p = Path(f"{pwd}/output/slope_CGILS_S6_COR_KDE_PIECEWISE.pq")
    df = pq.read_pandas(p).to_pandas()

    y = df.slope.to_numpy()
    x = np.arange(len(y)) / 60

    ax.plot(x, y, "*-")

    ax.set_xlabel("Time [hours]", fontsize=12)
    ax.set_ylabel(r"b", fontsize=12)

    file_name = Path(f"{pwd}/png/timeseries.png")
    plib.save_fig(fig, file_name)


if __name__ == "__main__":
    main()
