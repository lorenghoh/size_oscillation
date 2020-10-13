import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pathlib import Path
from tqdm import tqdm

# Set-up local (and temporary) sys.path for import
# All scripts for calculations and plots need this
from context import add_path
add_path(Path('.').resolve())

try:
    import lib.config

    from reg.samples import cloud_dz as sample
    from reg.distributions import kde as distribution
    from reg.slopes import piecewise_linear as slope
except Exception:
    raise Exception("Issue with dynamic import")

config = lib.config.read_config()
pwd = Path(config['pwd'])
src = Path(config['case']) / 'clusters'


def get_slope_piecewise(x, y):
    return slope(x, y, max_n=2, return_reg=False)


def get_slope(x, y):
    return slope(x, y, model='RidgeCV', remove_outlier=True)


def main():
    cluster_list = sorted(src.glob('*'))

    s_list = []
    for item in tqdm(cluster_list):
        samples = sample(item, ctype='core')
        x, y = distribution(samples)
        s = get_slope_piecewise(x, y)

        tqdm.write(f"{s}")
        s_list.append(s)

    df = pd.DataFrame(s_list, columns=['slope'])
    df.to_parquet(f"{pwd}/output/slope_CGILS_S6_COR_RANK_PIECEWISE.pq")


if __name__ == '__main__':
    main()
