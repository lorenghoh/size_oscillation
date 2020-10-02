from dataclasses import dataclass
from pathlib import Path

import xarray as xr

from lib import config

config = config.read_config()
pwd = Path(config['pwd'])
src = Path(config['case'])


@dataclass
class Model:
    case_name: str
    nx: int
    ny: int
    nz: int

    def __init__(self):
        with xr.open_dataset(src / 'CGILS_1728x576x192_25m_1s_ent.nc') as df:
            self.case_name = 'CGILS'
            self.nx = 1728
            self.ny = 576
            self.nz = 192


if __name__ == '__main__':
    print(Model())
