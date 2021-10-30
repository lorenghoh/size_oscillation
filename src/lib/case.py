from dataclasses import dataclass
from pathlib import Path

from lib import config

config = config.read_config()
pwd = Path(config['pwd'])
src = Path(config['case'])


@dataclass
class Model:
    case_name: str
    dx: int
    dy: int
    dz: int
    nx: int
    ny: int
    nz: int

    def __init__(self):
        self.case_name = 'CGILS'
        self.dx = 25
        self.dy = 25
        self.dz = 25
        self.nx = 1728
        self.ny = 576
        self.nz = 192


if __name__ == '__main__':
    print(Model())
