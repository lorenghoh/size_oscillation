nx = 1728
ny = 576
nz = 192


def index_to_zyx(index):
    z = index // (ny * nx)
    index = index % (ny * nx)
    y = index // nx
    x = index % nx
    return z, y, x


def zyx_to_index(z, y, x):
    return ny * nx * z + nx * y + x
