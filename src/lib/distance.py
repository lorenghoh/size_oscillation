import numpy as np
import numba as nb


@nb.jit(nopython=True, fastmath=True)
def calc_pdist(x, y):
    pdist = np.zeros((len(x), len(y)))

    for i in range(pdist.shape[0]):
        for j in range(pdist.shape[1]):
            dx = x[j] - x[i]
            dy = y[j] - y[i]

            pdist[i, j] = np.sqrt(dx ** 2 + dy ** 2)

    return pdist


@nb.jit(nopython=True, fastmath=True)
def _ad(a, b):
    return np.abs(a - b)


@nb.jit(nopython=True, fastmath=True)
def calc_pdist_pb(a, b, nx=1536, ny=512):
    pdist = np.zeros((a.shape[1], b.shape[1]))

    for i in range(pdist.shape[0]):
        for j in range(pdist.shape[1]):
            dx = _ad(a[0, i], b[0, j])
            dx_pb = np.minimum(dx, _ad(dx, nx))

            dy = _ad(a[1, i], b[1, j])
            dy_pb = np.minimum(dy, _ad(dy, ny))

            pdist[i, j] = np.sqrt(dx_pb ** 2 + dy_pb ** 2)

    return pdist
