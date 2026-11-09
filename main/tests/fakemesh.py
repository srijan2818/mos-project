"""A pure-numpy structured tet mesh with the attributes mos.core.regions expects."""
from itertools import permutations

import numpy as np


class FakeMesh:
    def __init__(self, p, t):
        self.p, self.t = p, t
        faces = np.sort(np.concatenate([t[[0, 1, 2]], t[[0, 1, 3]], t[[0, 2, 3]], t[[1, 2, 3]]], axis=1), axis=0)
        self.facets, _, counts = np.unique(faces, axis=1, return_inverse=True, return_counts=True)
        self._bnd = np.flatnonzero(counts == 1)

    def boundary_facets(self):
        return self._bnd


def box(L, H, W, nx, ny, nz):
    xs, ys, zs = (np.linspace(0, L, nx + 1), np.linspace(-H / 2, H / 2, ny + 1),
                  np.linspace(-W / 2, W / 2, nz + 1))
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    p = np.stack([X.ravel(), Y.ravel(), Z.ravel()])
    idx = lambda i, j, k: i + (nx + 1) * (j + (ny + 1) * k)
    tets = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for perm in permutations(range(3)):          # Kuhn split: 6 tets per cube
                    v, c = [0, 0, 0], [idx(i, j, k)]
                    for ax in perm:
                        v[ax] = 1
                        c.append(idx(i + v[0], j + v[1], k + v[2]))
                    tets.append(c)
    return FakeMesh(p, np.array(tets).T)
