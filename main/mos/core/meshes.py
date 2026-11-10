"""Mesh helpers (imports scikit-fem lazily)."""
import numpy as np


def box_mesh(L, H, W, nx, ny, nz, kind="tet"):
    """Beam along x: x in [0, L], y in [-H/2, H/2], z in [-W/2, W/2]."""
    from skfem import MeshHex, MeshTet
    cls = {"tet": MeshTet, "hex": MeshHex}[kind]
    return cls.init_tensor(np.linspace(0, L, nx + 1),
                           np.linspace(-H / 2, H / 2, ny + 1),
                           np.linspace(-W / 2, W / 2, nz + 1))


def mean_edge_length(mesh) -> float:
    """Mean length of the cells' vertex-to-vertex edges (tets: 6 per cell, hexes: the
    12 box edges), the mesh's characteristic element size."""
    t, P = np.asarray(mesh.t), np.asarray(mesh.p)
    pairs = ([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] if t.shape[0] == 4 else
             [tuple(e) for e in _HEX_EDGES])
    return float(np.mean([np.linalg.norm(P[:, t[a]] - P[:, t[b]], axis=0).mean() for a, b in pairs]))


# scikit-fem MeshHex local vertex order: 0 (0,0,0), 1 (0,1,0), 2 (1,0,0), 3 (0,0,1),
# 4 (1,1,0), 5 (0,1,1), 6 (1,0,1), 7 (1,1,1): edges join vertices differing in one coordinate
_HEX_EDGES = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
