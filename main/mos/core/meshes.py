"""Mesh helpers (imports scikit-fem lazily)."""
import numpy as np


def box_mesh(L, H, W, nx, ny, nz, kind="tet"):
    """Beam along x: x in [0, L], y in [-H/2, H/2], z in [-W/2, W/2]."""
    from skfem import MeshHex, MeshTet
    cls = {"tet": MeshTet, "hex": MeshHex}[kind]
    return cls.init_tensor(np.linspace(0, L, nx + 1),
                           np.linspace(-H / 2, H / 2, ny + 1),
                           np.linspace(-W / 2, W / 2, nz + 1))
