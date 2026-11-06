"""Arbitrary-plane traction adapters for solved FEA results."""
from __future__ import annotations

import numpy as np


def evaluate_traction(results, points, xyz, normal, where="nodal"):
    """Evaluate traction decomposition at a nearest mesh node, in SI units."""
    n = np.asarray(normal, dtype=float).reshape(3)
    if not np.isfinite(n).all() or np.linalg.norm(n) < 1e-14:
        raise ValueError("plane normal must be a finite, non-zero 3-vector")
    return results.traction_on_plane(xyz, n, points, where=where)
