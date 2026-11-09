"""Point-driven stress analysis adapters for solved FEA results."""
from __future__ import annotations

import numpy as np

from mos.core import post


PA_PER_MPA = 1.0e6


def evaluate_stress(results, points, xyz, where="nodal"):
    """Sample a stress tensor and invariants at the nearest mesh node.

    Results remain in SI (Pa); the returned tensor and scalar stresses are Pa.
    """
    fields = {"nodal": results.sig_nodal, "cell": results.sig_cell, "q": results.sig_q}
    if where not in fields:
        raise ValueError("where must be 'nodal', 'cell', or 'q'")
    tensor, node_index = post.stress_at_point(fields[where], points, xyz)
    values, vectors = post.principal(tensor.reshape(3, 3, 1, 1))
    values, vectors = values.reshape(3), vectors.reshape(3, 3)
    return {
        "tensor": tensor, "node_index": node_index,
        "principal": values, "directions": vectors,
        "von_mises": float(post.von_mises(tensor)),
        "tresca": float(values[0] - values[2]),
        "max_shear": float((values[0] - values[2]) / 2.0),
    }
