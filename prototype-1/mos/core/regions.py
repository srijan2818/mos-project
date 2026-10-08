"""Regions: named labels pointing at a subset of ONE mesh (the mesh itself is never split).

kind = "nodes"  -> node indices   (point forces, supports)
kind = "facets" -> boundary facet indices (tractions, pressure, moments, supports)

Selectors take any scikit-fem-like mesh: mesh.p (3, nnodes), mesh.facets (nv, nfacets),
mesh.boundary_facets().  Predicates receive coordinate arrays: pred(x, y, z) -> bool array.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Region:
    """A named subset of ONE mesh (the mesh is never split).

    name  label used by supports and loads
    kind  'nodes' (node indices) or 'facets' (boundary-facet indices)
    ids   integer index array into the mesh's nodes or facets
    """

    name: str
    kind: str            # "nodes" | "facets"
    ids: np.ndarray

    def __post_init__(self):
        if self.kind not in ("nodes", "facets"):
            raise ValueError("kind must be 'nodes' or 'facets'")
        self.ids = np.asarray(self.ids, dtype=int)

    def __len__(self):
        return len(self.ids)


def nodes_where(mesh, pred: Callable, name: str) -> Region:
    """Node region: all nodes where pred(x, y, z) is true."""
    mask = np.asarray(pred(*mesh.p), dtype=bool)
    return Region(name, "nodes", np.flatnonzero(mask))


def nearest_node(mesh, point, name: str) -> Region:
    """Node region holding the single node closest to `point`."""
    d = np.linalg.norm(mesh.p - np.asarray(point, float)[:, None], axis=0)
    return Region(name, "nodes", [int(np.argmin(d))])


def facets_where(mesh, pred: Callable, name: str) -> Region:
    """Boundary facets whose vertices ALL satisfy pred."""
    node_ok = np.asarray(pred(*mesh.p), dtype=bool)
    bf = np.asarray(mesh.boundary_facets())
    ok = node_ok[mesh.facets[:, bf]].all(axis=0)
    return Region(name, "facets", bf[ok])
