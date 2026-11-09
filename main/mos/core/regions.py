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
        raw = np.asarray(self.ids)
        if raw.ndim != 1:
            raise ValueError("region ids must be a one-dimensional index array")
        if raw.size and not np.issubdtype(raw.dtype, np.integer):
            if not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all() \
                    or not np.equal(raw, np.floor(raw)).all():
                raise ValueError("region ids must be finite integer indices")
        ids = raw.astype(int, copy=False)
        if np.any(ids < 0):
            raise ValueError("region ids must be non-negative")
        # A region is a set of indices. Deduplicate here so repeated IDs cannot cause
        # advanced-indexing updates to lose force contributions or double-integrate facets.
        self.ids = np.unique(ids)

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




def outward_facet_normals(mesh, facet_ids) -> np.ndarray:
    """Area-weighted outward normals (3, n) of boundary triangle facets.

    ``mesh.facets`` stores sorted vertex indices, so the raw cross product has an
    arbitrary sign per facet. Each normal is flipped to point away from the centroid of
    the element that owns the facet. The vector length is twice the facet area.
    """
    ids = np.asarray(facet_ids, dtype=int)
    tri = mesh.p[:, mesh.facets[:3, ids]]                   # (3, 3 vertices, n)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], axis=0)
    f2t = getattr(mesh, "f2t", None)
    if f2t is not None:
        owner = np.asarray(f2t)[0, ids]
        inside = mesh.p[:, mesh.t[:, owner]].mean(axis=1)   # owning element centroids
    else:                                                   # minimal mesh stand-ins: body centroid
        inside = np.asarray(mesh.p, float).mean(axis=1)[:, None]
    outward = tri.mean(axis=1) - inside
    return n * np.where(np.einsum("ij,ij->j", n, outward) < 0.0, -1.0, 1.0)


def boundary_normal(mesh, facet_ids) -> np.ndarray:
    """Unit area-weighted mean outward normal of a boundary facet set ((1,0,0) if degenerate)."""
    total = outward_facet_normals(mesh, facet_ids).sum(axis=1)
    norm = float(np.linalg.norm(total))
    return total / norm if norm > 1e-300 else np.array([1.0, 0.0, 0.0])


_CAD_SIDE_ALIASES = {
    "neg_x": (0, False), "pos_x": (0, True),
    "neg_y": (1, False), "pos_y": (1, True),
    "neg_z": (2, False), "pos_z": (2, True),
    "-x": (0, False), "+x": (0, True),
    "-y": (1, False), "+y": (1, True),
    "-z": (2, False), "+z": (2, True),
}


def cad_boundary_name_for_side(mesh, side: str) -> str | None:
    """Return the named CAD boundary furthest toward an axis-aligned side.

    This is a compatibility mapping for UI aliases such as ``neg_x``/``-x``.
    It uses each named boundary's facet centroid, so it works when a STEP
    model's coordinates are offset from the origin. Returns ``None`` when the
    mesh has no usable named boundaries or ``side`` is not a recognized alias.
    """
    choice = _CAD_SIDE_ALIASES.get(str(side).lower())
    boundaries = getattr(mesh, "boundaries", None)
    if choice is None or not boundaries:
        return None
    axis, want_max = choice
    points = np.asarray(mesh.p, dtype=float)
    facets = np.asarray(mesh.facets, dtype=int)
    candidates = []
    for name, raw_ids in boundaries.items():
        ids = np.asarray(raw_ids, dtype=int)
        if ids.ndim != 1 or ids.size == 0 or np.any(ids < 0) or np.any(ids >= facets.shape[1]):
            continue
        vertices = facets[:, ids]
        centroid = points[:, vertices].mean(axis=(1, 2))
        if np.isfinite(centroid[axis]):
            candidates.append((float(centroid[axis]), name))
    if not candidates:
        return None
    return (max if want_max else min)(candidates, key=lambda item: item[0])[1]


def facets_where(mesh, pred: Callable, name: str) -> Region:
    """Boundary facets whose vertices ALL satisfy pred."""
    node_ok = np.asarray(pred(*mesh.p), dtype=bool)
    bf = np.asarray(mesh.boundary_facets())
    ok = node_ok[mesh.facets[:, bf]].all(axis=0)
    return Region(name, "facets", bf[ok])
