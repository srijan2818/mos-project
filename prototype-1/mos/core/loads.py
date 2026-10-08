"""Declarative loads. They are plain data; the solver turns them into force vectors.

  PointForce  node region    total force F (N), shared equally among the region's nodes
  Traction    facet region   force per area (Pa), constant or Field
  Pressure    facet region   p (Pa) acting INTO the body (t = -p n), constant or Field
  BodyForce   whole body     force per volume (N/m^3), constant or Field
  Moment      facet region   moment M (N m) about `origin` (default: facet-set centroid),
                             applied as a traction field with zero net force.
                             Torsion is just a Moment along the face normal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .fields import as_field
from .regions import Region


def cross0(a, b) -> np.ndarray:
    """Cross product along axis 0. a: (3,) or (3, ...); b: (3, ...)."""
    b = np.asarray(b, float)
    a = np.asarray(a, float).reshape((3,) + (1,) * (b.ndim - 1))
    return np.stack([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def moment_coefficients(X, w, M, origin=None):
    """Traction t(x) = omega x (x - o) with  A omega = M,  A = sum w (|r|^2 I - r r^T).

    X: (3, N) quadrature points on the face set, w: (N,) area weights.
    Returns (omega, origin).  Net moment about `origin` is exactly M; net force is zero
    when origin is the area centroid (the default).
    """
    X = np.asarray(X, float)
    w = np.asarray(w, float)
    o = (X * w).sum(axis=1) / w.sum() if origin is None else np.asarray(origin, float)
    r = X - o[:, None]
    A = (w * (r * r).sum(axis=0)).sum() * np.eye(3) - (r * w) @ r.T
    if np.linalg.cond(A) > 1e12:
        raise ValueError("face set is degenerate for a moment (zero or line-like area)")
    return np.linalg.solve(A, np.asarray(M, float)), o


@dataclass
class PointForce:
    """Total force `force` (N) on a node region, shared equally among its nodes."""

    region: Region
    force: object            # 3-vector

    def __post_init__(self):
        if self.region.kind != "nodes":
            raise ValueError("PointForce needs a node region")
        self.force = np.asarray(self.force, float)
        if self.force.shape != (3,) or len(self.region) == 0:
            raise ValueError("PointForce needs a 3-vector and a non-empty region")


@dataclass
class Traction:
    """Force per area (Pa) on a facet region: a 3-vector or a vector Field."""

    region: Region
    field: object            # vector Field or 3-vector

    def __post_init__(self):
        _need_facets(self.region)
        self.field = as_field(self.field, vector=True)


@dataclass
class Pressure:
    """Pressure p (Pa) on a facet region, acting into the body (traction = -p n)."""

    region: Region
    p: object                # scalar Field or number

    def __post_init__(self):
        _need_facets(self.region)
        self.p = as_field(self.p, vector=False)


@dataclass
class BodyForce:
    """Force per volume (N/m^3) on the whole body: a 3-vector or a vector Field."""

    field: object            # vector Field or 3-vector, N/m^3

    def __post_init__(self):
        self.field = as_field(self.field, vector=True)


@dataclass
class Moment:
    """Moment M (N m) about `origin` (default: facet-set centroid) on a facet region,
    applied as a zero-net-force traction field. Torsion = a moment along the face normal."""

    region: Region
    M: object                # 3-vector, N m
    origin: Optional[object] = None

    def __post_init__(self):
        _need_facets(self.region)
        self.M = np.asarray(self.M, float)
        if self.M.shape != (3,):
            raise ValueError("M must be a 3-vector")


def _need_facets(region):
    if region.kind != "facets" or len(region) == 0:
        raise ValueError("this load needs a non-empty facet region")
