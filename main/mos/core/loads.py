"""Declarative loads. They are plain data; the solver turns them into force vectors.

  PointForce  node region    total force F (N), shared equally among the region's nodes
  SurfaceForce facet region  total force F (N), distributed uniformly over the face set
  Traction    facet region   force per area (Pa), constant or Field
  Pressure    facet region   p (Pa) acting INTO the body (t = -p n), constant or Field
  BodyForce   whole body     force per volume (N/m^3), constant or Field
  Moment      facet region   moment M (N m), applied as a pure couple with zero net force;
                             optional `origin` is a reference point for reporting / visualization.
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
    """Return a pure-couple traction field for the requested resultant moment.

    X: (3, N) quadrature points on the face set, w: (N,) area weights.

    The traction is ``t(x) = omega x (x - c)``, where ``c`` is the area centroid and
    ``A omega = M`` for ``A = integral(|r|^2 I - r r^T) dA``.  Centering at the area
    centroid makes the resultant force zero.  A pure couple's resultant moment is
    independent of the point about which it is reported, so an explicit ``origin``
    must not shift the traction field or introduce a spurious net force.
    """
    X = np.asarray(X, float)
    w = np.asarray(w, float)
    if X.ndim != 2 or X.shape[0] != 3 or w.shape != (X.shape[1],):
        raise ValueError("X must have shape (3, N) and w must have shape (N,)")
    if X.shape[1] == 0 or not np.isfinite(X).all() or not np.isfinite(w).all():
        raise ValueError("moment quadrature points and weights must be non-empty and finite")
    area = float(w.sum())
    if area <= 0.0 or np.any(w < 0.0):
        raise ValueError("moment face set must have positive area weights")
    M = np.asarray(M, float)
    if M.shape != (3,) or not np.isfinite(M).all():
        raise ValueError("M must be a finite 3-vector")
    if origin is not None:
        origin = np.asarray(origin, float)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("moment origin must be a finite 3-vector")
    # A couple has no unique point of application: its moment is a free vector.
    # Always center the distribution at the area centroid, even if the text model
    # includes `about x, y, z`.  Using that point directly would create a net force.
    o = (X * w).sum(axis=1) / area
    r = X - o[:, None]
    A = (w * (r * r).sum(axis=0)).sum() * np.eye(3) - (r * w) @ r.T
    if np.linalg.cond(A) > 1e12:
        raise ValueError("face set is degenerate for a moment (zero or line-like area)")
    return np.linalg.solve(A, M), o


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
        if not np.isfinite(self.force).all():
            raise ValueError("PointForce must be finite")


@dataclass
class SurfaceForce:
    """Total force `force` (N), distributed as uniform traction over a facet region."""

    region: Region
    force: object

    def __post_init__(self):
        _need_facets(self.region)
        self.force = np.asarray(self.force, float)
        if self.force.shape != (3,) or not np.isfinite(self.force).all():
            raise ValueError("SurfaceForce needs a finite 3-vector")


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
    """Pure-couple moment M (N m) on a facet region.

    ``origin`` is an optional reference point for reporting/visualization only: a pure couple's
    resultant moment is independent of reference point and must have zero resultant force.
    Torsion is a moment along the face normal.
    """

    region: Region
    M: object                # 3-vector, N m
    origin: Optional[object] = None

    def __post_init__(self):
        _need_facets(self.region)
        self.M = np.asarray(self.M, float)
        if self.M.shape != (3,) or not np.isfinite(self.M).all():
            raise ValueError("M must be a finite 3-vector")
        if self.origin is not None:
            self.origin = np.asarray(self.origin, float)
            if self.origin.shape != (3,) or not np.isfinite(self.origin).all():
                raise ValueError("moment origin must be a finite 3-vector")


def _need_facets(region):
    if region.kind != "facets" or len(region) == 0:
        raise ValueError("this load needs a non-empty facet region")
