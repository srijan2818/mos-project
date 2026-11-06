"""Model = mesh + one material + boundary conditions; LoadCases are solved against it."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .material import Material
from .regions import Region


@dataclass
class BC:
    """Prescribe displacement components on a node or facet region.
    components: subset of (0, 1, 2) for x, y, z.  value: number, or a vector Field."""
    region: Region
    components: tuple = (0, 1, 2)
    value: object = 0.0


@dataclass
class LoadCase:
    """A named list of loads (PointForce, SurfaceForce, Traction, Pressure, BodyForce, Moment)."""

    name: str = "case"
    loads: List[object] = field(default_factory=list)


@dataclass
class Model:
    """Mesh + one material + boundary conditions. Solve it with mos.core.solver.solve_case."""

    mesh: object             # a scikit-fem mesh (MeshTet / MeshHex)
    material: Material
    bcs: List[BC] = field(default_factory=list)
