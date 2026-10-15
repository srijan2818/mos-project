"""Core library: model objects, text-spec parser, post-processing. The FE solver is mos.core.solver."""
from .fields import Field
from .loads import BodyForce, Moment, PointForce, Pressure, SurfaceForce, Traction
from .material import Material
from .model import BC, LoadCase, Model
from .regions import Region, facets_where, nearest_node, nodes_where

# The solver needs scikit-fem:  from mos.core.solver import solve_case
