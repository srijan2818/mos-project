"""Plain-text model description -> (Model, LoadCase).

One directive per line, '#' starts a comment. Units are whatever you use consistently
(SI by default: m, N, Pa). Expressions follow mos.core.fields.

  param    NAME = expr                          named constant, usable below
  mesh     box L H W nx ny nz [tet|hex]         beam along x, y,z centred on 0
  mesh     file PATH                            volume mesh via meshio (.msh .vtk .vtu .xdmf ...)
  material E = ..., nu = ..., rho = ...         isotropic; E, nu, rho also become params

  region   NAME nodes  <condition>              nodes where the condition holds
  region   NAME facets <condition>              boundary facets whose vertices all satisfy it
  region   NAME nearest x, y, z                 the single node closest to a point

  support  REGION [xyz] [= value]               fix displacement components (default xyz = 0)
  force    REGION fx, fy, fz                    total point force, shared among region nodes
  traction REGION tx, ty, tz                    force per area; components may depend on x, y, z
  pressure REGION p                             pushes into the body; may depend on x, y, z
  body     bx, by, bz                           force per volume; may depend on x, y, z
  moment   REGION mx, my, mz [about x, y, z]    bending / torque as a zero-net-force traction
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .fields import (Field, RESERVED, compile_predicate, eval_constant,
                     split_top_level)
from .loads import BodyForce, Moment, PointForce, Pressure, Traction
from .material import Material
from .model import BC, LoadCase, Model
from .regions import facets_where, nearest_node, nodes_where


class SpecError(ValueError):
    """A problem in a model description; the message names the offending line."""


@dataclass
class Spec:
    """A parsed model description (see parse_spec). Regions are resolved later, by `build`.

    params     named constants (from `param` lines plus E, nu, rho from `material`)
    mesh       ("box", L, H, W, nx, ny, nz, "tet"|"hex")  or  ("file", path)
    material   the Material
    regions    list of (line, name, kind, arg); kind is nodes / facets / nearest
    supports   list of (line, region name, component tuple, value)
    loads      list of (line, kind, region name or None, payload)
    """

    params: dict = field(default_factory=dict)
    mesh: tuple | None = None          # ("box", L, H, W, nx, ny, nz, elem) | ("file", path)
    material: Material | None = None
    regions: list = field(default_factory=list)    # (line, name, kind, arg)
    supports: list = field(default_factory=list)   # (line, region, comps, value)
    loads: list = field(default_factory=list)      # (line, kind, region|None, payload)


# ------------------------------------------------------------------ parsing helpers
def _int(v: float, what: str) -> int:
    if abs(v - round(v)) > 1e-9 or v < 1:
        raise SpecError(f"{what} must be a positive integer, got {v}")
    return int(round(v))


def _three(s: str, spec: Spec, what: str) -> list:
    parts = split_top_level(s)
    if len(parts) != 3:
        raise SpecError(f"{what} needs 3 comma-separated components, got {len(parts)}")
    return parts


def _region_and_rest(rest: str, what: str):
    m = re.fullmatch(r"(\w+)\s+(.+)", rest)
    if not m:
        raise SpecError(f"{what} REGION <values>")
    return m.group(1), m.group(2)


# ------------------------------------------------------------------ directives
def _h_param(spec, rest, n):
    m = re.fullmatch(r"([A-Za-z_]\w*)\s*=\s*(.+)", rest)      # must be a valid identifier
    if not m:
        raise SpecError("param NAME = expression")
    name = m.group(1)
    if name in RESERVED:
        raise SpecError(f"'{name}' is reserved")
    spec.params[name] = eval_constant(m.group(2), **spec.params)


def _h_mesh(spec, rest, n):
    kind, _, arg = rest.partition(" ")
    kind, arg = kind.lower(), arg.strip()
    if kind == "file":
        if not arg:
            raise SpecError("mesh file PATH")
        spec.mesh = ("file", arg)
    elif kind == "box":
        toks = arg.split()
        elem = toks.pop().lower() if toks and toks[-1].lower() in ("tet", "hex") else "tet"
        if len(toks) != 6:
            raise SpecError("mesh box L H W nx ny nz [tet|hex]")
        L, H, W = (eval_constant(t, **spec.params) for t in toks[:3])
        if min(L, H, W) <= 0:
            raise SpecError("box dimensions must be positive")
        nx, ny, nz = (_int(eval_constant(t, **spec.params), "element count") for t in toks[3:])
        spec.mesh = ("box", L, H, W, nx, ny, nz, elem)
    else:
        raise SpecError("mesh box L H W nx ny nz [tet|hex]   or   mesh file PATH")


def _h_material(spec, rest, n):
    vals = {}
    for item in split_top_level(rest):
        m = re.fullmatch(r"(\w+)\s*=\s*(.+)", item)
        if not m or m.group(1) not in ("E", "nu", "rho"):
            raise SpecError("material E = ..., nu = ..., rho = ...")
        vals[m.group(1)] = eval_constant(m.group(2), **spec.params)
    if "E" not in vals or "nu" not in vals:
        raise SpecError("material needs at least E and nu")
    vals.setdefault("rho", 0.0)
    spec.material = Material.isotropic(vals["E"], vals["nu"], vals["rho"])
    spec.params.update(vals)


def _h_region(spec, rest, n):
    m = re.fullmatch(r"(\w+)\s+(nodes|facets|nearest)\s+(.+)", rest, flags=re.I)
    if not m:
        raise SpecError("region NAME nodes|facets <condition>   or   region NAME nearest x, y, z")
    name, kind, arg = m.group(1), m.group(2).lower(), m.group(3)
    if any(r[1] == name for r in spec.regions):
        raise SpecError(f"region '{name}' is already defined")
    if kind == "nearest":
        arg = [eval_constant(e, **spec.params) for e in _three(arg, spec, "nearest")]
    else:
        arg = compile_predicate(arg, **spec.params)
    spec.regions.append((n, name, kind, arg))


def _h_support(spec, rest, n):
    m = re.fullmatch(r"(\w+)(?:\s+([xyzXYZ]+))?(?:\s*=\s*(.+))?", rest)
    if not m:
        raise SpecError("support REGION [xyz] [= value]")
    region, comps, value = m.group(1), m.group(2), m.group(3)
    if value is not None and comps is None:
        raise SpecError("say which components to prescribe, e.g.  support right y = -0.002")
    comps = tuple(sorted({"xyz".index(c) for c in (comps or "xyz").lower()}))
    value = 0.0 if value is None else eval_constant(value, **spec.params)
    spec.supports.append((n, region, comps, value))


def _h_force(spec, rest, n):
    region, args = _region_and_rest(rest, "force")
    F = [eval_constant(e, **spec.params) for e in _three(args, spec, "force")]
    spec.loads.append((n, "force", region, np.array(F)))


def _h_traction(spec, rest, n):
    region, args = _region_and_rest(rest, "traction")
    f = Field.from_expr(_three(args, spec, "traction"), **spec.params)
    spec.loads.append((n, "traction", region, f))


def _h_pressure(spec, rest, n):
    region, args = _region_and_rest(rest, "pressure")
    spec.loads.append((n, "pressure", region, Field.from_expr(args, **spec.params)))


def _h_body(spec, rest, n):
    f = Field.from_expr(_three(rest, spec, "body"), **spec.params)
    spec.loads.append((n, "body", None, f))


def _h_moment(spec, rest, n):
    region, args = _region_and_rest(rest, "moment")
    main, *about = re.split(r"\s+about\s+", args, maxsplit=1)
    origin = None
    if about:
        origin = [eval_constant(e, **spec.params) for e in _three(about[0], spec, "'about'")]

    main_lower = main.strip().lower()
    if "axis" in main_lower and "mag" in main_lower:
        m_axis = re.search(r"axis\s+(.*?)(?:\s+mag\s+|$)", main, flags=re.I)
        m_mag = re.search(r"mag\s+(.*?)(?:\s+axis\s+|$)", main, flags=re.I)
        if not m_axis or not m_mag:
            raise SpecError("moment REGION axis ax, ay, az mag M [about x, y, z]")
        axis_raw = m_axis.group(1).strip()
        mag_raw = m_mag.group(1).strip()
        axis_vals = np.array([eval_constant(e, **spec.params) for e in _three(axis_raw, spec, "moment axis")], float)
        norm = np.linalg.norm(axis_vals)
        if norm < 1e-14:
            raise SpecError("moment axis cannot be the zero vector")
        mag = eval_constant(mag_raw, **spec.params)
        M = (mag / norm) * axis_vals
    else:
        M = [eval_constant(e, **spec.params) for e in _three(main, spec, "moment")]
    spec.loads.append((n, "moment", region, (np.asarray(M, float), origin)))


_HANDLERS = {"param": _h_param, "mesh": _h_mesh, "material": _h_material,
             "region": _h_region, "support": _h_support, "force": _h_force,
             "traction": _h_traction, "pressure": _h_pressure, "body": _h_body,
             "moment": _h_moment}


def parse_spec(text: str) -> Spec:
    """Parse model-description text into a Spec; raises SpecError with the line number."""
    spec = Spec()
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        head, rest = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")
        handler = _HANDLERS.get(head)
        try:
            if handler is None:
                raise SpecError(f"unknown directive '{parts[0]}' (known: {', '.join(_HANDLERS)})")
            handler(spec, rest, n)
        except (ValueError, SyntaxError, ZeroDivisionError) as e:   # SpecError is a ValueError
            raise SpecError(f"line {n}: {e}") from None
    if spec.mesh is None:
        raise SpecError("no 'mesh' line")
    if spec.material is None:
        raise SpecError("no 'material' line")
    if not spec.supports:
        raise SpecError("no 'support' line: the body would be free to move")
    return spec


# ------------------------------------------------------------------ building
def make_mesh(spec: Spec):
    """Create the scikit-fem mesh a Spec asks for (generated box, or loaded from file)."""
    if spec.mesh[0] == "box":
        from .meshes import box_mesh
        _, L, H, W, nx, ny, nz, elem = spec.mesh
        return box_mesh(L, H, W, nx, ny, nz, kind=elem)
    from skfem import Mesh
    return Mesh.load(spec.mesh[1])


def build(spec: Spec, mesh=None):
    """Resolve regions on the mesh and create the Model and LoadCase."""
    mesh = make_mesh(spec) if mesh is None else mesh
    regions = {}
    for n, name, kind, arg in spec.regions:
        try:
            if kind == "nearest":
                reg = nearest_node(mesh, arg, name)
            elif kind == "nodes":
                reg = nodes_where(mesh, arg, name)
            else:
                reg = facets_where(mesh, arg, name)
        except ValueError as e:
            raise SpecError(f"line {n}: {e}") from None
        if len(reg) == 0:
            raise SpecError(f"line {n}: region '{name}' is empty - no {kind} satisfy its condition "
                            "(tip: use abs(x - L) < 1e-6 rather than x == L)")
        regions[name] = reg

    def get(n, name):
        """Look up a region by name, or raise SpecError naming line n."""
        if name not in regions:
            raise SpecError(f"line {n}: unknown region '{name}' (defined: {', '.join(regions) or 'none'})")
        return regions[name]

    try:
        bcs = [BC(get(n, r), comps, v) for n, r, comps, v in spec.supports]
        loads = []
        for n, kind, r, payload in spec.loads:
            try:
                if kind == "body":
                    loads.append(BodyForce(payload))
                elif kind == "force":
                    loads.append(PointForce(get(n, r), payload))
                elif kind == "traction":
                    loads.append(Traction(get(n, r), payload))
                elif kind == "pressure":
                    loads.append(Pressure(get(n, r), payload))
                else:
                    loads.append(Moment(get(n, r), payload[0], payload[1]))
            except SpecError:
                raise
            except ValueError as e:
                raise SpecError(f"line {n}: {kind} on region '{r}': {e}") from None
    except SpecError:
        raise
    return Model(mesh, spec.material, bcs), LoadCase("spec", loads)
