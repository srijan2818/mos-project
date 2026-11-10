"""Plain-text model description -> (Model, LoadCase).

One directive per line, '#' starts a comment. Units are whatever you use consistently
(SI by default: m, N, Pa). Expressions follow mos.core.fields.

  param    NAME = expr                          named constant, usable below
  mesh     box L H W nx ny nz [tet|hex]         beam along x, y,z centred on 0
  mesh     file PATH                            volume mesh via meshio (.msh .vtk .vtu .xdmf ...)
  material E = ..., nu = ..., rho = ...         isotropic; E, nu, rho also become params
           [, sigma_pl = ...[, sigma_pl_c = ...]]  proportional limit(s), Pa (checked after solve)

  region   NAME nodes  <condition>              nodes where the condition holds
  region   NAME facets <condition>              boundary facets whose vertices all satisfy it
  region   NAME nearest x, y, z                 the single node closest to a point

  support  REGION [xyz] [= value]               fix displacement components (default xyz = 0)
  force    REGION fx, fy, fz                    total force (nodes: shared nodal force; facets: uniform resultant)
  traction REGION tx, ty, tz                    force per area; components may depend on x, y, z
  pressure REGION p                             pushes into the body; may depend on x, y, z
  body     bx, by, bz                           force per volume; may depend on x, y, z
  moment   REGION mx, my, mz [about x, y, z]    bending / torque as a zero-net-force traction
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import numpy as np

from .fields import (Field, RESERVED, compile_predicate, eval_constant,
                     split_top_level)
from .loads import BodyForce, Moment, PointForce, Pressure, SurfaceForce, Traction
from .material import Material
from .model import BC, LoadCase, Model
from .regions import (Region, cad_boundary_name_for_side, facets_where,
                      nearest_node, nodes_where)

log = logging.getLogger(__name__)


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
    m = re.fullmatch(r"([^\s]+)\s+(.+)", rest)
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
    if spec.material is not None and name in ("E", "nu", "rho", "sigma_pl", "sigma_pl_c"):
        # The material is already built; a later value would change expressions only.
        raise SpecError(f"param {name} must come before the material line")
    spec.params[name] = eval_constant(m.group(2), **spec.params)


def _split_step_path_and_size(arg: str, params: dict) -> tuple[str, float | None]:
    """Split a STEP path from its optional mesh size, allowing spaces in paths.

    Quoted paths may use single or double quotes. Backslashes are preserved except
    when escaping the active quote or another backslash, so Windows paths work too.
    For unquoted paths, the STEP suffix identifies the end of the path.
    """
    arg = arg.strip()
    if not arg:
        raise SpecError("mesh step PATH [mesh_size]")

    if arg[0] in ("'", '"'):
        quote = arg[0]
        chars = []
        i = 1
        while i < len(arg):
            char = arg[i]
            if char == "\\" and i + 1 < len(arg) and arg[i + 1] in (quote, "\\"):
                chars.append(arg[i + 1])
                i += 2
                continue
            if char == quote:
                path = "".join(chars)
                tail = arg[i + 1:].strip()
                if not path:
                    raise SpecError("STEP path cannot be empty")
                mesh_size = float(eval_constant(tail, **params)) if tail else None
                return path, mesh_size
            chars.append(char)
            i += 1
        raise SpecError("unterminated quoted STEP path")

    # A STEP suffix followed by whitespace marks the end of an unquoted path.
    match = re.match(r"^(.*?\.(?:step|stp))(?=\s|$)(?:\s+(.*))?$", arg, flags=re.I)
    if match:
        path = match.group(1).strip()
        tail = (match.group(2) or "").strip()
        mesh_size = float(eval_constant(tail, **params)) if tail else None
        return path, mesh_size

    # Backward-compatible fallback for extensionless paths, including optional sizes.
    pieces = arg.rsplit(None, 1)
    if len(pieces) == 2:
        try:
            mesh_size = float(eval_constant(pieces[1], **params))
        except (ValueError, SyntaxError, ZeroDivisionError):
            pass
        else:
            return pieces[0], mesh_size
    return arg, None


_MASK_USAGE = "mesh mask PATH size thickness [mesh_size]"


def _split_mask_args(arg: str, params: dict) -> tuple:
    """Parse ``PATH size thickness [mesh_size]`` for ``mesh mask`` (PATH may be quoted)."""
    arg = arg.strip()
    if not arg:
        raise SpecError(_MASK_USAGE)
    if arg[0] in ("'", '"'):
        quote, chars, i = arg[0], [], 1
        while i < len(arg):
            if arg[i] == "\\" and i + 1 < len(arg) and arg[i + 1] in (quote, "\\"):
                chars.append(arg[i + 1])
                i += 2
                continue
            if arg[i] == quote:
                break
            chars.append(arg[i])
            i += 1
        else:
            raise SpecError("unterminated quoted mask path")
        path, tail = "".join(chars), arg[i + 1:]
    else:
        m = re.match(r"^(.*?\.(?:png|jpe?g|bmp|gif|tiff?|webp))(?=\s|$)(.*)$", arg, flags=re.I)
        if m:
            path, tail = m.group(1).strip(), m.group(2)
        else:
            path, _, tail = arg.partition(" ")
    if not path:
        raise SpecError("mask path cannot be empty")
    toks = tail.split()
    if len(toks) not in (2, 3):
        raise SpecError(_MASK_USAGE)
    vals = [float(eval_constant(t, **params)) for t in toks]
    if vals[0] <= 0 or vals[1] <= 0 or (len(vals) == 3 and vals[2] <= 0):
        raise SpecError("mask size, thickness and mesh size must be positive")
    return path, vals[0], vals[1], (vals[2] if len(vals) == 3 else None)


def _h_mesh(spec, rest, n):
    kind, _, arg = rest.partition(" ")
    kind, arg = kind.lower(), arg.strip()
    if kind == "file":
        if not arg:
            raise SpecError("mesh file PATH")
        spec.mesh = ("file", arg)
    elif kind == "step":
        step_path, ms = _split_step_path_and_size(arg, spec.params)
        spec.mesh = ("step", step_path, ms)
    elif kind == "mask":
        spec.mesh = ("mask",) + _split_mask_args(arg, spec.params)
    elif kind == "hull":
        if not arg:
            raise SpecError("mesh hull SESSION_DIR [mesh_size]")
        session, ms = _split_step_path_and_size(arg, spec.params)
        spec.mesh = ("hull", session, ms)
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
        raise SpecError("mesh box L H W nx ny nz [tet|hex]   or   mesh file PATH   or   "
                       "mesh step PATH [mesh_size]   or   mesh hull SESSION_DIR [mesh_size]   or   "
                       + _MASK_USAGE)


def _h_material(spec, rest, n):
    vals = {}
    for item in split_top_level(rest):
        m = re.fullmatch(r"(\w+)\s*=\s*(.+)", item)
        if not m or m.group(1) not in ("E", "nu", "rho", "sigma_pl", "sigma_pl_c"):
            raise SpecError("material E = ..., nu = ..., rho = ...[, sigma_pl = ...[, sigma_pl_c = ...]]")
        vals[m.group(1)] = eval_constant(m.group(2), **spec.params)
    if "E" not in vals or "nu" not in vals:
        raise SpecError("material needs at least E and nu")
    vals.setdefault("rho", 0.0)
    try:
        spec.material = Material.isotropic(vals["E"], vals["nu"], vals["rho"],
                                           vals.get("sigma_pl"), vals.get("sigma_pl_c"))
    except ValueError as exc:
        raise SpecError(f"material: {exc}") from exc
    spec.params.update(vals)


def _h_region(spec, rest, n):
    m = re.fullmatch(r"([^\s]+)\s+(nodes|facets|nearest)\s+(.+)", rest, flags=re.I)
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
    m = re.fullmatch(r"([^\s]+)(?:\s+([xyzXYZ]+))?(?:\s*=\s*(.+))?", rest)
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
            if head in ("mesh", "material") and getattr(spec, head) is not None:
                raise SpecError(f"a model has exactly one '{head}' line; this is a second one")
            handler(spec, rest, n)
        except RecursionError:
            raise SpecError(f"line {n}: expression is nested too deeply") from None
        except (ValueError, SyntaxError, ZeroDivisionError) as e:   # SpecError is a ValueError
            raise SpecError(f"line {n}: {e}") from None
    if spec.mesh is None:
        raise SpecError("no 'mesh' line")
    if spec.material is None:
        raise SpecError("no 'material' line")
    return spec


# ------------------------------------------------------------------ building
def coarsen_mesh_text(text: str, factor: float, mesh=None) -> str:
    """``text`` with its ``mesh`` line asking for elements ``factor`` (> 1) times larger.

    ``box``: element counts divided by ``factor`` (at least 1). ``step`` / ``mask`` /
    ``hull``: mesh size multiplied by ``factor``; without a size in the line, the current
    ``mesh``'s mean edge length is the starting size. A ``file`` mesh cannot be re-meshed.
    The quadratic basis has ~ (1 / h)^3 DOFs, so ``factor = (n / n_target)^(1/3)``
    reaches about ``n_target``; Gmsh size bounds and minimum resolutions make it approximate.
    """
    if not factor > 1.0:
        raise ValueError("coarsening factor must be > 1")
    m = parse_spec(text).mesh
    kind = m[0]
    if kind == "box":
        _, L, H, W, nx, ny, nz, elem = m
        counts = [max(1, int(n / factor)) for n in (nx, ny, nz)]
        line = f"mesh box {L:.12g} {H:.12g} {W:.12g} {counts[0]} {counts[1]} {counts[2]} {elem}"
    elif kind in ("step", "hull", "mask"):
        ms = m[-1]
        if ms is None:
            if mesh is None:
                raise ValueError("the mesh line has no size: pass the current mesh")
            from .meshes import mean_edge_length
            ms = mean_edge_length(mesh)
        h = f"{ms * factor:.4g}"
        line = (f"mesh mask {json.dumps(m[1])} {m[2]:.12g} {m[3]:.12g} {h}" if kind == "mask"
                else f"mesh {kind} {json.dumps(m[1])} {h}")
    else:
        raise SpecError("a 'mesh file' mesh cannot be coarsened automatically; remesh the file")
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        body = ln.split("#", 1)[0].strip()
        if body.split(None, 1)[:1] and body.split(None, 1)[0].lower() == "mesh":
            comment = ln.split("#", 1)[1] if "#" in ln else None
            lines[i] = line + (f"   #{comment}" if comment is not None else "")
            return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    raise SpecError("no 'mesh' line")


def make_mesh(spec: Spec):
    """Create the scikit-fem mesh a Spec asks for (generated box, or loaded from file)."""
    if spec.mesh[0] == "box":
        from .meshes import box_mesh
        _, L, H, W, nx, ny, nz, elem = spec.mesh
        return box_mesh(L, H, W, nx, ny, nz, kind=elem)
    elif spec.mesh[0] == "step":
        from .cad import load_step_mesh
        _, step_path, ms = spec.mesh
        m, _, _ = load_step_mesh(step_path, mesh_size=ms)
        return m
    elif spec.mesh[0] == "mask":
        from .image_to_mesh import load_mask_mesh
        _, mask_path, size, thickness, ms = spec.mesh
        m, _ = load_mask_mesh(mask_path, size, thickness, ms)
        return m
    elif spec.mesh[0] == "hull":
        from .multiview_hull import load_hull_mesh
        _, session, ms = spec.mesh
        return load_hull_mesh(session, ms)
    elif spec.mesh[0] == "file":
        from .cad import is_step_file, load_step_mesh
        path = spec.mesh[1]
        if is_step_file(path):
            m, _, _ = load_step_mesh(path)
            return m
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
        if kind == "facets" and hasattr(mesh, "boundaries"):
            # The names neg_x, pos_x, ... are reserved on meshes with named boundaries.
            # Older generated specs carry predicates from a previous model's bounds, which
            # can match nothing or (offset STEP parts) the whole boundary, so the named
            # boundary furthest toward that side is used. An override is logged.
            cad_name = cad_boundary_name_for_side(mesh, name)
            if cad_name is not None:
                named = np.asarray(mesh.boundaries[cad_name])
                if len(reg) and not np.array_equal(np.sort(reg.ids), np.sort(named)):
                    log.warning("line %d: region '%s' is a reserved side alias; using boundary "
                                "'%s' instead of its condition", n, name, cad_name)
                reg = Region(name, "facets", named)
        if len(reg) == 0:
            raise SpecError(f"line {n}: region '{name}' is empty - no {kind} satisfy its condition "
                            "(tip: use abs(x - L) < 1e-6 rather than x == L)")
        regions[name] = reg

    def get(n, name):
        """Look up a region by name, or check named mesh boundaries, or raise SpecError naming line n."""
        if name not in regions:
            if hasattr(mesh, "boundaries") and mesh.boundaries and name in mesh.boundaries:
                return Region(name, "facets", mesh.boundaries[name])
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
                    reg = get(n, r)
                    if reg.kind == "nodes":
                        loads.append(PointForce(reg, payload))
                    elif reg.kind == "facets":
                        loads.append(SurfaceForce(reg, payload))
                    else:
                        raise SpecError(f"line {n}: force needs a node or facet region")
                elif kind == "traction":
                    loads.append(Traction(get(n, r), payload))
                elif kind == "pressure":
                    loads.append(Pressure(get(n, r), payload))
                else:
                    reg = get(n, r)
                    origin = payload[1]
                    if reg.kind == "nodes":
                        # If moment is applied to a node region (e.g. nearest point), adapt to adjacent boundary surface patch
                        bf = np.asarray(mesh.boundary_facets())
                        if len(bf) > 0 and hasattr(mesh, "facets") and mesh.facets is not None:
                            tri_nodes = mesh.facets[:, bf]
                            has_node = np.isin(tri_nodes, reg.ids).any(axis=0)
                            cand_bf = bf[has_node]
                            if len(cand_bf) > 0:
                                if origin is None:
                                    origin = np.mean(mesh.p[:, reg.ids], axis=1)
                                patch_reg = Region(f"{r}_patch", "facets", cand_bf)
                                loads.append(Moment(patch_reg, payload[0], origin))
                            else:
                                loads.append(Moment(reg, payload[0], origin))
                        else:
                            loads.append(Moment(reg, payload[0], origin))
                    else:
                        loads.append(Moment(reg, payload[0], origin))
            except SpecError:
                raise
            except ValueError as e:
                raise SpecError(f"line {n}: {kind} on region '{r}': {e}") from None
    except SpecError:
        raise
    return Model(mesh, spec.material, bcs), LoadCase("spec", loads)
