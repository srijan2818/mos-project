"""Parametric 3D CAD STEP solid generators using Gmsh OpenCASCADE.

Generates watertight, standard engineering 3D CAD solids for:
- L-Bracket (with mounting holes and arms)
- Tensile plate with central or offset hole
- Rectangular / Box beam
- Structural I-Beam (flanged section)
- Stepped or keyed cylindrical shaft
- Hollow circular pipe / tube
- Custom polygonal extruded profiles
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import functools

from .gmsh_session import GMSH_LOCK, HAS_GMSH, ensure_gmsh, gmsh


def _ensure_gmsh():
    try:
        ensure_gmsh()
    except RuntimeError as exc:
        raise RuntimeError(f"Gmsh OpenCASCADE is required for parametric CAD generation: {exc}") from None


def _locked(fn):
    """Run a generator while holding the process-wide Gmsh lock (Gmsh is not thread-safe)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        """Call the generator with the Gmsh lock held."""
        with GMSH_LOCK:
            return fn(*args, **kwargs)
    return wrapper


@_locked
def generate_bracket_step(
    output_path: Union[str, Path],
    leg_x: float = 0.20,
    leg_y: float = 0.20,
    width: float = 0.06,
    thickness: float = 0.015,
    hole_radius: Optional[float] = 0.008,
    hole_x: Optional[float] = None,
    hole_y: Optional[float] = None,
) -> Path:
    """Generate an L-shaped mounting bracket solid with optional bolt holes."""
    _ensure_gmsh()
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    gmsh.clear()
    gmsh.model.add(f"bracket_{out.stem}")

    # Horizontal arm along X
    b1 = gmsh.model.occ.addBox(0.0, 0.0, 0.0, float(leg_x), float(width), float(thickness))
    # Vertical arm along Z
    b2 = gmsh.model.occ.addBox(0.0, 0.0, 0.0, float(thickness), float(width), float(leg_y))
    fused, _ = gmsh.model.occ.fuse([(3, b1)], [(3, b2)])

    tool_cuts = []
    if hole_radius and hole_radius > 1e-4:
        hr = float(hole_radius)
        # Hole on horizontal leg
        hx = float(hole_x) if hole_x is not None else float(thickness + (leg_x - thickness) * 0.6)
        hy = float(width / 2.0)
        c1 = gmsh.model.occ.addCylinder(hx, hy, -0.01, 0.0, 0.0, float(thickness + 0.02), hr)
        tool_cuts.append((3, c1))

        # Hole on vertical leg
        vz = float(hole_y) if hole_y is not None else float(thickness + (leg_y - thickness) * 0.6)
        c2 = gmsh.model.occ.addCylinder(-0.01, hy, vz, float(thickness + 0.02), 0.0, 0.0, hr)
        tool_cuts.append((3, c2))

    if tool_cuts:
        gmsh.model.occ.cut(fused, tool_cuts)

    gmsh.model.occ.synchronize()
    gmsh.write(str(out))
    gmsh.clear()
    return out


@_locked
def generate_plate_hole_step(
    output_path: Union[str, Path],
    length: float = 0.40,
    width: float = 0.20,
    thickness: float = 0.02,
    hole_diameter: float = 0.06,
    hole_center_x: Optional[float] = None,
    hole_center_y: Optional[float] = None,
) -> Path:
    """Generate a flat plate with a through hole."""
    _ensure_gmsh()
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    gmsh.clear()
    gmsh.model.add(f"plate_hole_{out.stem}")

    L = float(length)
    W = float(width)
    T = float(thickness)
    d = float(hole_diameter)
    cx = float(hole_center_x) if hole_center_x is not None else L / 2.0
    cy = float(hole_center_y) if hole_center_y is not None else W / 2.0

    plate = gmsh.model.occ.addBox(0.0, 0.0, 0.0, L, W, T)
    if d > 1e-4:
        hole = gmsh.model.occ.addCylinder(cx, cy, -0.01, 0.0, 0.0, T + 0.02, d / 2.0)
        gmsh.model.occ.cut([(3, plate)], [(3, hole)])

    gmsh.model.occ.synchronize()
    gmsh.write(str(out))
    gmsh.clear()
    return out


@_locked
def generate_beam_step(
    output_path: Union[str, Path],
    length: float = 1.0,
    height: float = 0.15,
    width: float = 0.10,
) -> Path:
    """Generate a solid rectangular beam / bar."""
    _ensure_gmsh()
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    gmsh.clear()
    gmsh.model.add(f"beam_{out.stem}")
    gmsh.model.occ.addBox(0.0, 0.0, 0.0, float(length), float(height), float(width))
    gmsh.model.occ.synchronize()
    gmsh.write(str(out))
    gmsh.clear()
    return out


@_locked
def generate_ibeam_step(
    output_path: Union[str, Path],
    length: float = 1.0,
    height: float = 0.20,
    flange_width: float = 0.12,
    flange_thickness: float = 0.02,
    web_thickness: float = 0.015,
) -> Path:
    """Generate a structural I-beam / H-beam with flanged cross section."""
    _ensure_gmsh()
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    gmsh.clear()
    gmsh.model.add(f"ibeam_{out.stem}")

    L = float(length)
    H = float(height)
    B = float(flange_width)
    tf = float(flange_thickness)
    tw = float(web_thickness)

    # 12-vertex I-profile in the YZ plane centered at origin
    pts_yz = [
        (-B / 2.0, -H / 2.0),
        (B / 2.0, -H / 2.0),
        (B / 2.0, -H / 2.0 + tf),
        (tw / 2.0, -H / 2.0 + tf),
        (tw / 2.0, H / 2.0 - tf),
        (B / 2.0, H / 2.0 - tf),
        (B / 2.0, H / 2.0),
        (-B / 2.0, H / 2.0),
        (-B / 2.0, H / 2.0 - tf),
        (-tw / 2.0, H / 2.0 - tf),
        (-tw / 2.0, -H / 2.0 + tf),
        (-B / 2.0, -H / 2.0 + tf),
    ]

    p_tags = [gmsh.model.occ.addPoint(0.0, y, z) for y, z in pts_yz]
    lines = [gmsh.model.occ.addLine(p_tags[i], p_tags[(i + 1) % len(p_tags)]) for i in range(len(p_tags))]
    wire = gmsh.model.occ.addCurveLoop(lines)
    face = gmsh.model.occ.addPlaneSurface([wire])
    gmsh.model.occ.extrude([(2, face)], L, 0.0, 0.0)

    gmsh.model.occ.synchronize()
    gmsh.write(str(out))
    gmsh.clear()
    return out


@_locked
def generate_shaft_step(
    output_path: Union[str, Path],
    length: float = 0.60,
    diameter_main: float = 0.08,
    diameter_step: Optional[float] = 0.05,
    step_length: Optional[float] = 0.20,
    keyway: bool = True,
    keyway_w: Optional[float] = None,
    keyway_d: Optional[float] = None,
    keyway_l: Optional[float] = None,
) -> Path:
    """Generate a stepped cylindrical shaft with optional keyway slot."""
    _ensure_gmsh()
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    gmsh.clear()
    gmsh.model.add(f"shaft_{out.stem}")

    L = float(length)
    D1 = float(diameter_main)
    r1 = D1 / 2.0

    if diameter_step and step_length and step_length > 1e-4 and step_length < L:
        L_step = float(step_length)
        L_main = L - L_step
        D2 = float(diameter_step)
        r2 = D2 / 2.0
        s1 = gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, L_main, 0.0, 0.0, r1)
        s2 = gmsh.model.occ.addCylinder(L_main, 0.0, 0.0, L_step, 0.0, 0.0, r2)
        fused, _ = gmsh.model.occ.fuse([(3, s1)], [(3, s2)])
    else:
        s1 = gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, L, 0.0, 0.0, r1)
        fused = [(3, s1)]

    if keyway:
        kw = float(keyway_w) if keyway_w is not None else D1 * 0.25
        kd = float(keyway_d) if keyway_d is not None else D1 * 0.15
        kl = float(keyway_l) if keyway_l is not None else L * 0.25
        kx = L * 0.15
        box = gmsh.model.occ.addBox(kx, -kw / 2.0, r1 - kd, kl, kw, kd + 0.01)
        gmsh.model.occ.cut(fused, [(3, box)])

    gmsh.model.occ.synchronize()
    gmsh.write(str(out))
    gmsh.clear()
    return out


@_locked
def generate_pipe_step(
    output_path: Union[str, Path],
    length: float = 0.80,
    outer_diameter: float = 0.10,
    inner_diameter: float = 0.07,
) -> Path:
    """Generate a hollow circular tube / pipe solid."""
    _ensure_gmsh()
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    gmsh.clear()
    gmsh.model.add(f"pipe_{out.stem}")

    L = float(length)
    Do = float(outer_diameter)
    Di = float(inner_diameter)
    ro = Do / 2.0
    ri = Di / 2.0

    outer = gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, L, 0.0, 0.0, ro)
    inner = gmsh.model.occ.addCylinder(-0.01, 0.0, 0.0, L + 0.02, 0.0, 0.0, ri)
    gmsh.model.occ.cut([(3, outer)], [(3, inner)])

    gmsh.model.occ.synchronize()
    gmsh.write(str(out))
    gmsh.clear()
    return out


def generate_cad_from_parameters(
    object_type: str,
    params: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Dispatch to the appropriate parametric CAD generator based on detected object type."""
    ot = str(object_type).strip().lower()

    if ot in ("bracket", "l_bracket", "angle_bracket"):
        return generate_bracket_step(
            output_path,
            leg_x=params.get("leg_x", 0.20),
            leg_y=params.get("leg_y", 0.20),
            width=params.get("width", 0.06),
            thickness=params.get("thickness", 0.015),
            hole_radius=params.get("hole_radius", 0.008),
            hole_x=params.get("hole_x"),
            hole_y=params.get("hole_y"),
        )
    elif ot in ("plate_hole", "plate", "tensile_specimen"):
        return generate_plate_hole_step(
            output_path,
            length=params.get("length", 0.40),
            width=params.get("width", 0.20),
            thickness=params.get("thickness", 0.02),
            hole_diameter=params.get("hole_diameter", 0.06),
            hole_center_x=params.get("hole_center_x"),
            hole_center_y=params.get("hole_center_y"),
        )
    elif ot in ("beam", "bar", "box_beam"):
        return generate_beam_step(
            output_path,
            length=params.get("length", 1.0),
            height=params.get("height", 0.15),
            width=params.get("width", 0.10),
        )
    elif ot in ("ibeam", "i_beam", "h_beam"):
        return generate_ibeam_step(
            output_path,
            length=params.get("length", 1.0),
            height=params.get("height", 0.20),
            flange_width=params.get("flange_width", 0.12),
            flange_thickness=params.get("flange_thickness", 0.02),
            web_thickness=params.get("web_thickness", 0.015),
        )
    elif ot in ("shaft", "cylinder", "axle", "stepped_shaft"):
        return generate_shaft_step(
            output_path,
            length=params.get("length", 0.60),
            diameter_main=params.get("diameter_main", 0.08),
            diameter_step=params.get("diameter_step", 0.05),
            step_length=params.get("step_length", 0.20),
            keyway=params.get("keyway", True),
            keyway_w=params.get("keyway_w"),
            keyway_d=params.get("keyway_d"),
            keyway_l=params.get("keyway_l"),
        )
    elif ot in ("pipe", "tube", "hollow_cylinder"):
        return generate_pipe_step(
            output_path,
            length=params.get("length", 0.80),
            outer_diameter=params.get("outer_diameter", 0.10),
            inner_diameter=params.get("inner_diameter", 0.07),
        )
    else:
        # Fallback to beam with available dimensions
        return generate_beam_step(
            output_path,
            length=params.get("length", 0.50),
            height=params.get("height", 0.10),
            width=params.get("width", 0.10),
        )
