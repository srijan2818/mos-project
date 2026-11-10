"""Results -> things to draw. The array helpers are pure numpy; to_pyvista needs pyvista + meshio."""
from __future__ import annotations

import logging

import numpy as np

from .core import post
from .core.regions import boundary_normal

log = logging.getLogger(__name__)

STRESS_UNITS = {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9}


# ------------------------------------------------------------------ scalar helpers

UTILISATION = "Utilisation (σ/σ_pl)"


def compute_scalars(res, material=None) -> dict:
    """name -> (nodal array (nnodes,), kind) with kind in {'disp', 'stress', 'strain', 'ratio'}.
    `res` needs u_nodal (3, n), sig_nodal (3, 3, n), eps_nodal (3, 3, n). With a ``material``
    that has a proportional limit, the dimensionless :data:`UTILISATION` field is added."""
    u, S, E = res.u_nodal, res.sig_nodal, res.eps_nodal
    s, _ = post.principal(S)
    e, _ = post.principal(E)
    out = {"|u|": (np.linalg.norm(u, axis=0), "disp"),
           "ux": (u[0], "disp"), "uy": (u[1], "disp"), "uz": (u[2], "disp"),
           "von Mises": (post.von_mises(S), "stress"),
           "σ1 (max principal)": (s[0], "stress"),
           "σ3 (min principal)": (s[2], "stress"),
           "Tresca": (s[0] - s[2], "stress")}
    for name, (i, j) in {"σxx": (0, 0), "σyy": (1, 1), "σzz": (2, 2),
                         "τxy": (0, 1), "τyz": (1, 2), "τxz": (0, 2)}.items():
        out[name] = (S[i, j], "stress")
    for name, (i, j) in {"εxx": (0, 0), "εyy": (1, 1), "εzz": (2, 2)}.items():
        out[name] = (E[i, j], "strain")
    for name, (i, j) in {"γxy": (0, 1), "γyz": (1, 2), "γxz": (0, 2)}.items():
        out[name] = (2 * E[i, j], "strain")          # engineering shear strain
    out["ε1 (max principal)"] = (e[0], "strain")
    if material is not None and material.sigma_pl is not None:
        out[UTILISATION] = (post.utilisation(S, material.sigma_pl, material.sigma_pl_c), "ratio")
    return out


def suggest_scale(points, u_nodal) -> float:
    """Deformation factor that makes the largest displacement ~10 % of the model size."""
    umax = float(np.linalg.norm(u_nodal, axis=0).max())
    diag = float(np.linalg.norm(np.ptp(np.asarray(points), axis=0)))
    return 0.1 * diag / umax if umax > 0 else 1.0


def clip_origin(bounds, axis: int, frac: float):
    """Point on the clip plane at fraction `frac` of the bounding box along `axis`."""
    lo, hi = bounds[2 * axis], bounds[2 * axis + 1]
    o = [0.5 * (bounds[0] + bounds[1]), 0.5 * (bounds[2] + bounds[3]), 0.5 * (bounds[4] + bounds[5])]
    o[axis] = lo + frac * (hi - lo)
    return o


# ------------------------------------------------------------------ mesh conversion

def to_pyvista(mesh):
    """scikit-fem mesh -> pyvista UnstructuredGrid (same node order as mesh.p)."""
    import pyvista as pv
    from skfem.io.meshio import to_meshio
    return pv.from_meshio(to_meshio(mesh))


# ------------------------------------------------------------------ face helpers

def face_definitions(bounds, mesh=None) -> dict:
    """Dictionary of standard bounding faces: key -> {name, normal, centroid, center, direction, i_size, j_size}."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    cx, cy, cz = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)
    defs = {
        "+x": {"name": "+X (Tip / Right)", "normal": np.array([1.0, 0.0, 0.0]),
               "centroid": np.array([xmax, cy, cz]), "center": (xmax, cy, cz),
               "direction": (1.0, 0.0, 0.0), "i_size": dy, "j_size": dz},
        "-x": {"name": "-X (Root / Left)", "normal": np.array([-1.0, 0.0, 0.0]),
               "centroid": np.array([xmin, cy, cz]), "center": (xmin, cy, cz),
               "direction": (-1.0, 0.0, 0.0), "i_size": dy, "j_size": dz},
        "+y": {"name": "+Y (Top)", "normal": np.array([0.0, 1.0, 0.0]),
               "centroid": np.array([cx, ymax, cz]), "center": (cx, ymax, cz),
               "direction": (0.0, 1.0, 0.0), "i_size": dx, "j_size": dz},
        "-y": {"name": "-Y (Bottom)", "normal": np.array([0.0, -1.0, 0.0]),
               "centroid": np.array([cx, ymin, cz]), "center": (cx, ymin, cz),
               "direction": (0.0, -1.0, 0.0), "i_size": dx, "j_size": dz},
        "+z": {"name": "+Z (Front)", "normal": np.array([0.0, 0.0, 1.0]),
               "centroid": np.array([cx, cy, zmax]), "center": (cx, cy, zmax),
               "direction": (0.0, 0.0, 1.0), "i_size": dx, "j_size": dy},
        "-z": {"name": "-Z (Back)", "normal": np.array([0.0, 0.0, -1.0]),
               "centroid": np.array([cx, cy, zmin]), "center": (cx, cy, zmin),
               "direction": (0.0, 0.0, -1.0), "i_size": dx, "j_size": dy},
    }
    if mesh is not None and hasattr(mesh, "boundaries") and mesh.boundaries:
        for b_name, fac_ids in mesh.boundaries.items():
            if len(fac_ids) == 0:
                continue
            pts = mesh.p[:, mesh.facets[:, fac_ids]].reshape(3, -1)
            c = pts.mean(axis=1)
            # Facet vertex order is not consistently oriented; use outward normals.
            n_vec = boundary_normal(mesh, fac_ids)

            tag_part = b_name.replace("cad_face_", "Face ")
            defs[b_name] = {
                "name": tag_part,
                "normal": n_vec,
                "centroid": c,
                "center": tuple(c),
                "direction": tuple(n_vec),
                "i_size": max(dx, dy, dz) * 0.2,
                "j_size": max(dx, dy, dz) * 0.2,
                "facet_ids": fac_ids,
            }
    return defs


def create_face_mesh(face_key: str, bounds, mesh=None):
    """PyVista geometry representing the specified bounding face."""
    import pyvista as pv
    if mesh is not None and hasattr(mesh, "boundaries") and mesh.boundaries and face_key in mesh.boundaries:
        fac_ids = mesh.boundaries[face_key]
        if len(fac_ids) > 0:
            triangles = mesh.facets[:, fac_ids].T
            faces_pv = np.hstack([np.full((len(fac_ids), 1), 3), triangles]).ravel()
            return pv.PolyData(mesh.p.T, faces_pv)

    plane_fn = getattr(pv, "Plane", None)
    if plane_fn is None:
        return None
    faces = face_definitions(bounds, mesh)
    if face_key not in faces:
        return None
    info = faces[face_key]
    return plane_fn(center=info["center"], direction=info["direction"],
                    i_size=info.get("i_size", 1.0), j_size=info.get("j_size", 1.0))


# ------------------------------------------------------------------ scalar bar theming

def scalar_bar_args(title: str, is_dark: bool = True) -> dict:
    """Return scalar_bar_args dict themed for dark or light mode."""
    fg = "#f4f4f5" if is_dark else "#18181b"   # zinc-100 / zinc-900
    return {
        # VTK draws the top tick label half a line above the bar, where it overlapped
        # the title; the trailing newline lifts the title one line clear of it.
        "title": title + "\n",
        "color": fg,
        "title_font_size": 14,
        "label_font_size": 11,
        "shadow": False,
        "n_labels": 5,
        "italic": False,
        "fmt": "%.3g",
        "font_family": "arial",
        "vertical": True,
        "position_x": 0.86,
        "position_y": 0.08,
        "width": 0.11,
        "height": 0.82,
    }


# ------------------------------------------------------------------ glyph helpers


def glyph_magnitude_scale(magnitude: float, reference_magnitude: float = 1.0) -> float:
    """Map a physical load magnitude to a bounded square-root glyph scale."""
    ratio = np.sqrt(abs(float(magnitude)) / max(float(reference_magnitude), 1e-30))
    return float(np.clip(ratio, 0.12, 2.2))

def create_arrow_glyph(origin, vector, model_size: float, reference_magnitude=None):
    """Create an arrow aligned with ``vector``.

    When ``reference_magnitude`` is supplied, arrow length changes with the square
    root of the physical magnitude. This preserves a useful visible range while making
    live load scaling apparent in the viewport. The reference uses the vector's units.
    """
    import pyvista as pv
    arrow_fn = getattr(pv, "Arrow", None)
    if arrow_fn is None:
        return None
    v = np.asarray(vector, float)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return None
    direction = v / norm
    factor = 1.0 if reference_magnitude is None else glyph_magnitude_scale(norm, reference_magnitude)
    length = max(0.18 * model_size * factor, 1e-4)
    start = np.asarray(origin, float) - 0.25 * length * direction
    try:
        return arrow_fn(start=start, direction=direction, scale=length,
                        tip_length=0.28, tip_radius=0.10, shaft_radius=0.04)
    except Exception as exc:
        log.warning("create_arrow_glyph: %s: %s", type(exc).__name__, exc)
        return None


def create_moment_glyph(origin, M_vec, model_size: float, reference_magnitude=None):
    """Visualize a moment with a curved arc whose radius can reflect its magnitude."""
    import pyvista as pv
    m = np.asarray(M_vec, float)
    norm = np.linalg.norm(m)
    if norm < 1e-12:
        return None
    axis = m / norm
    factor = 1.0 if reference_magnitude is None else glyph_magnitude_scale(norm, reference_magnitude)
    length = max(0.20 * model_size * factor, 1e-4)
    radius = 0.5 * length

    # Find two orthonormal vectors perpendicular to the moment axis
    if abs(axis[0]) < 0.8:
        v1 = np.cross(axis, [1, 0, 0])
    else:
        v1 = np.cross(axis, [0, 1, 0])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(axis, v1)
    v2 = v2 / np.linalg.norm(v2)

    try:
        # 270 degree circular arc indicating rotation direction
        theta = np.linspace(0, 1.4 * np.pi, 28)
        o = np.asarray(origin, float)
        arc_pts = np.array([
            o + radius * (np.cos(t) * v1 + np.sin(t) * v2) for t in theta
        ])
        tube_r = max(0.015 * length, 1e-5)
        arc_spline = pv.Spline(arc_pts, 80).tube(radius=tube_r)

        # Arrowhead at arc endpoint pointing along tangent
        last_t = theta[-1]
        tangent = -np.sin(last_t) * v1 + np.cos(last_t) * v2
        arrow_head = pv.Arrow(
            start=arc_pts[-1] - 0.25 * (0.35 * length) * tangent,
            direction=tangent,
            scale=0.35 * length,
            tip_length=0.35,
            tip_radius=0.12,
            shaft_radius=0.04,
        )

        # Also add central axis arrow
        axis_arr = pv.Arrow(
            start=o - 0.15 * length * axis,
            direction=axis,
            scale=0.8 * length,
            tip_length=0.25,
            tip_radius=0.08,
            shaft_radius=0.03,
        )
        return arc_spline + arrow_head + axis_arr
    except Exception:
        # Fallback to single straight arrow
        try:
            arrow_fn = getattr(pv, "Arrow", None)
            start = np.asarray(origin, float) - 0.25 * length * axis
            return arrow_fn(start=start, direction=axis, scale=length,
                            tip_length=0.32, tip_radius=0.12, shaft_radius=0.05)
        except Exception as exc:
            log.warning("create_moment_glyph: %s: %s", type(exc).__name__, exc)
            return None


def _region_centroid_direct(region, mesh) -> np.ndarray:
    """Centroid of a region from its ids (nodes or facets)."""
    if region.kind == "nodes":
        pts = mesh.p[:, region.ids]
        return pts.mean(axis=1)
    else:
        # facet columns: mesh.facets is shape (verts_per_facet, n_facets)
        fac_ids = region.ids
        pts = mesh.p[:, mesh.facets[:, fac_ids]].reshape(3, -1)
        return pts.mean(axis=1)


def build_load_glyphs(case, mesh, model_size: float):
    """Return (glyph, color) pairs for loads in a case.

    Arrow direction indicates the load direction; arrow length responds to magnitude,
    so changing a load in the live inspector also produces immediate viewport feedback.
    Reference magnitudes are type-specific because N, N·m and Pa are not comparable.
    """
    import pyvista as pv
    from .core.loads import BodyForce, Moment, PointForce, Pressure, SurfaceForce, Traction

    glyphs = []
    arrow_len = max(0.18 * model_size, 1e-4)
    model_center = np.asarray(mesh.p, float).mean(axis=1)

    def _scaled_length(magnitude, reference, base=arrow_len):
        return base * glyph_magnitude_scale(magnitude, reference)

    def _add_arrow(origin, direction, length, color):
        v = np.asarray(direction, float)
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return
        d = v / norm
        start = np.asarray(origin, float) - 0.25 * length * d
        try:
            arr = pv.Arrow(start=start, direction=d, scale=length,
                           tip_length=0.28, tip_radius=0.10, shaft_radius=0.04)
            glyphs.append((arr, color))
        except Exception as exc:
            log.warning("_add_arrow: %s: %s", type(exc).__name__, exc)

    def _region_normal(region):
        """Area-weighted outward normal for a selected boundary-facet region."""
        if len(region.ids) == 0:
            return np.array([0.0, 0.0, 1.0])
        try:
            # Orient by the owning element, not the model centre: a hole or re-entrant
            # face would otherwise get an inward normal.
            return boundary_normal(mesh, region.ids)
        except (AttributeError, IndexError, ValueError):
            facets = np.asarray(mesh.facets[:, region.ids], dtype=int)
            xyz = np.asarray(mesh.p[:, facets], dtype=float)
            area = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0], axis=0)
            area[:, np.einsum("ij,ij->j", area, xyz.mean(axis=1) - model_center[:, None]) < 0] *= -1.0
            normal = area.sum(axis=1)
            magnitude = np.linalg.norm(normal)
            return normal / magnitude if magnitude > 1e-12 else np.array([0.0, 0.0, 1.0])

    def _field_value(field, point, vector=False):
        values = np.asarray(field(np.asarray(point, float).reshape(3, 1)), dtype=float)
        return values[:, 0] if vector else float(values.reshape(-1)[0])

    for load in case.loads:
        try:
            if isinstance(load, PointForce):
                origin = _region_centroid_direct(load.region, mesh)
                length = _scaled_length(np.linalg.norm(load.force), 1e5)
                _add_arrow(origin, load.force, length, "#ef4444")

            elif isinstance(load, SurfaceForce):
                origin = _region_centroid_direct(load.region, mesh)
                length = _scaled_length(np.linalg.norm(load.force), 1e5)
                _add_arrow(origin, load.force, length, "#ef4444")

            elif isinstance(load, Traction):
                origin = _region_centroid_direct(load.region, mesh)
                value = _field_value(load.field, origin, vector=True)
                length = _scaled_length(np.linalg.norm(value), 1e6)
                _add_arrow(origin, value, length, "#f97316")

            elif isinstance(load, Pressure):
                origin = _region_centroid_direct(load.region, mesh)
                pressure_value = _field_value(load.p, origin, vector=False)
                normal = _region_normal(load.region)
                length = _scaled_length(abs(pressure_value), 1e6)
                if abs(pressure_value) > 1e-14:
                    _add_arrow(origin, -np.sign(pressure_value) * normal, length, "#f59e0b")

            elif isinstance(load, Moment):
                origin = (np.asarray(load.origin, float) if load.origin is not None
                          else _region_centroid_direct(load.region, mesh))
                mg = create_moment_glyph(origin, load.M, model_size, reference_magnitude=1e5)
                if mg is not None:
                    glyphs.append((mg, "#a855f7"))
                else:
                    _add_arrow(origin, load.M, _scaled_length(np.linalg.norm(load.M), 1e5), "#a855f7")

            elif isinstance(load, BodyForce):
                value = _field_value(load.field, model_center, vector=True)
                length = _scaled_length(np.linalg.norm(value), 1e5, base=arrow_len * 1.2)
                _add_arrow(model_center, value, length, "#06b6d4")

        except Exception as exc:
            log.warning("_field_value: %s: %s", type(exc).__name__, exc)

    return glyphs
