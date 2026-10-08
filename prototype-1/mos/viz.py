"""Results -> things to draw. The array helpers are pure numpy; to_pyvista needs pyvista + meshio."""
from __future__ import annotations

import numpy as np

from .core import post

STRESS_UNITS = {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9}


# ------------------------------------------------------------------ scalar helpers

def compute_scalars(res) -> dict:
    """name -> (nodal array (nnodes,), kind) with kind in {'disp', 'stress', 'strain'}.
    `res` needs u_nodal (3, n), sig_nodal (3, 3, n), eps_nodal (3, 3, n)."""
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

def face_definitions(bounds) -> dict:
    """Dictionary of standard bounding faces: key -> {name, normal, centroid, center, direction, i_size, j_size}."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    cx, cy, cz = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)
    return {
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


def create_face_mesh(face_key: str, bounds):
    """PyVista Plane geometry representing the specified bounding face."""
    import pyvista as pv
    plane_fn = getattr(pv, "Plane", None)
    if plane_fn is None:
        return None
    faces = face_definitions(bounds)
    if face_key not in faces:
        return None
    info = faces[face_key]
    return plane_fn(center=info["center"], direction=info["direction"],
                    i_size=info["i_size"], j_size=info["j_size"])


# ------------------------------------------------------------------ scalar bar theming

def scalar_bar_args(title: str, is_dark: bool = True) -> dict:
    """Return scalar_bar_args dict themed for dark or light mode."""
    fg = "#f4f4f5" if is_dark else "#18181b"   # zinc-100 / zinc-900
    return {
        "title": title,
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

def create_arrow_glyph(origin, vector, model_size: float):
    """PyVista Arrow mesh aligned with vector and scaled relative to model_size."""
    import pyvista as pv
    arrow_fn = getattr(pv, "Arrow", None)
    if arrow_fn is None:
        return None
    v = np.asarray(vector, float)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return None
    direction = v / norm
    length = max(0.18 * model_size, 1e-4)
    start = np.asarray(origin, float) - 0.25 * length * direction
    try:
        return arrow_fn(start=start, direction=direction, scale=length,
                        tip_length=0.28, tip_radius=0.10, shaft_radius=0.04)
    except Exception:
        return None


def create_moment_glyph(origin, M_vec, model_size: float):
    """PyVista Arrow mesh visualizing a moment about an axis at origin."""
    import pyvista as pv
    arrow_fn = getattr(pv, "Arrow", None)
    if arrow_fn is None:
        return None
    m = np.asarray(M_vec, float)
    norm = np.linalg.norm(m)
    if norm < 1e-12:
        return None
    direction = m / norm
    length = max(0.20 * model_size, 1e-4)
    start = np.asarray(origin, float) - 0.25 * length * direction
    try:
        return arrow_fn(start=start, direction=direction, scale=length,
                        tip_length=0.32, tip_radius=0.12, shaft_radius=0.05)
    except Exception:
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
    """Return list of (pyvista_mesh, color_hex) for all loads in a LoadCase.

    Draws:
      - PointForce  -> red arrow at node-region centroid
      - Traction    -> bright orange arrow at facet-region centroid
      - Pressure    -> amber inward arrow at facet centroid
      - Moment      -> purple moment arrow
      - BodyForce   -> cyan arrow at model centre
    """
    import pyvista as pv
    from .core.loads import BodyForce, Moment, PointForce, Pressure, Traction

    glyphs = []
    arrow_len = max(0.18 * model_size, 1e-4)
    model_center = mesh.p.mean(axis=1)

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
        except Exception:
            pass

    for load in case.loads:
        try:
            if isinstance(load, PointForce):
                origin = _region_centroid_direct(load.region, mesh)
                _add_arrow(origin, load.force, arrow_len, "#ef4444")

            elif isinstance(load, Traction):
                origin = _region_centroid_direct(load.region, mesh)
                f = load.field
                try:
                    val = np.array([float(f.components[i](*origin)) for i in range(3)], float)
                except Exception:
                    val = np.array([0, -1, 0], float)
                _add_arrow(origin, val, arrow_len, "#f97316")

            elif isinstance(load, Pressure):
                origin = _region_centroid_direct(load.region, mesh)
                _add_arrow(origin, -origin / (np.linalg.norm(origin) + 1e-12), arrow_len, "#f59e0b")

            elif isinstance(load, Moment):
                origin = _region_centroid_direct(load.region, mesh)
                _add_arrow(origin, load.M, arrow_len * 1.1, "#a855f7")

            elif isinstance(load, BodyForce):
                try:
                    val = np.array([float(load.field.components[i](*model_center)) for i in range(3)], float)
                except Exception:
                    val = np.array([0, -1, 0], float)
                _add_arrow(model_center, val, arrow_len * 1.2, "#06b6d4")

        except Exception:
            pass

    return glyphs
