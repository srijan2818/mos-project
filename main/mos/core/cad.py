"""CAD STEP file importer and meshing interface using Gmsh (OpenCASCADE).

Provides robust conversion of .step / .stp files into scikit-fem MeshTet objects,
preserving CAD boundary surface entities as named boundary facet sets.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .regions import boundary_normal

from .gmsh_session import GMSH_ERROR, GMSH_LOCK, HAS_GMSH, ensure_gmsh, gmsh

from typing import TYPE_CHECKING

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from skfem import MeshTet


class CadImportError(RuntimeError):
    """Raised when CAD STEP importing or meshing encounters a problem."""


def gmsh_to_meshtet(surface_groups: Dict[str, list]) -> "MeshTet":
    """Convert the current Gmsh model's linear tetrahedra to a scikit-fem ``MeshTet``.

    ``surface_groups`` maps a boundary name to the Gmsh surface tags it contains; each
    becomes a named facet set in ``mesh.boundaries``. Nodes not used by any tetrahedron
    (CAD vertices, curve-only nodes) are dropped, since they would add zero-stiffness
    DOFs to the FE system. Everything is vectorised: no per-node or per-facet Python loops.
    Call with :data:`GMSH_LOCK` held after ``gmsh.model.mesh.generate(3)``.
    """
    from skfem import MeshTet

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    if len(node_tags) == 0:
        raise CadImportError("Gmsh failed to generate nodes for the model.")
    node_tags = np.asarray(node_tags, dtype=np.int64)
    coords = np.asarray(node_coords, dtype=float).reshape(-1, 3)
    order = np.argsort(node_tags)
    sorted_tags = node_tags[order]

    def to_index(tags):
        """Gmsh node tags -> row indices into ``coords``."""
        pos = np.searchsorted(sorted_tags, tags)
        return order[pos]

    elem_types, _, elem_nodes = gmsh.model.mesh.getElements(3)
    tet_blocks = [np.asarray(n, dtype=np.int64) for t, n in zip(elem_types, elem_nodes) if t == 4]
    if not tet_blocks:
        raise CadImportError("Gmsh did not generate linear tetrahedra (set Mesh.ElementOrder = 1).")
    tets = to_index(np.concatenate(tet_blocks)).reshape(-1, 4)

    used, compact = np.unique(tets, return_inverse=True)
    t = np.ascontiguousarray(compact.reshape(-1, 4).T)
    p = np.ascontiguousarray(coords[used].T)
    remap = np.full(len(coords), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))

    temp = MeshTet(p, t)
    key_dtype = np.dtype([("a", np.int64), ("b", np.int64), ("c", np.int64)])

    def keys(tri):
        """Order-independent structured keys for triangles given as (3, n) vertex indices."""
        return np.ascontiguousarray(np.sort(tri, axis=0).T.astype(np.int64)).view(key_dtype).ravel()

    facet_keys = keys(np.asarray(temp.facets))
    facet_order = np.argsort(facet_keys)
    sorted_facets = facet_keys[facet_order]

    outer_facets = np.asarray(temp.boundary_facets())
    boundaries: Dict[str, np.ndarray] = {}
    for name, tags in surface_groups.items():
        found = []
        for tag in tags:
            types2, _, nodes2 = gmsh.model.mesh.getElements(2, int(tag))
            for etype, nodes in zip(types2, nodes2):
                if etype != 2 or len(nodes) == 0:      # 3-node triangles only
                    continue
                tri = remap[to_index(np.asarray(nodes, dtype=np.int64))].reshape(-1, 3).T
                tri = tri[:, (tri >= 0).all(axis=0)]
                k = keys(tri)
                pos = np.clip(np.searchsorted(sorted_facets, k), 0, len(sorted_facets) - 1)
                hit = sorted_facets[pos] == k
                found.append(facet_order[pos[hit]])
        if found:
            # Surfaces shared by touching solids are interior after fragmenting: they
            # carry no boundary facets and must not be offered as load/support faces.
            ids = np.intersect1d(np.concatenate(found), outer_facets)
            if ids.size:
                boundaries[name] = ids
    return MeshTet(p, t, _boundaries=boundaries)



def is_step_file(path: str | Path) -> bool:
    """Check if the given path has a STEP file extension (.step or .stp)."""
    return str(path).lower().endswith((".step", ".stp"))


def load_step_mesh(
    step_path: str | Path,
    mesh_size: Optional[float] = None,
    order: int = 1,
) -> Tuple[MeshTet, Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Import a STEP CAD file and generate a 3D tetrahedral mesh.

    Parameters
    ----------
    step_path : str or Path
        Path to the .step or .stp file.
    mesh_size : float, optional
        Target characteristic element size. If None, automatically estimated from
        the model bounding box (~8% of the bounding box diagonal).
    order : int, default=1
        Element order (1 for linear 4-node tets, which are converted to quadratic TetP2
        during FEM basis assembly).

    Returns
    -------
    mesh : MeshTet
        The scikit-fem tetrahedral mesh with named boundaries for each CAD face.
    face_info : dict
        Mapping of boundary name (e.g. 'cad_face_1') to face metadata:
        {tag, name, area, centroid, bbox, normal, facets, friendly_name}.
    stats : dict
        Model geometry and mesh statistics:
        {bounds, dimensions, diagonal, volume, n_vertices, n_elements, n_faces}.
    """
    if not HAS_GMSH:
        raise CadImportError(
            "Gmsh is required for importing STEP CAD files but could not be loaded"
            + (f" ({GMSH_ERROR})" if GMSH_ERROR else "")
            + ". Install it with: conda install -c conda-forge python-gmsh"
        )

    try:
        from skfem import MeshTet
    except ImportError as exc:
        raise CadImportError("scikit-fem is required to convert a Gmsh mesh to MeshTet") from exc

    path_obj = Path(step_path).resolve()
    if not path_obj.is_file():
        raise FileNotFoundError(f"CAD STEP file not found: {step_path}")

    GMSH_LOCK.acquire()
    try:
        ensure_gmsh()
        gmsh.clear()
        # gmsh.clear() removes every model; the OpenCASCADE importer requires a
        # current model, so create one before importing the STEP geometry.
        gmsh.model.add(f"cad_import_{path_obj.stem}")

        # Import STEP shape via OpenCASCADE kernel
        gmsh.model.occ.importShapes(str(path_obj))
        gmsh.model.occ.synchronize()
        if len(gmsh.model.getEntities(3)) > 1:
            # Several touching solids must share interface nodes, otherwise the FE
            # model is a set of disconnected bodies. Fragment them into one conformal shape.
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

        surfs = gmsh.model.getEntities(2)
        vols = gmsh.model.getEntities(3)

        if not vols:
            raise CadImportError(
                f"No 3D solid volumes found in CAD file '{path_obj.name}'. "
                "Ensure the STEP file contains a closed 3D solid body."
            )

        # Compute overall bounding box across all volumes
        vol_bboxes = [gmsh.model.getBoundingBox(3, v[1]) for v in vols]
        xmin = min(b[0] for b in vol_bboxes)
        ymin = min(b[1] for b in vol_bboxes)
        zmin = min(b[2] for b in vol_bboxes)
        xmax = max(b[3] for b in vol_bboxes)
        ymax = max(b[4] for b in vol_bboxes)
        zmax = max(b[5] for b in vol_bboxes)

        dx, dy, dz = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9), max(zmax - zmin, 1e-9)
        diag = float(np.linalg.norm([dx, dy, dz]))
        total_volume = sum(gmsh.model.occ.getMass(3, v[1]) for v in vols)

        # Determine target mesh size
        char_len = float(mesh_size) if mesh_size is not None and mesh_size > 0 else max(diag * 0.08, 1e-4)

        # Register physical groups for 2D surfaces to retain boundary identifiers
        face_info: Dict[str, Dict[str, Any]] = {}
        cx, cy, cz = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)

        for dim, tag in surfs:
            name = f"cad_face_{tag}"
            gmsh.model.addPhysicalGroup(2, [tag], tag, name=name)

            area = float(gmsh.model.occ.getMass(2, tag))
            com = np.array(gmsh.model.occ.getCenterOfMass(2, tag), dtype=float)
            bbox = list(gmsh.model.getBoundingBox(2, tag))

            # Estimate normal vector from center of mass relative to overall centroid or surface derivative
            # We can compute normal at center of mass
            rel = com - np.array([cx, cy, cz])
            rel_norm = np.linalg.norm(rel)
            est_normal = rel / rel_norm if rel_norm > 1e-6 else np.array([1.0, 0.0, 0.0])

            # Classify face direction for a friendly name (e.g. Tip, Root, Top, Bottom, etc.)
            friendly_dir = ""
            if abs(com[0] - xmax) < 0.05 * dx:
                friendly_dir = "Tip (+X)"
            elif abs(com[0] - xmin) < 0.05 * dx:
                friendly_dir = "Root (-X)"
            elif abs(com[1] - ymax) < 0.05 * dy:
                friendly_dir = "Top (+Y)"
            elif abs(com[1] - ymin) < 0.05 * dy:
                friendly_dir = "Bottom (-Y)"
            elif abs(com[2] - zmax) < 0.05 * dz:
                friendly_dir = "Front (+Z)"
            elif abs(com[2] - zmin) < 0.05 * dz:
                friendly_dir = "Back (-Z)"

            label = f"Face {tag}"
            if friendly_dir:
                label += f" ({friendly_dir})"
            label += f" - Area: {area:.3g} m²"

            face_info[name] = {
                "tag": tag,
                "name": name,
                "friendly_name": label,
                "area": area,
                "centroid": com,
                "bbox": bbox,
                "normal": est_normal,
                "facets": np.array([], dtype=int),
            }

        # Add physical group for 3D volume
        gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], 1, name="volume")

        # Configure mesh parameters
        gmsh.option.setNumber("Mesh.MeshSizeMin", char_len * 0.6)
        gmsh.option.setNumber("Mesh.MeshSizeMax", char_len * 1.4)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)  # P2 is built by the FE basis, not Gmsh
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay

        # Generate 3D volume mesh
        gmsh.model.mesh.generate(3)

        mesh = gmsh_to_meshtet({f"cad_face_{tag}": [tag] for _, tag in surfs})
        for name, ids in mesh.boundaries.items():
            face_info[name]["facets"] = ids
            # Outward normal from the oriented mesh facets. The previous estimate (face
            # centroid minus body centroid) points inward on holes and re-entrant faces.
            face_info[name]["normal"] = boundary_normal(mesh, ids)

        stats = {
            "bounds": [xmin, xmax, ymin, ymax, zmin, zmax],
            "dimensions": [dx, dy, dz],
            "diagonal": diag,
            "volume": total_volume,
            "n_vertices": mesh.nvertices,
            "n_elements": mesh.nelements,
            "n_faces": len(surfs),
            "step_filename": path_obj.name,
        }

        return mesh, face_info, stats

    except CadImportError:
        raise
    except Exception as exc:
        raise CadImportError(
            f"Failed to import or mesh CAD file '{path_obj.name}': "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        try:
            gmsh.clear()
        except Exception as exc:
            log.warning("load_step_mesh: %s: %s", type(exc).__name__, exc)
        GMSH_LOCK.release()