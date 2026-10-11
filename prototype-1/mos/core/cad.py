"""CAD STEP file importer and meshing interface using Gmsh (OpenCASCADE).

Provides robust conversion of .step / .stp files into scikit-fem MeshTet objects,
preserving CAD boundary surface entities as named boundary facet sets.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gmsh
    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False

from skfem import MeshTet


class CadImportError(RuntimeError):
    """Raised when CAD STEP importing or meshing encounters a problem."""


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
            "Gmsh is required for importing STEP CAD files. "
            "Please install it with: pip install gmsh"
        )

    path_obj = Path(step_path).resolve()
    if not path_obj.is_file():
        raise FileNotFoundError(f"CAD STEP file not found: {step_path}")

    try:
        # Initialize gmsh safely
        if not gmsh.isInitialized():
            gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.clear()

        # Import STEP shape via OpenCASCADE kernel
        shapes = gmsh.model.occ.importShapes(str(path_obj))
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
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay

        # Generate 3D volume mesh
        gmsh.model.mesh.generate(3)

        # Extract nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        if len(node_tags) == 0:
            raise CadImportError("Gmsh failed to generate nodes for the CAD model.")

        node_idx_map = {tag: i for i, tag in enumerate(node_tags)}
        p = np.array(node_coords, dtype=float).reshape(-1, 3).T

        # Extract 3D tetrahedra
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3)
        if len(elem_node_tags) == 0:
            raise CadImportError("Gmsh failed to generate 3D tetrahedral elements.")

        # Map global 1-based node tags to 0-based index array
        tets_raw = np.array(elem_node_tags[0], dtype=int)
        tets = np.vectorize(node_idx_map.get)(tets_raw).reshape(-1, 4).T

        # Temporary mesh to obtain internal facet indices
        temp_mesh = MeshTet(p, tets)
        facet_triangles = np.sort(temp_mesh.facets, axis=0)
        tri_to_facet = {tuple(facet_triangles[:, i]): i for i in range(temp_mesh.nfacets)}

        # Extract 2D boundary facet sets for each CAD surface
        boundaries: Dict[str, np.ndarray] = {}
        for dim, tag in surfs:
            name = f"cad_face_{tag}"
            types_2d, tags_2d, node_tags_2d = gmsh.model.mesh.getElements(2, tag)
            if len(node_tags_2d) > 0:
                tris_nodes = np.vectorize(node_idx_map.get)(np.array(node_tags_2d[0], dtype=int)).reshape(-1, 3)
                facet_indices = []
                for tri in tris_nodes:
                    key = tuple(sorted(tri))
                    if key in tri_to_facet:
                        facet_indices.append(tri_to_facet[key])
                if facet_indices:
                    arr = np.array(facet_indices, dtype=int)
                    boundaries[name] = arr
                    face_info[name]["facets"] = arr

        mesh = MeshTet(p, tets, _boundaries=boundaries)

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

    finally:
        try:
            gmsh.clear()
        except Exception:
            pass
