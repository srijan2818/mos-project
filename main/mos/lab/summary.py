"""Read-only model and result summaries for the Learning Lab."""
from __future__ import annotations

from html import escape

import numpy as np


def model_summary(model, case, grid):
    """Return compact HTML readouts derived from the current solved model."""
    if model is None or grid is None:
        return "<p>No solved model is available. Solve a model in Analysis Workspace.</p>"
    material = getattr(model, "material", None)
    c11 = float(material.C[0, 0]) if material is not None else float("nan")
    rho = getattr(material, "rho", None)   # None = not given; shown as "—", never as 0
    loads = list(getattr(case, "loads", []) or [])
    supports = list(getattr(model, "bcs", []) or [])
    rows = [
        ("Mesh", f"{grid.n_points:,} nodes · {grid.n_cells:,} cells"),
        ("Material", f"C₁₁ = {c11 / 1e9:.4g} GPa · ρ = "
                     + (f"{float(rho):.4g} kg/m³" if rho is not None else "—")),
        ("Loads", f"{len(loads)} definitions"),
        ("Supports", f"{len(supports)} boundary-condition definitions"),
    ]
    summary = "".join(
        f'<div class="d-flex justify-space-between flex-wrap ga-2 py-1">'
        f'<span class="text-medium-emphasis">{escape(label)}</span>'
        f'<strong>{escape(value)}</strong></div>' for label, value in rows
    )
    if loads:
        items = []
        for load in loads:
            kind = type(load).__name__
            region = getattr(getattr(load, "region", None), "name", "whole body")
            vector = getattr(load, "force", getattr(load, "M", None))
            detail = f" · {np.array2string(np.asarray(vector), precision=3)}" if vector is not None else ""
            items.append(f"<li>{escape(kind)} · {escape(str(region))}{escape(detail)}</li>")
        summary += '<div class="text-caption font-weight-bold mt-2">Applied loads</div><ul>' + "".join(items) + "</ul>"
    if supports:
        items = []
        for bc in supports:
            region = getattr(getattr(bc, "region", None), "name", "region")
            components = "".join("xyz"[int(c)] for c in getattr(bc, "components", ()))
            items.append(f"<li>{escape(str(region))} · {escape(components or 'no components')}</li>")
        summary += '<div class="text-caption font-weight-bold mt-2">Supports</div><ul>' + "".join(items) + "</ul>"
    return summary


def tensor_matrix_html(tensor, scale=1.0, digits=4):
    """Render a compact escaped 3x3 matrix."""
    a = np.asarray(tensor, dtype=float) / float(scale)
    return ("<table class='mos-lab-matrix'><tbody>" +
            "".join("<tr>" + "".join(f"<td>{v:.{digits}g}</td>" for v in row) + "</tr>" for row in a) +
            "</tbody></table>")
