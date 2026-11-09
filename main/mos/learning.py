"""Small, transparent mechanics-of-materials teaching visualisations.

All stresses passed to this module are in MPa.  This module is deliberately
separate from the FEM solver: it powers idealised point-stress experiments in
the Learning Lab and does not claim to be an FEA result.
"""
from __future__ import annotations

from html import escape
import math
from typing import Any

import numpy as np


def _fmt(value: float, digits: int = 2) -> str:
    """Format a number for compact UI display, normalising tiny zeroes."""
    value = 0.0 if abs(float(value)) < 0.5 * 10 ** (-digits) else float(value)
    return f"{value:.{digits}f}"


def analyse_stress_tensor(
    sigma_xx: float,
    sigma_yy: float,
    sigma_zz: float,
    tau_xy: float,
    tau_yz: float,
    tau_zx: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> dict[str, Any]:
    """Return principal data and traction decomposition for a symmetric stress tensor."""
    stress = np.array([
        [sigma_xx, tau_xy, tau_zx],
        [tau_xy, sigma_yy, tau_yz],
        [tau_zx, tau_yz, sigma_zz],
    ], dtype=float)

    principal, directions = np.linalg.eigh(stress)
    order = np.argsort(principal)[::-1]
    principal = principal[order]
    directions = directions[:, order]

    theta = math.radians(float(azimuth_deg))
    phi = math.radians(float(elevation_deg))
    normal = np.array([
        math.cos(phi) * math.cos(theta),
        math.cos(phi) * math.sin(theta),
        math.sin(phi),
    ])
    traction = stress @ normal
    normal_stress = float(normal @ traction)
    shear_vector = traction - normal_stress * normal
    shear_stress = float(np.linalg.norm(shear_vector))

    vm_sq = 0.5 * (
        (sigma_xx - sigma_yy) ** 2
        + (sigma_yy - sigma_zz) ** 2
        + (sigma_zz - sigma_xx) ** 2
    ) + 3.0 * (tau_xy ** 2 + tau_yz ** 2 + tau_zx ** 2)
    von_mises = math.sqrt(max(0.0, vm_sq))
    tresca = float(principal[0] - principal[2])

    directions_out = []
    for index, stress_value in enumerate(principal):
        vector = directions[:, index].copy()
        # Choose a stable sign for a readable direction-cosine display.
        dominant = int(np.argmax(np.abs(vector)))
        if vector[dominant] < 0:
            vector *= -1
        azimuth = math.degrees(math.atan2(vector[1], vector[0]))
        elevation = math.degrees(math.asin(float(np.clip(vector[2], -1.0, 1.0))))
        directions_out.append({
            "label": f"σ{index + 1}",
            "stress": float(stress_value),
            "vector": vector,
            "azimuth": azimuth,
            "elevation": elevation,
        })

    return {
        "matrix": stress,
        "principal": principal,
        "directions": directions_out,
        "normal": normal,
        "traction": traction,
        "normal_stress": normal_stress,
        "shear_vector": shear_vector,
        "shear_stress": shear_stress,
        "von_mises": von_mises,
        "tresca": tresca,
        "max_shear": tresca / 2.0,
        "mean_stress": float(np.trace(stress) / 3.0),
    }


def _project(point: np.ndarray, cx: float = 310, cy: float = 180) -> tuple[float, float]:
    """Oblique projection for a compact pseudo-3D cube illustration."""
    x, y, z = (float(v) for v in point)
    return cx + 76.0 * (x - y), cy + 40.0 * (x + y) - 78.0 * z


def _plane_polygon(normal: np.ndarray) -> list[np.ndarray]:
    """Find the intersection polygon of n·x=0 with the unit cube."""
    corners = [np.array([x, y, z], dtype=float)
               for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    edges = []
    for i, a in enumerate(corners):
        for j, b in enumerate(corners):
            if j <= i or np.count_nonzero(a != b) != 1:
                continue
            edges.append((a, b))

    points: list[np.ndarray] = []
    for a, b in edges:
        fa, fb = float(normal @ a), float(normal @ b)
        if abs(fa) < 1e-9:
            points.append(a)
        if fa * fb < -1e-12:
            points.append(a + (fa / (fa - fb)) * (b - a))
        if abs(fb) < 1e-9:
            points.append(b)

    unique: list[np.ndarray] = []
    for point in points:
        if not any(np.linalg.norm(point - item) < 1e-7 for item in unique):
            unique.append(point)
    if len(unique) < 3:
        return []

    center = np.mean(unique, axis=0)
    n = normal / max(float(np.linalg.norm(normal)), 1e-12)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(n, u)
    unique.sort(key=lambda point: math.atan2(float((point - center) @ v), float((point - center) @ u)))
    return unique


def tensor_diagram_svg(result: dict[str, Any]) -> str:
    """Build an SVG of the cube, oriented plane, unit normal, and traction vector."""
    corners = [np.array([x, y, z], dtype=float)
               for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    edge_lines = []
    for i, a in enumerate(corners):
        for j, b in enumerate(corners):
            if j <= i or np.count_nonzero(a != b) != 1:
                continue
            x1, y1 = _project(a)
            x2, y2 = _project(b)
            edge_lines.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" />'
            )

    normal = result["normal"]
    polygon = _plane_polygon(normal)
    if polygon:
        coords = " ".join(f"{_project(point)[0]:.1f},{_project(point)[1]:.1f}" for point in polygon)
        plane = f'<polygon points="{coords}" fill="#8b5cf6" fill-opacity=".24" stroke="#a78bfa" stroke-width="2.2" />'
    else:
        plane = ""

    principal_axis_svg = []
    for item, color in zip(
        result["directions"], ("#fb7185", "#34d399", "#60a5fa")
    ):
        direction = item["vector"]
        axis_end = _project(direction * 0.88)
        principal_axis_svg.append(
            f'<line x1="310" y1="180" x2="{axis_end[0]:.1f}" y2="{axis_end[1]:.1f}" '
            f'stroke="{color}" stroke-width="2.1" stroke-dasharray="5 4" marker-end="url(#arr)" />'
            f'<text x="{axis_end[0] + 7:.1f}" y="{axis_end[1] - 6:.1f}" '
            f'fill="{color}" font-size="12" font-weight="700">{item["label"]}</text>'
        )

    traction = result["traction"]
    max_component = max(float(np.linalg.norm(traction)), 1.0)
    normal_end = _project(normal * 0.95)
    traction_dir = traction / max_component
    traction_end = _project(traction_dir * 1.0, cx=330, cy=178)
    normal_start = _project(np.zeros(3))
    traction_start = (330.0, 178.0)

    # Small coordinate triad at the lower-left of the figure.
    origin = (92, 285)
    triad = [
        ("x", (origin[0] + 42, origin[1] + 14), "#fb7185"),
        ("y", (origin[0] - 32, origin[1] + 14), "#34d399"),
        ("z", (origin[0], origin[1] - 47), "#60a5fa"),
    ]
    triad_svg = "".join(
        f'<line x1="{origin[0]}" y1="{origin[1]}" x2="{p[0]}" y2="{p[1]}" stroke="{c}" stroke-width="2.5" marker-end="url(#arr)" />'
        f'<text x="{p[0] + 7}" y="{p[1] + 4}" fill="{c}" font-size="12">{label}</text>'
        for label, p, c in triad
    )

    return f'''<svg viewBox="0 0 620 360" role="img" aria-label="Stress element with an oriented plane, plane normal and traction vector" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="context-stroke" /></marker>
      </defs>
      <rect x="0" y="0" width="620" height="360" rx="18" fill="#0b1220" />
      <g stroke="#64748b" stroke-opacity=".78" stroke-width="1.8" fill="none">{''.join(edge_lines)}</g>
      {plane}
      {''.join(principal_axis_svg)}
      <line x1="{normal_start[0]:.1f}" y1="{normal_start[1]:.1f}" x2="{normal_end[0]:.1f}" y2="{normal_end[1]:.1f}" stroke="#22d3ee" stroke-width="3.2" marker-end="url(#arr)" />
      <text x="{normal_end[0] + 10:.1f}" y="{normal_end[1] - 8:.1f}" fill="#22d3ee" font-size="14" font-weight="700">n</text>
      <line x1="{traction_start[0]:.1f}" y1="{traction_start[1]:.1f}" x2="{traction_end[0]:.1f}" y2="{traction_end[1]:.1f}" stroke="#fbbf24" stroke-width="3.2" marker-end="url(#arr)" />
      <text x="{traction_end[0] + 10:.1f}" y="{traction_end[1] + 16:.1f}" fill="#fbbf24" font-size="14" font-weight="700">t = σ · n</text>
      {triad_svg}
      <text x="340" y="316" fill="#cbd5e1" font-size="12">Dashed: principal plane normals σ₁, σ₂, σ₃</text>
      <text x="340" y="334" fill="#94a3b8" font-size="11">Plane orientation changes with θ and φ · not an FEA contour</text>
    </svg>'''


def mohr_diagram_svg(principal: np.ndarray) -> str:
    """Draw the three 3D Mohr circles from the ordered principal stresses.

    The SVG uses equal x/y scales, so the circles remain circles rather than
    stretching into ellipses when the stress range changes.
    """
    s1, s2, s3 = (float(x) for x in principal)
    pairs = [
        (s1, s2, "σ₁–σ₂", "#22d3ee"),
        (s2, s3, "σ₂–σ₃", "#a78bfa"),
        (s1, s3, "σ₁–σ₃", "#fbbf24"),
    ]
    center_sigma = (s1 + s3) / 2.0
    max_radius = max((s1 - s3) / 2.0, 0.0)
    y_max = max(max_radius * 1.18, 5.0)

    # A square plotting region and one shared scale preserve circular geometry.
    cx, cy, plot_radius = 360.0, 184.0, 126.0
    scale = plot_radius / y_max
    sx = lambda value: cx + (float(value) - center_sigma) * scale

    circles = []
    for a, b, label, color in pairs:
        center, radius = (a + b) / 2.0, abs(b - a) / 2.0
        circles.append(
            f'<circle cx="{sx(center):.2f}" cy="{cy:.2f}" r="{radius * scale:.2f}" '
            f'fill="{color}" fill-opacity=".055" stroke="{color}" stroke-width="2.2" />'
        )

    points = "".join(
        f'<circle cx="{sx(value):.2f}" cy="{cy:.2f}" r="4.2" fill="{color}" />'
        for value, color in [(s1, "#22d3ee"), (s2, "#a78bfa"), (s3, "#fbbf24")]
    )
    # Labels are ~70 px wide at font-size 11: a label closer than that to the previous one
    # (in x order) drops to the next row instead of printing on top of it.
    placed, rows_end = [], []
    for value, label, color in sorted([(s1, "σ₁", "#22d3ee"), (s2, "σ₂", "#a78bfa"),
                                       (s3, "σ₃", "#fbbf24")], key=lambda t: t[0]):
        x = sx(value)
        row = next((r for r, end in enumerate(rows_end) if x - end >= 72.0), len(rows_end))
        rows_end[row:row + 1] = [x]
        placed.append((x, cy + 22 + 15 * row, value, label, color))
    labels = "".join(
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="middle" '
        f'fill="{color}" font-size="11">{label} {_fmt(value)}</text>'
        for x, y, value, label, color in placed
    )
    legend = "".join(
        f'<g transform="translate({70 + i * 180}, 18)"><line x1="0" y1="0" x2="22" y2="0" stroke="{color}" stroke-width="3"/><text x="30" y="4" fill="#cbd5e1" font-size="11">{escape(label)}</text></g>'
        for i, (_, _, label, color) in enumerate(pairs)
    )
    zero_axis = ""
    zero_x = sx(0.0)
    if cx - plot_radius - 8 <= zero_x <= cx + plot_radius + 8:
        zero_axis = f'<line x1="{zero_x:.2f}" y1="{cy - plot_radius - 8}" x2="{zero_x:.2f}" y2="{cy + plot_radius + 8}" stroke="#64748b" stroke-dasharray="4 5" stroke-opacity=".6" />'
    return f'''<svg viewBox="0 0 720 360" role="img" aria-label="Three geometrically scaled Mohr circles for the principal stresses" xmlns="http://www.w3.org/2000/svg">
      <rect width="720" height="360" rx="16" fill="#0b1220" />
      {legend}
      <line x1="{cx - plot_radius - 24}" y1="{cy:.2f}" x2="{cx + plot_radius + 24}" y2="{cy:.2f}" stroke="#94a3b8" stroke-opacity=".85" />
      {zero_axis}
      <text x="{cx - plot_radius - 22}" y="{cy - plot_radius - 10}" fill="#94a3b8" font-size="11">τ (MPa)</text>
      {''.join(circles)}{points}{labels}
      <text x="{cx + plot_radius + 24}" y="{cy + 4:.2f}" text-anchor="end" fill="#94a3b8" font-size="11">σ →</text>
      <text x="{cx}" y="342" text-anchor="middle" fill="#94a3b8" font-size="11">Normal stress σ (MPa) · equal axis scale</text>
    </svg>'''


def render_direction_rows(result: dict[str, Any]) -> str:
    """Render principal values and their direction cosines as a compact table."""
    rows = []
    for item in result["directions"]:
        x, y, z = item["vector"]
        rows.append(
            "<tr>"
            f"<td><b>{escape(item['label'])}</b></td>"
            f"<td>{_fmt(item['stress'])} MPa</td>"
            f"<td>{_fmt(x)}, {_fmt(y)}, {_fmt(z)}</td>"
            f"<td>{_fmt(item['azimuth'], 1)}° / {_fmt(item['elevation'], 1)}°</td>"
            "</tr>"
        )
    return (
        '<table style="width:100%;border-collapse:collapse;font-size:12px;color:#cbd5e1">'
        '<thead><tr style="color:#94a3b8;text-align:left">'
        '<th style="padding:9px 6px">Plane</th><th>Principal stress</th>'
        '<th>Direction cosines (nₓ, nᵧ, n_z)</th><th>Azimuth / elevation</th>'
        '</tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )


def render_tensor_matrix(result: dict[str, Any]) -> str:
    """Render the symmetric stress matrix, in MPa."""
    matrix = result["matrix"]
    cells = [
        f'<td style="padding:10px 13px;text-align:center;border:1px solid #334155;min-width:76px">{_fmt(matrix[i, j])}</td>'
        for i in range(3) for j in range(3)
    ]
    rows = "".join(
        f'<tr>{"".join(cells[i * 3:(i + 1) * 3])}</tr>' for i in range(3)
    )
    return (
        '<div style="overflow-x:auto"><table style="border-collapse:collapse;color:#e2e8f0;min-width:255px">'
        '<caption style="caption-side:top;text-align:left;color:#94a3b8;padding:0 0 8px;font-size:12px">Stress tensor · MPa</caption>'
        + rows + '</table></div>'
    )


def mechanics_quick_check(mode: str, q1: float, q2: float, q3: float, q4: float) -> dict[str, str]:
    """Evaluate a compact beam, shaft, or thin-wall vessel textbook check.

    Inputs and conventions are intentionally explicit in the UI.  Outputs are
    analytical teaching checks, not substitutes for a design-code assessment.
    """
    values = [float(q1), float(q2), float(q3), float(q4)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Enter finite numeric values.")
    a, b, c, d = values
    if mode == "beam":
        moment_nm, width_mm, height_mm, allowable_mpa = values
        if width_mm <= 0 or height_mm <= 0 or allowable_mpa <= 0:
            raise ValueError("Section dimensions and allowable stress must be positive.")
        inertia = width_mm * height_mm ** 3 / 12.0
        sigma = 6.0 * abs(moment_nm) * 1000.0 / (width_mm * height_mm ** 2)
        safety = allowable_mpa / max(sigma, 1e-12)
        return {
            "title": "Rectangular beam · elastic bending",
            "primary_label": "Maximum bending stress",
            "primary_value": f"{_fmt(sigma)} MPa",
            "secondary_label": "Second moment of area I",
            "secondary_value": f"{_fmt(inertia, 1)} mm⁴",
            "tertiary_label": "Illustrative allowable / demand",
            "tertiary_value": f"{_fmt(safety)} ×",
            "formula": "σₘₐₓ = M c / I = 6 M / (b h²) · I = b h³ / 12",
            "note": "Euler–Bernoulli rectangular section; elastic, prismatic beam assumption.",
        }
    if mode == "shaft":
        torque_nm, diameter_mm, length_mm, shear_modulus_gpa = values
        if diameter_mm <= 0 or length_mm <= 0 or shear_modulus_gpa <= 0:
            raise ValueError("Diameter, length and shear modulus must be positive.")
        polar = math.pi * diameter_mm ** 4 / 32.0
        tau = 16.0 * abs(torque_nm) * 1000.0 / (math.pi * diameter_mm ** 3)
        twist_rad = abs(torque_nm) * 1000.0 * length_mm / (shear_modulus_gpa * 1000.0 * polar)
        return {
            "title": "Circular shaft · Saint-Venant torsion",
            "primary_label": "Maximum surface shear stress",
            "primary_value": f"{_fmt(tau)} MPa",
            "secondary_label": "Angle of twist",
            "secondary_value": f"{_fmt(math.degrees(twist_rad), 3)}°",
            "tertiary_label": "Polar second moment J",
            "tertiary_value": f"{_fmt(polar, 1)} mm⁴",
            "formula": "τₘₐₓ = T r / J = 16 T / (π d³) · θ = T L / (G J)",
            "note": "Solid circular shaft, uniform torque and linear-elastic material.",
        }
    if mode == "vessel":
        pressure_mpa, diameter_mm, thickness_mm, allowable_mpa = values
        if diameter_mm <= 0 or thickness_mm <= 0 or allowable_mpa <= 0:
            raise ValueError("Diameter, thickness and allowable stress must be positive.")
        # Membrane (thin-wall) theory is the usual approximation for r/t >= 10, i.e. D/t >= 20;
        # there the hoop stress is within ~5 % of the Lamé thick-cylinder value.
        if diameter_mm / thickness_mm < 20:
            note = "Thin-wall assumption is weak here; use a thick-cylinder model for design."
        else:
            note = "Thin-walled, closed-end cylindrical vessel; membrane theory assumption."
        hoop = abs(pressure_mpa) * diameter_mm / (2.0 * thickness_mm)
        longitudinal = abs(pressure_mpa) * diameter_mm / (4.0 * thickness_mm)
        safety = allowable_mpa / max(hoop, 1e-12)
        return {
            "title": "Thin-walled cylinder · pressure loading",
            "primary_label": "Hoop stress",
            "primary_value": f"{_fmt(hoop)} MPa",
            "secondary_label": "Longitudinal stress",
            "secondary_value": f"{_fmt(longitudinal)} MPa",
            "tertiary_label": "Illustrative allowable / hoop stress",
            "tertiary_value": f"{_fmt(safety)} ×",
            "formula": "σₕ = p d / (2 t) · σₗ = p d / (4 t)",
            "note": note,
        }
    raise ValueError(f"Unknown mechanics check: {mode}")
