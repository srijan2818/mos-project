"""Example model descriptions shown in the app's 'Load example' menu."""

TEMPLATES = {
    "Cantilever - tip load": """\
# Cantilever, tip shear load.  Units: SI (m, N, Pa).
# Beam theory: tip deflection = P L^3 / (3 E I),  I = W H^3 / 12
param L = 10
param H = 1
param W = 1
param P = 1e6

mesh     box L H W 20 2 2
material E = 210e9, nu = 0.3, rho = 7850

region   fixed facets x < 1e-9
region   tip   facets x > L - 1e-9

support  fixed xyz
traction tip 0, -P/(H*W), 0
""",
    "Cantilever - self weight": """\
# Cantilever under its own weight (body force from rho, defined by 'material').
# Beam theory: tip deflection = q L^4 / (8 E I),  q = rho g W H
param L = 10
param H = 1
param W = 1
param g = 9.81

mesh     box L H W 20 2 2
material E = 210e9, nu = 0.3, rho = 7850

region   fixed facets x < 1e-9
support  fixed xyz

body     0, -rho*g, 0
# any field works, e.g. a body force that doubles toward the tip:
# body   0, -rho*g*(1 + x/L), 0
""",
    "Bar - torsion": """\
# Square bar twisted by a torque T about its axis (a moment along the end-face normal).
param L = 10
param a = 1
param T = 1e5

mesh     box L a a 24 4 4
material E = 210e9, nu = 0.3

region   fixed facets x < 1e-9
region   tip   facets x > L - 1e-9

support  fixed xyz
moment   tip T, 0, 0
""",
    "Cantilever - partial pressure": """\
# Pressure on the top face over the outer half only, built with step().
# Equivalent:  pressure top where(x > L/2, p0, 0)
param L = 10
param H = 1
param W = 1
param p0 = 5e6

mesh     box L H W 30 3 3
material E = 210e9, nu = 0.3

region   fixed facets x < 1e-9
region   top   facets y > H/2 - 1e-9

support  fixed xyz
pressure top step(x - L/2) * p0
""",
    "Simply supported - center load": """\
# Simply supported beam with knife-edge bottom supports and center point load.
# Beam theory: midspan deflection = P L^3 / (48 E I),  I = W H^3 / 12
param L = 10
param H = 1
param W = 1
param P = 1e6

mesh     box L H W 20 2 2
material E = 210e9, nu = 0.3

region   left_line  nodes x < 1e-9 and y < -H/2 + 1e-9
region   right_line nodes x > L - 1e-9 and y < -H/2 + 1e-9
region   mid_top    nearest L/2, H/2, 0

support  left_line  xyz
support  right_line yz
force    mid_top    0, -P, 0
""",
    "Cantilever - pure end moment": """\
# Cantilever with a pure bending moment applied at the tip face about the z-axis.
# Beam theory: tip deflection = M L^2 / (2 E I), tip slope = M L / (E I)
# Constant bending moment M throughout the beam length.
param L = 10
param H = 1
param W = 1
param M = 1e6

mesh     box L H W 20 2 2
material E = 210e9, nu = 0.3

region   fixed facets x < 1e-9
region   tip   facets x > L - 1e-9

support  fixed xyz
moment   tip axis 0, 0, 1 mag M
""",
    "Bar - axial tension": """\
# Prismatic bar under uniform axial tension.
# Theory: axial stress sigma_xx = F / (H * W)
# Tip elongation delta = F L / (E H W)
param L = 10
param H = 1
param W = 1
param F = 2.1e7

mesh     box L H W 20 2 2
material E = 210e9, nu = 0.3

region   fixed facets x < 1e-9
region   tip   facets x > L - 1e-9

support  fixed xyz
traction tip F/(H*W), 0, 0
""",
    "Fixed-fixed beam - uniform load": """\
# Both ends fully clamped under uniform downward pressure on top face.
# Beam theory: center deflection = p0 W L^4 / (384 E I),  I = W H^3 / 12
param L = 10
param H = 1
param W = 1
param p0 = 1e6

mesh     box L H W 20 2 2
material E = 210e9, nu = 0.3

region   left_fixed  facets x < 1e-9
region   right_fixed facets x > L - 1e-9
region   top_face    facets y > H/2 - 1e-9

support  left_fixed  xyz
support  right_fixed xyz
pressure top_face    p0
""",
    "Cantilever - point force and custom axis moment": """\
# Cantilever demonstrating point force and moment about a custom axis.
# Force applied at a specific point; moment applied about an inclined axis [1, 1, 0].
param L = 10
param H = 1
param W = 1
param P = 5e5
param M = 2e5

mesh     box L H W 20 2 2
material E = 210e9, nu = 0.3

region   fixed facets x < 1e-9
region   tip   facets x > L - 1e-9
region   pt    nearest L, H/2, 0

support  fixed xyz
force    pt  0, -P, 0
moment   tip axis 1, 1, 0 mag M about L, 0, 0
""",
    "Overhanging beam - tip and span loads": """\
# Overhanging beam with left support at x = 0, middle support at x = 7, and overhang to L = 10.
# Downward point load on the overhang tip (x = 10).
param L = 10
param H = 1
param W = 1
param P = 5e5

mesh     box L H W 20 2 2
material E = 210e9, nu = 0.3

region   root_pin facets x < 1e-9
region   mid_roll facets abs(x - 7) < 0.6 and y < -H/2 + 1e-9
region   tip_pt   nearest L, 0, 0

support  root_pin xyz
support  mid_roll y
force    tip_pt   0, -P, 0
""",
    "CAD - Structural L-Bracket (STEP)": """\
# Structural L-bracket imported from a STEP CAD file
# Meshed automatically into 3D tetrahedral elements via Gmsh
mesh     step cad_models/bracket.step 0.6
material E = 200e9, nu = 0.3, rho = 7850

support  cad_face_1 xyz
traction cad_face_6 0, -1e6, 0
""",
    "CAD - Stepped Shaft Torsion (STEP)": """\
# Stepped shaft imported from a STEP CAD file
# Fixed at root face (x = 0), pure torque applied at tip face (x = 10)
mesh     step cad_models/shaft.step 0.4
material E = 210e9, nu = 0.3, rho = 7850

support  cad_face_3 xyz
moment   cad_face_5 1000, 0, 0
""",
    "CAD - Plate with Hole (STEP)": """\
# Tensile plate with circular hole imported from STEP
# Stress concentration benchmark (Kirsch problem)
mesh     step cad_models/plate_hole.step 0.35
material E = 70e9, nu = 0.33, rho = 2700

region   left_face  facets x < 1e-6
region   right_face facets x > 6.0 - 1e-6

support  left_face  xyz
traction right_face 1e7, 0, 0
""",
    "CAD - Image to STEP Drawing Profile": """\
# Extruded profile generated directly from a 2D drawing / sketch
mesh     step cad_models/drawing_extruded.step 0.25
material E = 200e9, nu = 0.3, rho = 7850

support  cad_face_1 xyz
traction cad_face_2 0, -5e5, 0
""",
}

