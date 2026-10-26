# The app

```text
python -m mos.app                       # starts a local server and prints its address
python -m mos.app --spec model.mos      # start with a model file loaded
python -m mos.app --port 8080
```

The app is a [trame](https://kitware.github.io/trame/) server: a Python process renders the 3D
scene with PyVista (VTK) and streams it to a browser tab.

## Workflow

1. Pick a model from **Load example**, or edit the text in the editor (see
   [model-language.md](model-language.md)). The examples include:
   * *Cantilever - tip load*
   * *Cantilever - self weight*
   * *Bar - torsion*
   * *Cantilever - partial pressure*
   * *Simply supported - center load*
   * *Cantilever - pure end moment*
   * *Bar - axial tension*
   * *Fixed-fixed beam - uniform load*
   * *Cantilever - point force and custom axis moment*
   * *Overhanging beam - tip and span loads*
   * *CAD - Structural L-Bracket (STEP)*
   * *CAD - Stepped Shaft Torsion (STEP)*
   * *CAD - Plate with Hole (STEP)*
   * *CAD - Image to STEP Drawing Profile*
2. Press **Solve**. The status panel shows the cell and DOF counts, the maximum displacement and
   von Mises stress, the total reaction, and the equilibrium error. The status turns green when the
   equilibrium error is below 1e-6 and amber otherwise; errors in the model text are shown in red
   with their line number.
3. Explore with the view controls, or use the **Apply Loads** tab to interactively apply moments
   and forces on faces and points.

## Controls

| Control | Effect |
|---|---|
| **Theme** | Dark mode by default (slate theme with dark 3D background); toggle to light mode anytime via the toolbar icon. |
| **Show** | scalar plotted on the body: `\|u\|`, `ux uy uz`, von Mises, principal stresses, Tresca, every stress component, strain components (`gamma` are engineering shear strains), principal strain |
| **Stress unit** | Pa, kPa, MPa or GPa; display only, the model is not rescaled |
| **Deformation scale** | multiplies the displacement drawn. Defaults to **1:1 (True Scale)**. Quick presets allow switching instantly between `1:1 True`, `Auto (~10%)` and `0 Undeformed`. A faint outline of the undeformed shape is drawn whenever the scale is above 0. |
| **Interactive Loads** | select faces or specify points on the body, then click to apply moments about an axis of your choice or forces with real-time 3D highlight and glyph indicators. |
| **Mesh edges** | draws the element edges |
| **Load glyphs** | draws 3D arrows and moment indicators representing applied forces and moments |
| **Section plane normal / position** | cuts the body with a plane normal to x, y or z at a fraction of the bounding box. The colour range stays that of the whole body. |

Colours show the smoothed nodal values (see [theory.md](theory.md), section 6): good for the
overall picture, not for the exact peak.

## Real-Life CV Mode (Computer Vision to CAD)

The interactive builder includes a dedicated **Real-Life CV Mode** under Geometry Source:
1. **Camera or Image Capture**: Take a real-life picture with your device camera / webcam using the snapshot button, upload an image file from disk, or select from built-in real-life presets (Structural L-Bracket, Tensile Plate with Hole, Rectangular Beam, Structural I-Beam, Stepped Motor Shaft, or Hollow Pipe).
2. **Lightweight CV Classification & Feature Extraction**: The lightweight vision model segments the object from the desk/background, extracts invariant morphological metrics (aspect ratio, convex hull solidity, Euler hole topology, re-entrant notch angles, and bilateral symmetry), and identifies the mechanical part with a confidence score.
3. **Parametric CAD Solid Generation**: Extracted pixel dimensions are calibrated to physical metric units (meters) and displayed in editable parameter fields. Clicking **Generate 3D CAD Solid & Solve FEA** synthesizes a watertight 3D STEP solid using OpenCASCADE and Gmsh, automatically meshes it, identifies boundary faces for supports and loads, and renders the solved linear-elastic stresses in the 3D viewport.

## How it is built

`mos/app/main.py` keeps all UI state in trame's `state` (`theme`, `spec_text`, `field`, `stress_unit`,
`scale`, `scale_display`, `show_edges`, `selected_face`, `load_target`, `clip_axis`, `clip_pos`, ...).
The Solve button runs `parse_spec`, `build` and `solve_case`, converts the mesh with `viz.to_pyvista`,
computes the plottable scalars with `viz.compute_scalars` and stores them; changing any view control
redraws from the stored solution without re-solving. All numerical work is in `mos.core`; the app only
wires it to widgets.

## Troubleshooting

* **Nothing happens or a Python error on start**: the app needs `pyvista`, `trame`, `trame-vtk`,
  `trame-vuetify` and `meshio` (`pip install -r requirements.txt`). On a headless Linux machine
  VTK needs an off-screen OpenGL backend (OSMesa or EGL).
* **Red status with a line number**: a model-text error; see the table in
  [model-language.md](model-language.md).
* **Huge displacements or a warning status**: usually an under-constrained model (add supports).
* **A hex mesh looks scrambled**: the viewer relies on scikit-fem's meshio export for node order
  on hexahedra; tetrahedra (the default) are the tested path. Report it.
* **A solve error mentioning a scikit-fem call**: see [testing.md](testing.md), "First run".
