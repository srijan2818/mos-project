# The app

```text
python -m mos.app --server                       # starts a local server and prints its address
python -m mos.app --server --spec model.mos      # start with a model file loaded
python -m mos.app --server --port 8080
```

The app is a [trame](https://kitware.github.io/trame/) server: a Python process renders the 3D
scene with PyVista (VTK) and streams it to a browser tab. The toolbar switches between the
**Analysis Workspace** and a full-width **Learning Lab**.

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
3. Explore the **Results** controls, or use **Loads & Supports** to apply moments and forces on
   faces and points.
4. Switch to **Learning Lab** for point-stress experiments, principal-plane visualisation, Mohr's
   circles, beam bending, shaft torsion and thin-wall pressure-vessel checks. See
   [learning-lab.md](learning-lab.md) for equations, conventions and limitations.

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


## Universal Image → 3D → CAD Viewer

Under **Geometry Source → Real-Life CV Mode**, open **Universal Image → 3D → CAD Viewer** and
upload a JPG, PNG, WEBP or BMP image (up to 15 MB). This is separate from the existing mechanical
part detector and does not replace the multi-view reconstruction workflow.

1. **AI relative-depth relief** runs the Hugging Face `depth-anything/Depth-Anything-V2-Small-hf`
   model through Transformers. The first run downloads model weights; later runs use the local
   Transformers cache. The result is a colored, closed 2.5D relief mesh saved as PLY. Its depth is
   relative, not metric, and the back side is an extrusion approximation.
2. **Silhouette extrusion** skips neural depth estimation and creates a flat relief from the
   foreground mask. It is useful for transparent PNGs and simple-background product photos.
3. **Foreground extraction** first uses an input PNG alpha mask, then optional `rembg` when
   installed, and otherwise estimates the foreground from border colors. For difficult/busy
   backgrounds, install the optional CPU background-removal package with `pip install "rembg[cpu]"`
   or upload a transparent PNG. The **Use entire image frame** option intentionally creates a
   rectangular image-backed relief.
4. **Reconstruct 3D Preview** writes `cad_models/image_reconstruction/*_2_5d_preview.ply`, a mask
   PNG, and (for AI mode) a relative-depth PNG. Click **Show 3D Preview in CAD Viewer** to inspect
   the colored surface. This preview is isolated from the FEM model: it does not replace the last
   solved mesh or cached solver. **Clear Preview** returns to the previous FEM view.
5. When Gmsh is available, the same foreground mask can also be exported as a constant-thickness
   STEP silhouette. **Load Silhouette STEP into CAD / FEM** explicitly switches the active geometry
   to this STEP file and uses the existing CAD importer / mesher. This STEP body is a 2D silhouette
   extrusion; it is not the estimated depth-relief surface mesh.

The approximate width and thickness are user inputs, not measurements inferred from the photo.
Monocular depth models predict relative depth and cannot recover hidden geometry from one view.
Treat the output as a visualization/starting shape, inspect it, and do not interpret an FEM result
as physically reliable until the geometry, dimensions, materials, supports and loads have been
independently checked. For a more complete object, take 20–60 overlapping views and use the
multi-view workflow below. See [image-reconstruction.md](image-reconstruction.md) for details.

## How it is built

`mos/app/main.py` keeps UI state in trame's `state`. The Solve button runs `parse_spec`, `build` and
`solve_case`, converts the mesh with `viz.to_pyvista`, computes the plottable scalars with
`viz.compute_scalars` and stores them; changing a view control redraws from the stored solution
without re-solving. The Learning Lab's idealised point-stress calculations and SVG renderers live in
`mos/learning.py`; they are deliberately separate from the FEM result state.

## Troubleshooting

* **Nothing happens or a Python error on start**: the app needs `pyvista`, `trame`, `trame-vtk`,
  `trame-vuetify`, `trame-pyvista` and `meshio` (`pip install -r requirements.txt`). On a headless Linux machine
  VTK needs an off-screen OpenGL backend (OSMesa or EGL).
* **Red status with a line number**: a model-text error; see the table in
  [model-language.md](model-language.md).
* **Huge displacements or a warning status**: usually an under-constrained model (add supports).
* **A hex mesh looks scrambled**: the viewer relies on scikit-fem's meshio export for node order
  on hexahedra; tetrahedra (the default) are the tested path. Report it.
* **A solve error mentioning a scikit-fem call**: see [testing.md](testing.md), "First run".


## Multi-view NeRF / Gaussian Splatting and Gaussian-to-CAD preparation

The **Multi-view 3D Reconstruction** accordion is separate from the existing single-image
detector. Upload multiple photos of one stationary object. The capture builder normalizes the
images, removes exact pixel duplicates, measures approximate sharpness and brightness, saves a
manifest, and writes a session under `cad_models/reconstruction_sessions/`. Capture validation
and Gaussian PLY inspection work without installing a neural reconstruction framework.

For a useful scan, take roughly 20–60 sharp, overlapping views around the object at multiple
heights. Move the camera rather than the object, keep zoom/focus and lighting consistent, and
avoid motion blur, reflective glare, and large gaps between adjacent views.

Choose the reconstruction method in the panel:

* **3D Gaussian Splatting · Splatfacto**: with Nerfstudio, COLMAP and FFmpeg installed,
  **Prepare Nerfstudio Poses · COLMAP** runs `ns-process-data images` to estimate camera poses
  and create `transforms.json`. Then run `ns-train splatfacto --data <processed-folder>`. A trained
  Gaussian model can be exported with `ns-export gaussian-splat --load-config <config.yml>
  --output-dir <output-folder>`.
* **NeRF · Nerfacto**: uses the same Nerfstudio preprocessing path, followed by
  `ns-train nerfacto --data <processed-folder>`. This represents a neural radiance field rather
  than a Gaussian PLY.
* **3D Gaussian Splatting · Graphdeco trainer**: **Build Standard COLMAP Dataset** runs
  COLMAP feature extraction, exhaustive matching, sparse mapping, and PLY export. It creates the
  conventional `images/` and `sparse/0/` layout expected by the Graphdeco trainer, using a
  `SIMPLE_PINHOLE` camera model supported by the common 3DGS loaders. Run
  `python train.py -s <capture-session> -m <model-output>` from a separately installed
  `gaussian-splatting` repository.
* **2D Gaussian Splatting**: uses the same standard COLMAP dataset action and is trained from
  the external 2DGS repository with `python train.py -s <capture-session> -m <model-output>`.
  After training, that repository can export a TSDF-fused surface mesh with
  `python render.py -m <model-output> -s <capture-session>`. These training and meshing commands
  run in the external repository; the app only prepares the compatible camera/image dataset.

The **Gaussian PLY → CAD preparation point cloud** panel reads common ASCII and binary
little-endian Gaussian PLY files. It extracts the Gaussian means and RGB values into a standard
colored point cloud, reports bounds/centroid and available opacity/scale statistics, and can
apply a uniform metric scale from one known X/Y/Z extent. This is a useful bridge for inspection
and downstream surface-fitting tools, not a direct Gaussian-to-STEP conversion: splat centers
are not a watertight boundary representation. Surface reconstruction, outlier cleanup, scale
validation and CAD feature fitting are still needed before engineering analysis.

Nerfstudio, COLMAP and FFmpeg are optional external tools; the base project does not install
or execute neural training. The app reports which command-line tools are available and leaves
training to their respective environments. COLMAP's direct dataset builder currently uses CPU
SIFT extraction/matching for compatibility with CPU-only installations; large image sets can
therefore take substantially longer than GPU-assisted processing.
