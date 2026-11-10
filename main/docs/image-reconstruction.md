# Single-image reconstruction

The **Universal Image → 3D → CAD Viewer** workflow turns one uploaded image into an inspectable
colored 2.5D surface. It is intentionally not described as complete single-view 3D reconstruction:
one image cannot reveal hidden surfaces, and monocular depth predictions are relative rather than
metric.

## Pipeline

```text
JPG / PNG / WEBP / BMP
       │
       ▼
foreground mask ──► downsample / clean mask
       │                        │
       │                        ▼
       │                 optional Depth Anything V2
       │                        │
       └────────────────────────┘
                                ▼
                     colored closed relief mesh
                                │
                     PLY preview in CAD viewer
                                │
                     optional silhouette mask
                                ▼
                       Gmsh STEP extrusion
                                │
                     existing CAD / FEM importer
```

### Masking

The masking order is: PNG alpha channel, optional `rembg`, then a border-color contrast estimate.
Install background removal when photos have complex surroundings:

```bash
pip install "rembg[cpu]"
```

A transparent PNG is the most predictable input. The border-color fallback is heuristic, and the
UI reports when it is used. The full-frame option intentionally makes a rectangular relief from the
entire image.

### Relative depth

AI mode uses the Transformers pipeline for
[`depth-anything/Depth-Anything-V2-Small-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf).
The checkpoint is lazily loaded and cached. On first use, Transformers needs network access to
fetch the weights; set `MOS_DEPTH_MODEL` to another compatible depth-estimation model ID to
override the default. CPU inference is supported, although it can be slower than a CUDA device.

The inferred values are normalized to a user-selected relief amplitude. They are not metric depth.
The displayed scene width is an explicit user assumption, and the hidden side is approximated by a
planar extrusion. This is useful for visual inspection and rough concept geometry, not dimensional
metrology.

### Export and FEM boundary

The colored preview is a closed front-relief/back-plane mesh stored as PLY with RGB vertex colors.
The STEP export is generated separately from the binary foreground silhouette using the existing
`image_to_step` / Gmsh path. It is a constant-thickness extrusion, so it will not preserve the
front-surface depth relief. Both the PLY preview and the STEP extrusion keep enclosed cutouts as
through-holes (the largest part by area and its holes are traced). The STEP import action is
explicit and reuses the existing CAD-to-FEM pipeline. If Gmsh is unavailable, the PLY preview still
works, but STEP/FEM loading is not offered.

## Direct mask meshing (no STEP)

`mos.core.image_to_mesh` meshes a mask straight into the FE model, which is the recommended route
from a photo to an analysis:

```text
mask --contours (pixel-edge accurate)--> outer polygon + holes --Gmsh built-in kernel, in memory-->
2D triangles --structured extrusion--> conforming tetrahedra --> MeshTet with named faces
```

It is exposed in the model language as `mesh mask PATH size thickness [mesh_size]` (see
[model-language.md](model-language.md)), so the text model stays the single source of truth, and the
mesh is cached per (file, settings). Compared with the template route (classify the photo as one of
six part types, guess the other dimensions, write STEP, re-import, re-mesh) it meshes what was
actually segmented, cannot pick the wrong category, keeps arbitrary outlines and holes, and takes
well under a second. Its single assumption is a constant depth normal to the photo, which is exact
for plates, side-on brackets and end-on cross-sections of prismatic members.

Before interpreting a simulation, manually validate the reconstructed shape, scale, wall thickness,
material, supports and loads. A successful mesh or solver run does not validate the physical accuracy
of a shape inferred from a single photograph.

## 3-photo solid (real 3D, no markers)

One photo has no depth and no hidden sides. Three photos along the part's axes do: treating each
as an orthographic projection, a point belongs to the part only if it lies inside the outline in
all three, so the solid is the **intersection of the three extruded outlines** (the visual hull of
three orthogonal views, `mos/core/multiview_hull.py`).

**Photos** (any camera, any zoom, plain background):

| photo | camera | image → | image ↑ |
|---|---|---|---|
| Front | level with the part, looking straight at it | length x | height z |
| Top | straight down, the front edge at the bottom of the photo | length x | depth y (away) |
| Right side | level, from the part's right | depth y (front → back) | height z |

In each photo *Select part* + *Accept* (click-to-select, as for a single photo). The panel then
shows each accepted outline as it will be used, cropped and oriented, with its axes; *Flip* and
*Rotate 90°* fix a photo taken from the other side or in portrait. Enter **one measured
dimension** (ruler or caliper, along x, y or z) and *Build solid & solve FEA*.

**Registration and scale.** No markers: each outline is cropped to its bounding box. Every axis
is seen by two photos, so their scales `k` (m/px) satisfy `k_i n_i = k_j n_j` (`n` = pixels along
that axis): three equations in three unknowns, one of which is the common factor fixed by the
measured dimension (`fit_scales`). In logarithms the equations form a cycle; least squares spreads
the closure residual `r` equally, and `|e^r − 1|` is reported as how far the photos **disagree on
the proportions** (warning above 5 %). Each outline is then stretched onto the common size and the
three are intersected on a grid of 128 voxels along the longest side.

**Checking a selection.** The proportion mismatch is the reliable check: a hand or a second
object in one outline changes its proportions (the test case reads 66 %). Projecting the solid back
into each photo (the agreement score of the earlier marker-mat method) is *not* used: with three
orthogonal views the solid reproduces an outline almost whenever the other two are solid shapes,
so a mirrored photo or a hand inside the outline usually still "agrees". A stepped block with its
side photo mirrored gives a different, fully consistent solid (`tests/test_multiview_hull.py`),
which is why the oriented outlines are shown for checking by eye.

The surface comes from marching cubes on a lightly smoothed grid; Gmsh fills the volume directly on
those triangles. Faces `cad_face_N` are grown from boundary triangles whose normals stay within 35°
of the face's mean. The session (masks + `ortho.json` with orientation and the measured dimension)
is saved under `cad_models/_generated/hull_sessions/` and solved through
`mesh hull SESSION_DIR [mesh_size]`; the mesh is in metres, y up, table at y = 0.

**Accuracy (synthetic silhouettes, known geometry).**

* Orthographic photos, 60 × 40 × 30 mm box at three different zooms: size within one voxel
  (~0.5 mm), volume within 3 %, scales recovered within 1 %, proportion mismatch ~1 % (pixel
  rounding of the bounding boxes).
* L-bracket (80 × 30 mm foot, 50 mm upright, 10 mm thick): volume −0.6 %.
* **Perspective.** A magnification common to a whole photo cancels (bounding boxes are rescaled,
  the measured dimension fixes the scale), so a box is exact even from close up. Parts of one
  photo at different depths are magnified differently (relative error ~ depth difference /
  distance). Pinhole renders of the L-bracket, camera at d × its largest size:

  | d | extents | volume | 10 mm upright |
  |---|---|---|---|
  | 3× | ≤ 11 % | +7.6 % | 13.8 mm |
  | 5× | ≤ 6 % | +7.1 % | 12.5 mm |
  | 10× | ≤ 3 % | +3.8 % | 11.3 mm |

  Shoot from far and zoom in.
* A 120 × 20 × 20 mm bar at the default element size (1/15 of the part): hull mesh
  120.0 × 20.0 × 20.0 mm, cantilever tip deflection 21.6 µm vs 20.7 µm for a box mesh of the
  same size and 20.6 µm from beam theory (+5 %: the smoothed surface rounds the section's edges).

**Meshing.** For the FE mesh the hull is carved again at about two voxels per element (default
element size 1/15 of the part, at least 24 voxels along the longest side) and surfaced by
marching cubes on every voxel with 0.5-voxel smoothing. Surfacing a fine grid with marching-cubes
steps of 3–4 instead left Gmsh unable to fill the surface (no tetrahedra; found in the browser
test). Smoothing by a full voxel rounded the section's edges more (+11 % on the bar); 0.5 voxel
can leave the rounded edges as extra small faces next to the six main ones.

**Limits.** Exact (to the voxel) only for shapes that are an intersection of three extrusions:
blocks, brackets, plates with through-holes seen along an axis. Concavities that no view shows
(blind pockets, recesses, chamfers seen only obliquely) are filled in, so volume and stiffness are
over-estimated there. A tilted photo adds about D sin θ of the depth to the width. Real phone
photos (lighting, lens distortion, a hand in view) have not been benchmarked yet.

## Tests

`tests/test_image_reconstruction.py` verifies transparent-image masking, generated colored PLY
records, scale handling, browser data-URI input, transparent cutout preservation with closed-mesh edge
incidence, and invalid/empty image handling. It does not run
Depth Anything inference or validate the PLY rendering in a live VTK browser; model downloads and
actual CAD/FEM export require the optional runtime tools described above.
