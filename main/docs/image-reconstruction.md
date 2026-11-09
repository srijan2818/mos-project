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

## Three-view solid on the L-shaped marker mat (real 3D)

One photo has no depth and no hidden sides, so every single-image path above assumes them. With
several photos whose camera poses are known, each silhouette removes all space outside it; what is
left is the **visual hull**, the largest solid consistent with every photo.

**Setup.** *Make marker sheets* writes `cad_models/_generated/marker_mat.pdf`: two A4 pages, FLOOR
and BACK, each a 9 × 6 ChArUco board of 30 mm squares (`mos/core/marker.py`, disjoint marker ids).
Print at 100 % and check a square with a ruler. Lay FLOOR flat under the part and stand BACK upright
behind it along the far edge. Their exact relative placement doesn't matter: it is measured.

**Photos** (same camera and zoom):

1. top, straight down;
2. front at **table level**, turned ~20° to one side (a head-on sheet gives the least certain pose);
3. side, ~20° above the table;
4. a 3/4 view from above that shows **both** sheets (it locates BACK relative to FLOOR).

**Why the standing sheet.** A flat mat is not detected below ~15° elevation, and downward-looking
photos leave the height poorly constrained. Measured on a 60 × 40 × 30 mm box with exact silhouettes:

| setup | height | volume excess |
|---|---|---|
| flat mat, 3 photos at 35° | 53.5 mm | +38 % |
| flat mat, 3 photos at 20° | 41.6 mm | +22 % |
| L-mat, 4 photos as above | 30.7 mm | +7 % |

**Pipeline.** `marker.solve_rig` detects both sheets and estimates one focal length (principal point
at the centre, no distortion). It locates BACK from the photo that sees both, then refines all
poses, the BACK pose and the focal length jointly (least squares on every detected corner).

Per photo, **Select part** opens it in the click-to-select preview. Both sheets' projected outlines
are excluded (otherwise the mat is the most prominent "object"). Once two other photos are
accepted, the solid they define, projected into this photo, picks the prompt point and the mask.
`multiview_hull.carve` then carves coarse-to-fine (48 voxels to find the part, 128 across it) and
reports each photo's **agreement**: IoU of its mask with the hull projected back into it. The hull
can only shrink a correct mask, so agreement well below 1 means that mask shows something the
others rule out (wrong object, hand); select that photo again.

The surface comes from marching cubes on a lightly smoothed grid. Gmsh fills the volume directly on
those triangles (re-meshing the surface failed on a real session). Faces `cad_face_N` are grown from
boundary triangles whose normals stay within 35° of the face's mean. The session (masks +
`cameras.json`) is saved under `cad_models/_generated/hull_sessions/` and solved through
`mesh hull SESSION_DIR [mesh_size]`; the mesh is in metres, y up, table at y = 0.

**Accuracy (synthetic, known geometry).**

* Poses: camera within 3 mm and 0.5° (flat mat); L-mat reprojection on the part < 2 px (~0.8 mm).
* 60 × 40 × 30 mm box from estimated poses: 63.3 × 43.5 × 29.8 mm, volume +5 %.
* In the browser, with SAM masks: 64 × 42 × 30 mm.
* Plan dimensions run 5–7 % large in every setup: each camera's cone widens behind the part, so the
  bottom edges are over-estimated. The error shrinks with camera distance.
* A 120 × 20 × 20 mm bar cantilever: tip deflection 21.8 µm vs 20.6 µm (beam theory, true bar).

**Limits.** Concavities that no photo shows (blind pockets, recessed faces) are filled in, so volume
and stiffness are over-estimated there. A through-hole is kept only if some photo looks through it.
Real photos (lighting, focus, a hand in view) still need measuring: a benchmark with
caliper-measured parts is the next step.

## Tests

`tests/test_image_reconstruction.py` verifies transparent-image masking, generated colored PLY
records, scale handling, browser data-URI input, transparent cutout preservation with closed-mesh edge
incidence, and invalid/empty image handling. It does not run
Depth Anything inference or validate the PLY rendering in a live VTK browser; model downloads and
actual CAD/FEM export require the optional runtime tools described above.
