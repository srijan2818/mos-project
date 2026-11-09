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
front-surface depth relief. The PLY preview preserves enclosed transparent cutouts as through-holes;
the current contour-to-STEP converter uses the largest outer contour, so those inner holes may be
omitted from STEP and must be checked before FEM. The STEP import action is explicit and reuses the existing CAD-to-FEM
pipeline. If Gmsh is unavailable, the PLY preview still works, but STEP/FEM loading is not offered.

Before interpreting a simulation, manually validate the reconstructed shape, scale, wall thickness,
material, supports and loads. A successful mesh or solver run does not validate the physical accuracy
of a shape inferred from a single photograph.

## Tests

`tests/test_image_reconstruction.py` verifies transparent-image masking, generated colored PLY
records, scale handling, browser data-URI input, transparent cutout preservation with closed-mesh edge
incidence, and invalid/empty image handling. It does not run
Depth Anything inference or validate the PLY rendering in a live VTK browser; model downloads and
actual CAD/FEM export require the optional runtime tools described above.
