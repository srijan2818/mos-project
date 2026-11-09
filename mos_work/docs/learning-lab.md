# Integrated Analysis and Learning Lab

The Learning Lab is integrated into the Analysis Workspace as a contextual analysis dock.
The original 3D viewport, coordinate space, camera and view controls remain the primary
work surface. Opening the dock hides the setup drawer and overlays one analysis surface on
the viewport; it does not create a third side-by-side panel or replace the model view.

The dock keeps three items in one context: the active load being explored, the selected
sample point, and the selected analysis plane. Picking another point or editing its XYZ
coordinates updates the result readouts while retaining the current plane selection. The
stress tensor, principal values/directions, von Mises, Tresca, maximum shear, Mohr circles,
traction vector and normal/shear decomposition are all derived from the current `Results`.
Stress values are displayed in MPa; core arrays and APIs remain in SI units (Pa).

## Cached live-load exploration

For a supported, linear-elastic model, the first Solve assembles the stiffness matrix and
factorises the constrained system once. The same factorisation is used for the initial
solution and retained for interactive exploration. When a user selects an applied point
force, surface force, moment or uniform pressure, the app prepares response fields for the
selected load:

- Force and moment: one unit response for each X, Y and Z component.
- Uniform pressure: one unit-pressure response.
- Other applied loads remain in the baseline response.

Changing a component or the load-scale control then composes the cached displacement and
force fields by linear superposition. Strain, stress, reactions, scalar contours and the
selected point/plane outputs are recovered from that combined state. Moving these controls
does not reassemble/factorise the stiffness matrix and does not perform another FE backsolve.
The viewport load glyph length also changes with load magnitude, with direction shown by the
arrow or moment symbol. While the analysis dock is open, the selected mesh node is marked on
the displayed (possibly deformed) shape, with a translucent plane patch, green plane-normal
arrow and amber traction arrow. Picking on an amplified deformed view is mapped back to the
corresponding reference-mesh node before stress sampling.

This optimisation relies on linear elasticity and an unchanged stiffness operator. A new
full Solve is required after changing geometry, mesh, material or supports. The current live
explorer is limited to 30,000 displacement DOFs to bound factorisation memory use. It
supports physically constrained models and point forces, surface forces, moments, and
uniform numeric pressure. Free-body inertial-frame cases and spatially varying pressure
fields continue to use the normal Solve workflow; their live controls are disabled rather
than displaying stale results. Adding or removing loads/supports also uses the regular Solve
path to establish a new baseline and response cache.

## Point-driven stress analysis

Click a point on the 3D viewport while the analysis dock is open, or edit the XYZ coordinates
in the dock. The sample uses the nearest mesh node and its nodal, volume-weighted stress
field. The app reports the selected coordinates, nearest-node index, stress tensor, principal
stresses and directions, von Mises, Tresca and maximum shear. Strain and displacement at
the sampled node are available in the expandable details. When live-load exploration is
active, all of these readouts follow the newly composed result immediately.

The standalone tensor experiment fields remain available in the `mos.learning` renderers;
they are not used to modify the mesh solution.

## Traction on a selected plane

Choose a Cartesian normal, a principal-plane normal at the sampled point, a mesh face normal,
or enter an arbitrary non-zero normal vector. The vector is normalised before evaluation.
The dock reports `t = σ n`, normal stress `σn = n · t`, shear vector `τ = t - σn n`,
and shear magnitude `||τ||`. Changing the point, plane or live load refreshes the traction
results without clearing the other selections.

## Solved-model snapshot

The dock exposes mesh counts, material properties, load/support counts, strain and
 displacement at the sampled node as read-only information derived from the current solved
model/result. These readouts are refreshed after a full solve and after every live response
update.

## Idealised analytical checks

The beam, shaft and thin-wall vessel checks are independent closed-form calculations. They
are explicitly idealised: they neither read nor modify the mesh solution, loads, supports or
solver state. They are not code-compliant design verification.

## Extensibility

Post-FEA features live under `mos.lab`. Each adapter consumes solved result/model data and is
registered in `mos.lab.FEATURES`; adding future analysis such as buckling or torsion should
not require duplicating solver logic. Core result arrays remain in SI units.
