# Sparse FEM method

`SolverFEM` advances position-only Tet4 and Q1 Hex8 solids in SI units.
Tet4 uses constant reference gradients; Hex8 uses $2\times2\times2$ Gauss
quadrature. Both assemble $3\times3$ blocks into a sparse BSR matrix. “Linear”
refers to the shape functions; the constitutive force is nonlinear.

The [paper-to-code mapping](papers.md) identifies the implemented portions of
Stable Neo-Hookean elasticity, embedded cutting and STA-FEM streaming assembly.

## Elasticity and stepping

For deformation gradient $F$, $I_C=\mathrm{tr}(F^T F)$, $J=\det F$, and
Lamé coefficients $\mu,\lambda>0$, the smooth-branch energy is

$$
\psi(F)=\frac\mu2(I_C-3)-\frac\mu2\log(I_C+1)
 +\frac\lambda2(J-\alpha)^2,\qquad
\alpha=1+\frac{3\mu}{4\lambda}.
$$

Its stress is $P=\mu(1-1/(I_C+1))F+\lambda(J-\alpha)\mathrm{cof}\,F$.
The force evaluates clamped principal stretches (default $[0.05,20]$) and
replaces $J$ by $\max(J,0.01)$ in the volumetric stress. Thus the displayed
energy derivative applies on the unclamped branch. Viscous forces are explicit.
These regularizations do not guarantee positive integrated element volumes.

With lumped mass $M$, step $h$, approximate stiffness $\widetilde K$, and
assembled spring damping $C_s$, each substep solves

$$
A=M+h^2\widetilde K+hC_s,\qquad
A\Delta v=h(f_{\rm elastic}+f_{\rm viscous}+f_{\rm contact}+f_{\rm ext}+Mg)
-h^2\widetilde K v,
$$

$$
v^+=v+\Delta v,\qquad x^+=x+h v^+.
$$

The omitted geometric Hessian terms make this a linearly implicit update,
not a converged nonlinear equilibrium. Block-Jacobi Krylov solves never form
a dense global matrix. The showcase refreshes the tangent each frame and uses
10 substeps/50 CG iterations; cutting uses 4/30 at 60 Hz. Tet showcase/bear
meshes also carry unique-edge springs. Hex showcase masses use a node-0/node-6
volume estimate and per-node floor $15h\,\mathrm{kg/s}$; cutting instead
integrates occupied volume. These example mass prescriptions differ.

Ground contact is an analytic penalty with inward normal damping and velocity
friction capped by both Coulomb force and particle weight. Surface contact
uses evolving boundary geometry. External forces enter through
`step(..., external_forces=forces)`; incoming `particle_f` is workspace.

Stationary boundaries use `SolverFEM(..., fixed_particle_indices=indices)`.
With $P$ the projector onto free nodes, the solve is
$(PAP+I-P)\Delta v=Pb$. Boundary positions remain at the model's construction
positions and boundary velocities are zero. Physical masses and the original
matrix remain intact, including the constrained columns needed by the elastic
velocity predictor. Boundary state entries are restored before force evaluation
and after integration; tangent/topology reassembly also restores fixed positions.
This option currently excludes differentiable stepping,
streaming topology and component scaling; no gradient support is claimed for it.

## Cuts and differentiation

Cuts clip material into convex fragments while preserving the parent affine
basis. Disconnected regions receive copied basis nodes; fragment volumes weight
energy and barycentric first moments determine mass. Cached element blocks are
subtracted/added in reserved sparse adjacency, and active rows are compacted
without moving captured storage. Capacity failures roll back. Tiny components
use mass scaling; a bounded volume guard limits distortion. Neither the guard
nor the penalty contact guarantees inversion/intersection freedom.

For a fixed assembled $A$, differentiable mode records the force/predictor and
state update while using the sparse linear-solve adjoint

$$
\Delta v=A^{-1}b(x,v,u),\qquad
A^Tz=\overline{\Delta v},\qquad \bar b\mathrel{+}=z.
$$

Supported derivatives are initial position/velocity, external nodal force, and
Tet activation. Geometry, mass, material, $h$, contact coefficients, and matrix
assembly are fixed; cut topology and contact creation are discrete. Clamps and
contact switches have branch-local derivatives. Checked forward/transpose
solves require $\|Ax-b\|\le2\cdot10^{-5}\|b\|+10^{-10}$; failures raise errors.
This differentiates a converged fixed-matrix solve, not finite CG iterations.
Use unique taped states; rebuild graphs after model changes. Ordinary untaped
stepping allocates no adjoint/trajectory storage. `SolverLinearFEM` is an alias.

The 1,986-node/5,354-tet bear uses 80 substeps at 60 Hz. Eight sinusoidal inputs
produce $u_e(t)=0.3\tanh(\sum_k W_{ek}\sin(5t+2\pi k/8))$; Adam minimizes
$\sum_n(|\bar v_y|+|\bar v_z|-\bar v_x)$ with arithmetic mean vertex velocity.
Frame checkpoints recompute substeps during backward; display runs fresh
inference. Material parameters stay fixed. See [validation](validation.md)
for gradient checks, measured performance and reproducible commands.

## Phantom region geometry

The phantom retains its original USD payload identities and native coordinates.
Only the original connected volume containing wound 5 is partitioned into
three domains: region 5 contains the slit/oval and its thin upper rim, region 8
retains the remainder including the lower-left tail, and region 12 contains
the upper slit between wounds 3 and 4.
The other nine original region payloads are unchanged. This preparation assigns
existing tetrahedra to three groups; it adds no nodes, changes no reference
positions and makes no physical cut. The example presents one numbering scheme
throughout: dataset wounds 1–6 and remaining pad regions 7–12.
In `fem_phantom_regions`, selecting the smaller domain reduces the active FEM
system. Its interface
against inactive tissue is clamped, so it is a local deformation approximation
rather than a simulation of the entire freely deforming pad. The 2D map retains
its original orientation; only the 3D viewing camera faces the lettering edge.

## SutureBot recorded motion and camera overlays

The current `fem_suturebot` entry point loads the prepared
`tissue_3/2_needle_throw/20250428-215300-489212` recording through
[_suturebot_recording.py](../src/newton_solver_fem/examples/_suturebot_recording.py).
Its 601 source frames depict work on dataset region 1. USD supplies the tool
and camera meshes, static full phantom, RCM fixture, table and endoscope housing.
Prepared link poses are derived
from the recorded Cartesian tool poses and wrist/jaw chain, with the declared
frame-alignment transforms. The example assigns these poses directly to the
Newton state. Tool replay remains kinematic and does not infer measured forces.
The phantom remains static. The replay contains no needle or thread assets and
does not step dynamics or FEM.

Integer source timestamps $\tau_k$ remain intact in the package. Relative
times and the displayed source index are

$$
t_k=10^{-9}(\tau_k-\tau_0),\qquad
k(t)=\max\{k:t_k\le t\}.
$$

Playback holds the last measured frame until the next timestamp. The tool
poses, recorded images and prepared optical poses use that same index; the
runtime does not interpolate motion independently of the video or remap it
through nominal frame rate. The loader checks matching camera/body timestamps,
finite poses and proper rotations. Explicit camera availability masks preserve
missing poses at recording boundaries, including when preparation uses a
camera timing offset. Such views display an unavailable frame rather than
clamping the optical pose to a valid endpoint.

The dataset's two final-left-frame point labels are intersected with the visible
tetrahedral exterior during preparation. For a hit on triangle $(a,b,c)$,
the stored barycentric coordinates give $p=\lambda_a a+\lambda_b b+\lambda_c c$,
with $\sum\lambda=1$. Face-varying USD texture coordinates map each exterior
triangle into its own atlas tile. Rasterizing the original colored mask through
the calibrated rays produces material marks without changing vertices, region
ownership or triangle indices. Visibility testing prevents labels from painting
through a nearer surface. The existing USD texture path supplies both the
viewer and camera renderer. These bindings can follow updated triangle vertices,
but the default replay does not deform them.

[_suturebot_cameras.py](../src/newton_solver_fem/examples/_suturebot_cameras.py)
uses the public Newton `SensorCamera` renderer. Lens inversion and frame
resizing happen under the MSc preparation tree. For a displayed pixel
$(u,v)$ in an image of size $(w,h)$, its prepared ray corresponds to native
pixel coordinates

$$
u_{\rm native}=(u+\tfrac12)\frac Ww-\tfrac12,\qquad
v_{\rm native}=(v+\tfrac12)\frac Hh-\tfrac12.
$$

Here $(W,H)$ is the original camera image size; $K$ and distortion coefficients
remain defined in those original pixels. The package supplies a normalized
inverse-lens direction $r_{\rm CV}(u,v)$ and validity mask. With prepared
camera-to-world pose $(R_k,p_k)$, the world ray is
$x(s)=p_k+sR_k r_{\rm CV}$, $s>0$. Runtime conversion to the renderer's
OpenGL camera axes uses $\mathrm{diag}(1,-1,-1)$ on both camera orientation
and ray direction, preserving this world ray. Invalid lens rays remain absent.

Each view raytraces the same model state shown in the main 3D viewer. Public
model BVH refits update moving bounds; nearest-hit depth provides model-to-model
occlusion. Visible albedo, shaded using the rendered surface normals, is blended
over the matching recorded pixels. Visible depth/normal discontinuities are
outlined in cyan; coplanar boundaries between numbered regions create no edges.
Recorded pixels have no depth
map, so an observed object cannot automatically occlude the model. Users can
disable the overlay to inspect the original frame or change its fill opacity.

Offline scene fitting preserves all native phantom vertices and region geometry.
It combines visible wound crests, yaw-cap landmarks, partial tool masks and the
independently observed negative-Y outer pad silhouette with explicit fixture
constraints. Camera focal lengths, principal points and extrinsics are fitted;
lens distortion remains an inherited, unmeasured prior. The preserved base is
an analytic rectangle whose border differs
from the original rounded CAD by about 1–2 mm. Only one exterior side is visible,
so the fit does not determine or validate every pad corner.

The additional shaft objective uses independently marked, visible outer
cylinder edges in the wrist images. The polished shaft reflects the pink pad;
its narrow central dark reflection is not a silhouette. The finite cylindrical
portion is selected from the retained roll CAD, excluding its milled connector
and articulated head. Its radius is $r=4.37236$ mm and its roll-local axial
interval is $[-556.5921,-9.2324]$ mm. For camera center $c$ in that frame, the
two perspective tangent generators have cross-section unit vectors $u_\pm$
satisfying

$$
\|u_\pm\|=1,\qquad u_\pm^T c_{xy}=r,\qquad
X_\pm(z)=(r u_{\pm,x},r u_{\pm,y},z).
$$

Preparation transforms and clips these finite generators in camera space
before projecting them through the candidate intrinsics and lens distortion.
It freezes a one-to-one assignment of observed left/right edges to the two
projected curves. Each observed pixel contributes its nearest point-to-curve
residual divided by the annotation's 6–9 pixel uncertainty to the robust
`soft_l1` objective, alongside wound, pad-side, tool-landmark and mechanical
prior terms. Visibility endpoints are not assigned invented axial coordinates;
occluded or indistinguishable edges contribute no constraints. The independent
render audit uses the original shaft triangles and full-scene depth ordering,
so it also checks the analytic cylinder approximation.

For a wrist camera rigidly mounted to its own roll link,
$T_{WC}=T_{WR}T_{RC}$ gives $T_{CW}T_{WR}=T_{RC}^{-1}$. Its own-shaft projection
therefore constrains the mount and lens model jointly with the assumed radius;
global arm or table translation cancels. Repeated frames share this same
mount geometry and are correlated observations. Reserved frames check agreement
under different background, lighting and occlusion, not an independent metric
calibration. No reliable long-shaft edge was identified in the stereo views,
and exact PSM2 shaft geometry remains a surrogate assumption. A catalog field
of view is not enforced as a measurement of these particular lenses.

Sparse partial tool masks add reviewed foreground/background pixels and visible
exterior contour fragments. Unlabelled pixels remain ignored; no missing
perimeter is closed or filled. The mesh check uses the retained articulated
triangles, rejects projected internal overlap edges using forward ray occupancy,
and evaluates distance $d$ to the external projected contour. With signed
distance $s$ positive inside the projected tool and annotation disk radius $a$,
the contour, foreground and background residuals are respectively
$d$, $\max(0,a-s)$ and $\max(0,a+s)$. Each is scaled by the observation's pixel
uncertainty before robust fitting. Original-image head contours are separate
from jaw observations, whose geometry and needle-held opening may disagree
with the replay. Exact PSM2 instrument CAD is unavailable; its silhouette
residuals include that surrogate-model discrepancy.

The metric fixture-native frame $F$ is the shared calibration reference, with
native +X along the support normal. Each arm has one fitted rotation $R_i$ and
a constant registration

$$
E_i=\begin{bmatrix}R_i & s a_i-R_i q_i\\0&1\end{bmatrix},\qquad
T^W_i(t)=T^W_F E_i T^{S_i}_i(t+\Delta).
$$

Here $q_i$ is the measured SUJ pivot, $a_i$ its original CAD port center, $s$ the
fitted fixture scale and $\Delta$ the shared image/robot clock offset. This maps
each pivot onto its port exactly while preserving every relative increment of
the recorded motion. No per-frame correction is fitted. Canonical metadata
stores these transforms as `T_world_from_reference` and
`T_reference_from_recorded`. The wrist mount obeys $R_{RC} e_z=e_z$ exactly;
only optical roll, mount translation and intrinsics vary.
The reported axial offset locates the optical center relative to the roll-link
origin at the wrist-pitch pivot, not relative to a housing endpoint. Stereo
intrinsics and extrinsics remain coupled: an image fit with displaced principal
points does not independently establish baseline or physical optical-axis angle.

The unchanged pad underside defines $n^T x=h$. Pad yaw and displacement relative
to the fixture are fitted, with $R^W_F e_x=n$ enforcing parallel normals. The
pad bottom's offset from the source fixture feet is an independent parameter;
the table follows the transformed feet plane. Zero port/parallelism residuals
are enforced modeling constraints, not measured calibration precision. CAD port
midpoint ambiguity is approximately 3 mm. The transparent endoscope housing
follows the stereo optics and uses the recorded 154.98 mm ECM tip-to-RCM length
for scale. That length does not determine the physical optical-to-shaft angle;
the straight-scope convention, 12 mm diameter and table extent are visual
assumptions. The third fixture opening and optical hand-eye are unmeasured.

See [validation](validation.md#suturebot-recorded-placement-replay-2026-11-03)
and packaged reports for fitting errors and frame splits. These are conditional
image-alignment estimates; the tools follow measured motion with fitted frame
corrections, while cameras and phantom registration are estimated. Exact PSM2
CAD and independently measured optical/metric calibration are unavailable.
The known-scene checks in
[test_suturebot_cameras.py](../tests/test_suturebot_cameras.py) verify ray/pose
conventions, asymmetric projection, depth ordering and state updates on CPU
and CUDA. They validate the rendering implementation, not the source fit.

## Legacy SutureBot rigid and deformable contact experiment

[_suturebot_contact_demo.py](../src/newton_solver_fem/examples/_suturebot_contact_demo.py)
retains the earlier approximate rig experiment separately from the recording
replay. It imports the rigid articulation from USD. MuJoCo
advances its joints, free needle and rigid contacts. The existing FEM update
advances one selected wound, retaining fixed cut and underside nodes. Its
reference surface includes the inactive regions so a needle can also rest on
fixed tissue.

[_suturebot_contacts.py](../src/newton_solver_fem/examples/_suturebot_contacts.py)
samples the axes of imported capsule colliders at spacing no greater than
their radius. For sample centre $c$, radius $r$, signed distance $d$ to the
current triangle surface, outward normal $n$, and relative contact velocity
$v_r$, the penalty is

$$
\delta=\max(0,r-d),\quad
f_n=k\delta-c_d\min(v_r\cdot n,0),\quad
f=w\left(f_n n-\mu f_n\frac{v_t}{\max(\|v_t\|,0.001\,\mathrm{m/s})}\right).
$$

The force is applied only for positive penetration. Here $v_t$ is tangential
relative velocity, and sample weights $w$ sum to one per body. Each contact
adds $f$ and its moment about the rigid centre of mass to MuJoCo, and distributes
$-f$ to the triangle's active FEM vertices using barycentric weights. Inactive
or clamped vertices transmit their reaction to the fixed support. Rigid point
velocity includes both centre-of-mass velocity and angular velocity; tissue
velocity is interpolated from the same triangle.

Each substep evaluates contact from the current states, then advances FEM and
MuJoCo with those forces. This is explicit partitioned penalty coupling, with
one force exchange per substep, not a monolithic constrained solve. Closest
triangle queries and sampled capsule envelopes are approximate; they do not
guarantee intersection freedom. This experiment has no needle puncture, thread,
topology changes or differentiable rigid/FEM coupling. Numerical parameters
are demonstration choices, not fitted material or robot dynamics.

The legacy press demonstration uses scratch-state forward kinematics to solve
for RCM yaw, pitch and insertion at a free upward-facing tissue triangle.
Smooth joint targets approach, press, hold and retract a closed PSM1 jaw; only
MuJoCo drives change the simulated tool poses. A settled pre-press state is the
displacement reference. Contact diagnostics sum the barycentric force fraction
delivered to free FEM nodes separately for each rigid body, excluding clamped
and inactive support. Validation requires jaw-to-free-tissue force, deformation,
unloading and recovery, so needle-only or fixed-pad contact cannot satisfy it.
This trajectory is a scripted contact experiment, not a dataset replay or a
learned suturing controller.
Its reproducible entry point is
[validate_suturebot_press.py](../tools/validate_suturebot_press.py); it does not
drive the `fem_suturebot` recording or its camera overlays.
