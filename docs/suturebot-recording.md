# SutureBot recording replay

`fem_suturebot` (menu item 3) opens a second terminal menu with four distinct bundled
recordings. Compatibility aliases are omitted from both recording menus; the
former choice 5 uses the same assets as default choice 1. Choose by number or
sample name; Enter selects `tissue_3/2_needle_throw/20250428-215300-489212`, working on region 1.
Each choice shows its region, source-frame range, playback duration, point-label
availability and remaining alignment issues. Enter `q` to quit before the viewer opens.
It replays the measured tool poses and wrist joints over the original phantom,
with the RCM fixture, supporting table and transparent endoscope housing in the
same fitted scene. Four small model/video overlays show the left/right endoscope
and PSM1/PSM2 wrist cameras. Dataset point labels are UV-textured onto the phantom
surface. Needle and thread estimates have been removed from the prepared examples
and their runtime controls. The source videos are unchanged. This is a kinematic
scene replay with a static phantom.

```bash
# Choose fem_suturebot, then choose a recording:
uv run --locked --extra examples python -m newton_solver_fem.examples
# Open the recording menu directly:
uv run --locked --extra examples python -m newton_solver_fem.examples fem_suturebot
# List recordings or launch a named recording directly:
uv run --locked --extra examples python -m newton_solver_fem.examples fem_suturebot --list-samples
uv run --locked --extra examples python -m newton_solver_fem.examples fem_suturebot --sample tissue3_222119_approach
# Load another independently prepared package:
uv run --locked --extra examples python -m newton_solver_fem.examples fem_suturebot --sample /path/to/recording.json
# Optional playback controls:
uv run --locked --extra examples python -m newton_solver_fem.examples fem_suturebot --loop --playback-speed 0.5
```

The panel provides recording selection, play/pause, an exact source-frame slider
and restart. `--sample` accepts a bundled sample name or a `recording.json` path.
The same recording menu appears with `./docker/run.sh fem_suturebot`.
Every bundled recording uses separate USD assets for the PSM1 Large Needle
Driver and the revised PSM2 DeBakey jaw estimate. The DeBakey has straight outer
edges, rounded tips and an inward serrated gripping face with a compact curved
neck recess. Both shafts use the revised approximately 8.36 mm diameter.
The `tissue3_215300_mesh_preview` compatibility name remains available through
`--sample` and uses the same tool assets as the default recording:

```bash
./docker/run.sh fem_suturebot --sample tissue3_215300_mesh_preview
```

Each recording contains its own portable copy of the two tool assets. The mesh
update preserves its recorded motion, opening angles, cameras and phantom. The DeBakey shape is an estimate refined against the reference
photo; four-view calibration remains unfinished. The four recording choices and
default selection remain available.

Redirected-input, headless, test, demo and benchmark runs keep the default sample
without prompting; `--sample` always selects directly. Use `--choose-sample` to
request the picker explicitly, including before a headless run.
The catalog contains four bundled recordings; eight additional source recordings
remain under preparation and review in the workbench:

| Sample | Source episode | Wound | Frames | Availability |
| --- | --- | --- | --- | --- |
| `tissue3_215300` | `tissue_3/2_needle_throw/20250428-215300-489212` | 1 | 601 | Bundled alignment estimate |
| `tissue3_222119_approach` | `tissue_3/2_needle_throw_recovery/20250428-222119-449047_recovery` | 1 | 575 retained; replay 5–141 | Estimated alignment; green labels |
| `tissue3_224616_approach` | `tissue_3/2_needle_throw_recovery/20250428-224616-539134_recovery` | 1 | 461 retained; replay 4–106 | Region 1 placement and green labels corrected; alignment remains estimated |
| `tissue3_215248_pickup` | `tissue_3/1_needle_pickup/20250428-215248-974340` | 1 | 282 retained; replay 4–281 | Recorded pickup tool motion; pad alignment unvalidated; no labels |
| `tissue3_212958` | `tissue_3/2_needle_throw/20250428-212958-618853` | 1 | 668 | MSc preparation; alignment review failed |
| `tissue4_211216` | `tissue_4/2_needle_throw/20250429-211216-730508` | 1 | 640 | MSc preparation; alignment review failed |
| `tissue2_111109` | `tissue_2/1_needle_pickup/20250424-111109-544182` | 1 | 306 | MSc candidate; native tool/pad alignment failed |
| `tissue5_102359` | `tissue_5/2_needle_throw/20250430-102359-757268` | 1 | 598 | Earlier alignment failed; reviewed tracking retry has insufficient PSM1 coverage |
| `tissue6_221332` | `tissue_6/2_needle_throw/20250501-221332-135591` | 1 | 547 | Source reviewed; insufficient PSM1 tracking coverage |
| `tissue7_200743` | `tissue_7/2_needle_throw/20250502-200743-038955` | 1 | 603 | Earlier alignment failed; reviewed tracking retry has insufficient PSM1 coverage |
| `tissue8_220859` | `tissue_8/2_needle_throw_recovery/20250505-220859-555095_recovery` | 1 | 623 | MSc candidate; native tool/pad alignment failed |
| `tissue9_100142` | `tissue_9/2_needle_throw_recovery/20250506-100142-221421_recovery` | 1 | 655 | Source reviewed; PSM2 tracks, but independent PSM1 image support fails |

The tissue session number is not a wound number. The eight additional source clips
show the same long region-1 incision; their preparation configs retain that
source-image review before fitting. This set does not establish recognition or
calibration accuracy on wounds 2–6. Additional placements are conditional
transfers of the reference calibration, with per-episode constant arm-base
corrections and image-fit reports. Inherited lens, fixture-scale, timing and
depth assumptions remain explicit in each preparation's `automatic_fit.json`.
Prepared candidates have source-motion, UV and rendering checks, but their camera
overlays show unresolved placement errors. Other clips stop at insufficient or
inconsistent image evidence. None is installed as a validated example.

The movable four-view window blends actual USD model surfaces onto the matching
recorded frames, with cyan visible edges, adjustable fill opacity and a
source-image-only toggle. Each view uses its prepared inverse-lens ray grid and
optical pose, including moving wrist cameras. All views and tool bodies share
one original frame index and int64 timestamp. Camera/image resizing is performed
only during preparation; no calibration runs inside the example.

The default recording's 601 samples span approximately 20 seconds. Motion uses the measured
Cartesian pose at the jaw pivot and the dVRK wrist/jaw chain. It is kinematic
replay for checking placement, not a dynamics controller or force estimate.
The fitted image/robot time offset is approximately -107 ms. Frames 0–3 fall
outside recorded robot-pose support and remain marked unavailable; playback
and the slider cover original frames 4–600 without inventing boundary motion.
The phantom keeps its original native coordinates and region geometry. Camera
mounts, camera intrinsics and constant arm-base registrations are estimated
from visible wound crests, tool landmarks, partial tool masks, shaft edges and
one independently identified outer pad side. Each arm uses one rigid base
transform for the entire recording, with its SUJ pivot tied exactly to a fixture
port. Wrist optical axes are constrained parallel to their shafts. The pad
normal is parallel to the fixture support normal, while its displacement above
the fixture feet is fitted independently. The feet define the table plane.
The endoscope housing follows the stereo optics; its straight optical-to-shaft
convention, diameter and table dimensions are illustrative. The recorded ECM
length does not establish the physical endoscope's optical angle. These are
scene constraints, not measured hand-eye calibration.

The published [alignment report](../src/newton_solver_fem/examples/assets/suturebot/recordings/tissue3_215300/alignment.json)
and [constrained fit report](../src/newton_solver_fem/examples/assets/suturebot/recordings/tissue3_215300/constrained_fit_metrics.json)
report image residuals, fitted optics and mechanical constraints. See
[validation](validation.md#suturebot-recorded-placement-replay-2026-11-03) for
the numerical results and their limitations. Current fitting excludes frames
300 and 525 from every new image objective; earlier diagnostic exposure means
these are reserved feature checks, not a blind whole-pipeline benchmark.
Endoscope images are 960×540 and wrist images 640×480. Only one pad side is
observable: the other sides, corners and third fixture opening are not
independently registered. The preserved analytic base differs from the rounded
source CAD border by about 1–2 mm. Exact PSM2 instrument CAD is unavailable, so
its displayed geometry uses a documented surrogate. These residuals do not
establish absolute metric accuracy.

The shaft observations use short visible outer edges in the wrist images;
the polished cylinder reflects the pink pad, while its dark central streak
is not a silhouette. Several edges are too blurred or occluded to annotate,
including PSM1 at frame 300 and the long shafts in the stereo views. A camera
viewing its own shaft constrains its mount and optics, so moving the whole
tool or table cannot fix that image mismatch. Repeated views do not supply
independent metric calibration.

The example's data is included in
[`examples/assets/suturebot/recordings/tissue3_215300/`](../src/newton_solver_fem/examples/assets/suturebot/recordings/tissue3_215300/README.md).
This directory is self-contained: its assets use relative paths, and it can be
copied to another checkout or loaded from an installed wheel. Source dataset
identifiers and hashes are portable provenance records. Preparation code and
intermediate outputs remain separately in `MSc/SutureBot-workbench`; the
original recording is unchanged.

To prepare additional motions, run the batch command from
`MSc/SutureBot-workbench` with its preparation environment and the original
dataset available:

```bash
python -m suturebot.simulation.prepare_recording_batch \
  --config config/recording_batch/four_holdouts.json --max-nfev 160 \
  --point-labels \
  --render-python /path/to/newton-solver-fem/.venv/bin/python
```

The pipeline extracts source evidence, fits constant transforms, verifies motion
and input preservation, prepares UV labels and renders review views. It prepares
tools, cameras, rig and pad only; caches with needle or thread extensions are
rebuilt. Publication rejects bundles containing those extensions. Reprepared
packages need a fresh four-view review. Publication uses `--publish-root` and
requires the applicable quality checks. Every new recording needs a
source-defined playback scope and passing image checks in all four cameras,
including reserved observations, reviewed source/model overlays and unchanged
geometry/motion/camera hashes. Runtime success alone cannot publish a sample.
`--publish-review-required` permits an explicitly labeled absolute-calibration
estimate after those checks; it cannot waive a failed or missing camera review.
Only the original `tissue3_215300` reference without automatic-fit metadata retains
its documented legacy estimate status. The input/script fingerprint permits reuse
of unchanged preparation;
`--force` recomputes the fit. All preparation remains under MSc. The example
only loads the published data.

Point annotations come from the dataset's existing `clicked_point.csv` and
`clicked_point_mask.png`. These are two targets in the final original left
endoscope frame, not a time series. Preparation intersects those camera rays
with the visible phantom exterior, records the region, triangle and barycentric
coordinates, and bakes the mask coverage into a bright green RGB `(0, 255, 0)`
USD UV texture. Source click coordinates and the original mask are preserved. The dots
remain attached to that material surface during playback. The replay surface
is static; this does not add FEM stepping.

- `recording.json`: sample identity, relative asset paths, camera conventions,
  timing, alignment status and provenance.
- `scene.usda` and referenced USDs: tools, cameras, fixture, table, endoscope
  housing and unchanged phantom surface geometry in a common world frame.
- `motion.npz`: exact source timestamps and body transforms in declared order.
- `calibration.json`: fitted transforms, camera intrinsics/distortion, source
  evidence, held-out image residuals and explicit accuracy limitations.
- `constrained_fit_metrics.json`: optical parameters, per-feature residuals,
  exact RCM/parallelism checks and the independently fitted pad displacement.
- `camera_poses.npz`: four optical poses and explicit availability per frame.
- Prepared camera rays and indexed JPEGs: original camera geometry at the small
  overlay resolution, with no runtime dependency on the source-data checkout.
- `point_labels.json`, `point_labels_source/`, `textures/`: original point labels,
  their mesh bindings and the baked UV atlas.
## Bounded approach samples

Two additional recordings are bundled as short approaches before visible
tissue deformation: `tissue3_222119_approach` uses region 1 and
`tissue3_224616_approach` is also region 1, confirmed by the user for the entire
recording. Its earlier region-3 interpretation is superseded. Publication and portable runtime checks are recorded in
[the approach validation report](validation/suturebot-approaches.json).

| Sample | Region | Source recording under `tissue_3/2_needle_throw_recovery/` | Source frames, inclusive | Playback duration |
| --- | ---: | --- | --- | ---: |
| `tissue3_222119_approach` | 1 | `20250428-222119-449047_recovery` | 5–141 | 4.527 s |
| `tissue3_224616_approach` | 1 | `20250428-224616-539134_recovery` | 4–106 | 3.401 s |

The source videos define these bounds independently of the fit residuals.
Recorded frame numbers, timestamps and complete motion arrays are retained;
the frame slider and playback use only the declared approach. The later motion
is available for diagnostics but is not validated as a static-pad replay.

These samples contain tools, the fixture, estimated cameras and pad placement,
and the existing dataset point labels rendered in green on the mesh.
Four-camera image thresholds and source-motion preservation are checked
within each approach. Calibration remains an estimate with `accepted: false`;
passing those scoped checks does not establish absolute dimensions, exact
camera calibration or pixel-exact tool and wound geometry. PSM2 uses surrogate
instrument geometry, and local jaw and groove-bank differences remain visible.
The table, pad depth and fixture scale inherit the stated calibration assumptions.
Recording 3 retains its fitted first-frame PSM1 shaft and head alignment, with
residual errors in other views. This is a partial replay estimate, not a fully
aligned reconstruction.
Recording 4's source identity is now confirmed as region 1, including both
visible incision fragments; nearby feature A is tentatively region 2. The
obsolete region-3 placement has been replaced by one rigid pad pose using the
region-1 incision and the identified outer edge. Both green labels were rebound
to Region01. Ten synchronized frames were rendered in all four native views;
reviewed cloth-background points are no longer covered by the phantom. Reserved
wrist border errors fell from 226/366 to 13/16 pixels, so visible residuals
remain and full alignment is still unvalidated. The twelve original meshes,
their geometry and canonical IDs are preserved. Current evidence is in
`pad_edge_alignment.json` and `audit/region1_pad_fit/`; the previous region-3
reports remain historical evidence.

Select them in the recording panel or run:

```bash
uv run --locked --extra examples python -m newton_solver_fem.examples fem_suturebot --sample tissue3_222119_approach
uv run --locked --extra examples python -m newton_solver_fem.examples fem_suturebot --sample tissue3_224616_approach
```

Preparation lives in `MSc/SutureBot-workbench`. Frozen source-image annotations
and phase definitions are explicit inputs; this is not an automatic six-region
classifier. The region-1 batch fitter estimates session mounts and a constant
pad translation from all four views and rejects other regions before fitting
or reusing a cache. The four corrected `224616` batch entries declare region 1
and fresh output directories under `outputs/simulation/review_224616_region1_correction/`.
Their earlier region-3 results are retained as history. New recordings must pass
the pad-boundary, incision, tool and camera checks before publication. The
existing example-4 correction retains its explicit estimate status. The Newton
example only loads prepared data.

## Full supported pickup replay

`tissue3_215248_pickup` covers source frames 4–281 (9.226 s) of
`tissue_3/1_needle_pickup/20250428-215248-974340`. It preserves the complete
supported pickup motion before tissue interaction, with the same conditional
camera/tool/pad calibration policy as the approaches. Its source recording has
no clicked-point labels, so it contains no UV markers. All 278 supported source
frames retain their recorded tool motion. Pad alignment still differs from the
videos; the camera calibration remains an estimate. Earlier pose-replay results
are recorded in the [structured validation report](validation/suturebot-approaches.json).

The sample is bundled. Select `tissue3_215248_pickup` in the recording panel or run:

```bash
uv run --locked --extra examples python -m newton_solver_fem.examples fem_suturebot --sample tissue3_215248_pickup
```

The reproducible workbench configuration is
`config/recording_batch/pickup_215248_reviewed.json`. It uses source-reviewed
stereo incision seeds because dataset click labels are absent, and the frozen
wrist observations supply independent tool and pad checks. No point labels are
fabricated.

The earlier approximate MuJoCo/FEM press experiment is retained separately in
`_suturebot_contact_demo.py` and `tools/validate_suturebot_press.py` for contact
validation. It does not drive the recording replay or its alignment overlays.

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

## SutureBot data and implementation scope

Source: Haworth et al., [SutureBot (2025)](https://suturebot.github.io/static/SutureBot_NeurIPS_2025.pdf), Fig. 2 and §2.1.

[fem_suturebot](../src/newton_solver_fem/examples/example_fem_suturebot.py) replays the recorded tool motion for `tissue_3/2_needle_throw/20250428-215300-489212` over the original static phantom, with four model/source-video overlays. Prepared USD geometry, link poses, optical poses and lens rays come from the separate MSc preparation pipeline. Region 1 identifies the training wound; the regional FEM example presents wounds 1–6 and other areas 7–12. Camera/phantom registration remains a fitted estimate. Tool motion is an alignment replay. Needle and thread reconstruction are excluded from the prepared replay; it does not reproduce the paper's policy or suturing results.

| Source concept | Current implementation |
| --- | --- |
| Fig. 2: six numbered pad wounds | [fem_phantom_regions](../src/newton_solver_fem/examples/example_fem_phantom_regions.py) uses one visible numbering scheme, 1–12. Offline preparation partitions one connected original feature into wound 5 with its thin upper rim, remainder 8 including the lower-left tail, and upper slit 12; the other nine original payloads and native tetrahedral coordinates are preserved. |
| §2.1: recorded tool motion and four camera views | [Recording](../src/newton_solver_fem/examples/_suturebot_recording.py) reads a structured JSON/USD/NPZ package and preserves exact source timestamps. The replay selects one source index for tool poses, camera poses and all four images, holding each sample until its successor. Derived link poses are assigned kinematically; no dynamics or force measurement is inferred. |
| Checking model placement against recorded pixels | [FourCameraOverlay](../src/newton_solver_fem/examples/_suturebot_cameras.py) renders actual Newton model geometry through prepared inverse-lens rays and camera-to-world poses, then blends visible surfaces and outlines onto indexed source frames. This rendering/registration workflow is an implementation addition, not a claimed algorithm or calibration result from the paper. Numerical projection tests do not establish accuracy of the fitted recording alignment. |
| Visible tool-shaft alignment | Offline preparation compares independently observed wrist-image cylinder edges with finite perspective tangents of the retained shaft CAD, and checks the result against actual triangle renders. The [shaft objective](#suturebot-recorded-motion-and-camera-overlays) estimates mount/lens alignment; an own-shaft view cannot independently determine global tool/table height. This is an implementation addition, not a calibration procedure or measured optical specification supplied by the paper. |
| Fig. 2: pad, RCM fixture and endoscope arrangement | Offline preparation ties each constant arm-base calibration to a fixture port, constrains wrist optical axes parallel to shafts and the pad normal parallel to fixture support, and fits the pad's independent displacement. The source fixture feet define the table plane. A transparent endoscope housing follows the stereo optics; its optical-to-shaft angle is unmeasured. These scene-registration constraints are implementation choices, not measured hand-eye calibration supplied by the paper. |
| Dataset point targets | Existing final-frame clicked points bind to the unchanged tet exterior through barycentric coordinates and a USD UV atlas. The bindings and static replay surface preserve the supplied material targets. |
| Earlier approximate physical rig demonstration | [_suturebot_contact_demo.py](../src/newton_solver_fem/examples/_suturebot_contact_demo.py) retains the MuJoCo tools/free needle and sampled two-way FEM contact experiment. Available instrument CAD includes a documented DeBakey surrogate. The [legacy contact method](#legacy-suturebot-rigid-and-deformable-contact-experiment) and press validation are separate from the default recording replay. |

All fitting, source-image decoding and USD preparation remain under the MSc
workbench; the solver package consumes published assets. The package's alignment
report records the status of fitted camera mounts, phantom registration and
timing; [validation](validation.md#suturebot-recorded-placement-replay-2026-11-03)
reports the numerical image errors. Native phantom geometry and recorded
relative tool motion are preserved. Only one pad side is observed;
other sides/corners and the third fixture opening are not independently
registered. PSM2 uses surrogate instrument CAD, and the endoscope housing/table
dimensions include visual assumptions. Measured tool motion therefore does not
make the fitted camera/phantom poses measured calibration. No absolute metric
accuracy or reproduction of the paper's results is claimed.
