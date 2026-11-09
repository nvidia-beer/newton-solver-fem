# Numerical validation

These retained measurements were collected on 2026-10-28 with Newton 1.7.0,
Warp 1.17.0, Python 3.12, Linux ARM64 and NVIDIA GB10. They describe those
configurations and hardware; they are not fresh timing measurements from a
documentation build. Rendering and cold compilation are excluded unless stated.

## Forward simulation

| Scene | Physical configuration | Measured result |
| --- | --- | --- |
| Baymax cutting | 8,965 nodes, 38,730 tets; 60 Hz; 4 substeps; 30 CG iterations; 300 frames; three cuts | Ordinary median 11.783 ms, p95 13.460 ms; cut-frame median 18.721 ms, max 21.424 ms |
| Six-object showcase | 10,030 particles, 39,800 Tet4, 385 Hex8, 54,392 edge springs; 10 substeps; 50 iterations | 46.758 ms/frame over 300 warmed frames; 0.356× realtime |
| Differentiable bear inference | 1,986 particles, 5,354 tets, 9,323 edge springs; 80 substeps; 300 frames | Median 26.836 ms, p95 28.330 ms; 0.619× realtime |

Cutting retained captured storage, committed all three incisions, and ended
with 40,315 fragments/146,350 active BSR blocks in 1,720,180 reserved blocks.
Physical mass remained 7.51521 kg to float32 accumulation accuracy; minimum
Jacobian was approximately 0.329. Every frame was finite with positive element
volume. Ordinary cutting frames met 60 Hz; editing frames did not. The full
showcase and checked bear were slower than realtime. These are physical stepping
measurements, not video playback rates or guarantees of all-scene stability.

A separate unchanged-scene before/after differentiation measurement gave
48.308/48.342 ms per showcase frame. The 0.07% difference is smaller than a
meaningful effect established by single samples; ordinary simulation records
no tape. See `validation/showcase-{before,after}-differentiation.json`.

## Derivatives and optimization

The CPU suite ran 62 tests: 58 passed, four CUDA-only skips (187.537 s).
The CUDA suite passed all 62 (7.522 s). Coverage includes Tet4/Hex8 state/force
VJPs, Tet activation/controller gradients, fixed contact branches, repeated and
clamped stretches, owned snapshots, revision guards, checkpoint/full-Tape
agreement, sparse cut updates, rollback, holds and surface growth. A 120-frame
CPU Tet/Hex fixture passed. No derivative through cuts is claimed.

One genuine Adam update on the 300-frame bear trajectory reduced the objective
227.444229 → 197.617142 (gradient norm 15.003851). Forward/transpose residual guards
passed, updated frame checkpoints had minimum Jacobian 0.786158, and 300 fresh
inference frames remained finite. Initial forward including setup took 8.229 s;
backward with recomputation/setup took 18.207 s; updated forward took 8.044 s.
Frame checkpoints used 14,346,864 bytes plus an 81-state recomputation segment;
that is not total solver memory. Training adds backward/recomputation cost.

A small force fit reduced loss 0.372905731 → 1.68e-12 in four accepted updates,
recovering -0.200000033 N for a -0.2 N target. Its fixed support is an explicit
differentiable projection after each step. Historical results are retained in
`validation/force-fit-differentiable.json`, `validation/bear.json`,
`validation/cut.json`, and `validation/showcase.json`.

## Reproduce checks

```bash
uv sync --locked --extra examples
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -m unittest discover -s tests -v
NEWTON_TEST_DEVICE=cuda:0 uv run --locked python -m unittest discover -s tests -v
uv run --locked python tests/benchmark_showcase.py --device cuda:0 --warmup 30 --frames 300
uv run --locked python tests/benchmark_cut.py --frames 300 --warmup 10 --validate_every_frame --output /tmp/fem-cut.json
uv run --locked python -m newton_solver_fem.examples diffsim_bear --device cuda:0 --train-iters 1 --sim-steps 300 --viewer null --num-frames 300 --save-weights /tmp/bear-policy.npz --test
```

GPU reductions and contact switches can change long trajectories. Timing
comparisons require unchanged scenes, budgets, synchronized warm execution and
an otherwise idle device. Refer to [method](method.md) for the gradient scope.

## Cleanup verification (2026-10-28)

The final standalone suite ran with `python -m unittest discover -s tests -v`:

| Device | Tests | Result | Wall time |
| --- | --- | --- | --- |
| CPU | 72 | Passed; four CUDA-only skips | 211.386 s |
| CUDA (`NEWTON_TEST_DEVICE=cuda:0`) | 72 | All passed | 34.598 s |

The CUDA run used `WARP_CACHE_PATH=/tmp/newton-solvers-cleanup-warp`; test time
includes compilation and setup, so it is not a solver performance measurement.
Lint and formatting passed for all 77 Python files. Wheel and source-archive
builds succeeded. The Docker image built and passed imports, example discovery
and native USD loading in an isolated container with no network or host mounts.
The Docker launcher also passed a two-frame CPU headless example with `--test`.
These checks validate the cleanup and retained behavior; they do not establish
a real-time guarantee.

## SutureBot wound regions (2026-11-03)

The final catalog uses one public numbering scheme, 1–12. Only the original
connected volume containing regions 5, 8 and 12 is subdivided. Nine other
original payloads remain byte-identical, and the full mesh retains its original
coordinates and connectivity. The lower-left wrap remains in region 8; only
the thin upper rim joins region 5. The prepared domains contain 18,790
tetrahedra in region 5 and 9,801 in region 12.

The complete 140-test suite passed with `NEWTON_TEST_DEVICE=cuda:0`. The CPU
run passed 135 tests and skipped five CUDA-only checks. Additional 30-frame
mouse-grab checks of regions 5 and 12 produced measurable deformation with
finite states, positive Jacobians and exactly fixed cut/underside anchors.

The retained [regional replay report](validation/phantom-regions.json) covers
all 540 frames of the video workload on NVIDIA GB10: wound 4, wound 6,
no active regions, then wound 4 again. Their actual FEM sizes were
11,795, 13,909, zero and 11,795 tetrahedra. The null viewer replays the same
physical ray picks and spring targets as the visible mouse controls.

Every replay frame had finite state, positive tetrahedron Jacobians and exactly
stationary cut/underside anchors. The minimum Jacobian was 0.924977. The 3 mm
grab-target motions produced maximum nodal displacements of approximately
0.811 mm and 1.258 mm in wounds 4 and 6. These are checks of this scripted
demonstration, not fitted tissue parameters or guarantees for arbitrary pulls.

The nine-second [actual GPU recording](media/fem_phantom_regions.mp4) has 540
decoded frames at 1280 × 720 and 60 fps. It shows clicks on the 2D region map, active
tetrahedron counts and mouse-grab markers; its [capture metadata](media/fem_phantom_regions.json)
retains the selection and deformation measurements. The movie's playback rate
is not a live performance result.

```bash
uv run --locked --extra examples python tools/validate_phantom_regions.py --output /tmp/phantom-regions.json
uv run --locked --extra examples python tools/record_examples.py fem_phantom_regions --width 1280 --height 720
```

## Legacy SutureBot rigid mechanism and tissue contact (2026-11-03)

These retained results describe `_suturebot_contact_demo.py`, the earlier
approximate physics experiment. Menu item `fem_suturebot` now runs the separate
recording replay described below; its current video does not show this press.

All four [rig/contact tests](../tests/test_suturebot_example.py) passed with
Newton device `cpu` and again with `cuda:0`. These tests use MuJoCo's CPU backend
for rigid dynamics. They check force and torque balance for a transformed body,
driven RCM and jaw motion with a fixed fixture, needle settling on the pad,
rejection of invalid control targets without partial updates, and an active
MuJoCo contact constraint between a jaw and the free needle. The jaw/needle
test deliberately places the needle at a jaw to verify contact response; it
does not establish a successful pickup or sustained grasp.

A [540-frame press-and-release report](validation/suturebot-press.json)
on `cuda:0` exercises the earlier scripted `--demo` trajectory,
with a driven jaw pressing the training-wound surface and retracting.
The maximum sampled PSM1 jaw force delivered specifically to free FEM nodes
was 0.02701 N, with up to eight active contact samples. Jaw/active-tissue contact
was observed in 147 press/hold frames. The largest nodal displacement relative
to the settled pre-press state was 0.5634 mm; after retraction and recovery it
was 0.0807 mm (about 86% recovery). The maximum RCM line-to-pivot error over the
sequence was approximately `1.15e-8 m`. Every frame was checked for finite
states, positive tetrahedral volumes and exactly fixed boundary nodes.
These measurements demonstrate local contact deformation and
recovery with the chosen demonstration parameters. They do not validate tissue
material properties, needle puncture, suture thread, knot tying or a suturing
policy, and they contain no live performance measurement.

The [runtime press tests](../tests/test_suturebot_press.py) check the ordinary
demo path, motion of the three free vertices on its target triangle, unloading
after retraction, elastic recovery, and plan reset on region changes. A separate
empty-selection case lets the needle contact fixed tissue and requires zero
active-tissue force/counts, preventing fixed-pad contacts from passing as FEM
interaction. The contact kernel tests also cover partially and fully fixed
triangles without changing physical force/torque conservation.
The complete suite, including these regressions, passed all 142 tests with
`NEWTON_TEST_DEVICE=cuda:0`. The earlier nine-second recording independently
checks the same press/release criteria and measured zero final jaw load on
active tissue.
The additional full 540-frame CPU press run was terminated with exit 143 before
completion, without a test traceback; it is not counted as a CPU pass. The
CPU contact/rig tests and the empty-selection contact regression passed.

```bash
uv run --locked --extra examples python -m unittest discover -s tests -p test_suturebot_example.py -v
NEWTON_TEST_DEVICE=cuda:0 uv run --locked --extra examples python -m unittest discover -s tests -p test_suturebot_example.py -v
uv run --locked --extra examples python tools/validate_suturebot_press.py --device cuda:0 --output /tmp/suturebot-press.json
```

## SutureBot example scope update (2026-11-09)

Needle and thread reconstruction have been removed from the four prepared
recording bundles, their compatibility alias and the example's runtime controls.
The recordings retain their tools, cameras, fixture, table, phantom and available
UV point labels. Source recordings and reconstruction research remain in the
MSc workbench. Preparation now rebuilds caches containing these extensions and
publication rejects them, retaining the four-camera image-validation requirement.
This removal does not change the existing camera-alignment estimates or promote
any failed candidate. The selector lists four distinct recordings; the former
fifth choice remains a direct-name compatibility alias of the default.

Validation of the scoped check-in: the complete staged FEM suite passes
**175 tests on CUDA** (172.354 s). The built wheel passes 21 replay and
selector tests on CPU. Ruff checks and formatting pass. The separate
preparation/publication checks pass 46 tests and 16 subtests. Bundle checks resolve all 27 USD layers locally
and verify that all 8,024 retained nonmetadata payloads are byte-identical to
the pre-removal copies. The refreshed 1,202-frame, 1280 × 720 demo renders all
597 supported source frames with coverage in each of the four camera overlays.
These are replay and packaging checks; camera alignment remains unvalidated.

The dated reports below describe earlier snapshots. Their needle/thread assets,
shape counts and runtime behavior are historical evidence, not the current
example's contents. Original payloads removed from the FEM bundles are archived
under `MSc/SutureBot-workbench/outputs/simulation/example_feature_removal_20261109/`.

## SutureBot recorded placement replay (2026-11-03)

The [prepared replay report](validation/suturebot-recording.json) covers
`tissue_3/2_needle_throw/20250428-215300-489212`, region 1. Runtime and packaging
checks pass; the camera, pad and needle alignment remains an unaccepted visual
estimate. The package retains
all 601 source timestamps exactly. The fitted image-to-robot offset is
`-0.1069642378 s`; frames 0–3 lack pose support and are excluded from playback
and timed tool fitting. The 597 supported frames are 4–600. Wrist optical
poses equal the registered roll-link poses multiplied by their constant
mount transforms. The render bodies and camera trajectories share one clock.

The current default translates only the pad by **(+0.2584, +0.0775, +1.0083) mm**
in world coordinates, retaining its orientation, regional geometry and material
labels. Tools, cameras, fixture, table and the original needle grasp stay fixed.
At source frame 127, the original needle's rendered sharp tip is approximately
0.075 mm from the translated surface and 0.461 mm from the insertion marker.
This is a conditional placement adjustment, not confirmed recorded contact.
The two fixed material labels project 9.21 and 8.63 pixels from their original
source clicks; they were not rebound to conceal that discrepancy. Aggregate
pad fitting/reserved residuals change from 3.61/8.13 to 6.38/8.80 pixels.
The original calibration tables and clearance audits below are retained as
**pre-translation evidence**; current placement and preservation checks are in
the bundled `pad_placement.json` and the replay report.

The core replay USD imports 14 kinematic bodies and 27 visual shapes. With the
separate rigid needle and its collision support, the runtime imports 15 bodies
and 61 shapes. All twelve
phantom reference-position arrays and exterior triangle indices match the
prepared native source, with no repartition or coordinate change. Original
dataset and phantom input hashes remain unchanged. The phantom remains static,
with no tissue puncture or force inference. The added needle has separate
rigid-body dynamics, described below.

The constrained estimate includes visible wound crests, the unchanged negative-Y
base side, tool landmarks, observed shaft edges and sparse tool masks. Each arm
has one fitted rigid registration tied exactly to its RCM port; no per-frame
pose correction is allowed. Wrist optical axes are parallel to their shafts.
The pad normal is parallel to the fixture support, with a free displacement
relative to its feet. Camera focal lengths and principal points are fitted;
distortion retains its earlier estimated prior. The packaged
[constrained report](../src/newton_solver_fem/examples/assets/suturebot/recordings/tissue3_215300/constrained_fit_metrics.json)
records each feature's fitting and reserved residuals in original video pixels,
plus the optical parameters and mechanical checks.

The local optimizer stopped on `xtol` after 30 evaluations, with no active
bounds. Its reported first-order optimality was approximately 357; this is not
a stationary or globally optimal solution. Wrist optical centers are fitted
47.59/50.24 mm behind their wrist-pitch pivot origins, with radial offsets
6.69/6.08 mm and zero axis tilt (PSM1/PSM2). The pad underside is 17.92 mm above
the fixture feet/table plane. These are conditional estimates. The estimated
stereo baseline is 7.60 mm with an 8.73° relative frame rotation (8.64° between
forward axes); displaced
principal points and coupled intrinsics/extrinsics prevent interpreting them
as uniquely measured optical parameters.

| Projected feature | Fitting RMS (px) | Reserved RMS (px) |
| --- | ---: | ---: |
| Shaft, PSM1 wrist | 5.23 | 2.39 |
| Shaft, PSM2 wrist | 2.09 | 2.61 |
| Pad side, PSM1 wrist | 1.31 | 9.09 |
| Pad side, PSM2 wrist | 1.74 | 1.99 |
| Wound, left endoscope | 5.78 | 9.04 |
| Wound, right endoscope | 5.98 | 8.34 |
| Wound, PSM1 wrist | 4.58 | 13.88 |
| Wound, PSM2 wrist | 5.16 | 11.55 |

Only one physical pad side is observable; opposite edges and corners are not
validated. Wound annotations have 7–10 pixel uncertainty, boundary annotations
5–6 pixels, and shaft annotations 6–9 pixels. Frames 300 and 525 are excluded
from all image objectives of this constrained stage. Some earlier initialization
and diagnostic stages exposed those images, so the split is a reserved-feature
check rather than a blind whole-pipeline benchmark.

The historical lip-only estimate had reserved pad-edge RMS of 58.51/88.05 pixels
for PSM1/PSM2. A later soft fixture fit reduced those values to 9.79/2.68 pixels,
but allowed wrist-axis tilt and required coincident pad/fixture support heights.
Those results and its near-contact clearance values belong to the preserved
baseline; they do not describe the current hard-parallel, free-offset fit.

Independent native-resolution rendering of the actual shaft triangles gives
reserved RMS of 2.71/2.31 px, versus 24.39/38.12 px for the published baseline
(PSM1/PSM2). This audit preserves edge identity and uses 3/12 original reserved
points; PSM1 has only a short frame-525 fragment. The analytic table above uses
resampled ideal-cylinder curves. The retained polygonal tube's finite endpoint
projection and approximately 1 px raster quantization explain why these are
separate measurements. Actual phantom-only raster checks give reserved
frame-300 pad-side RMS of 8.21/3.48 px and frame-525 RMS of 11.78/2.24 px.
The 240-wide pad render has about 2.7 native pixels per wrist pixel; residual
quantization and the larger PSM1 mismatch remain visible.

The fixture feet define the table plane. Exact RCM-port closure and parallelism
are enforced constraints, not measured calibration precision: the CAD port
centers have approximately 3 mm uncertainty. The transparent endoscope follows
the stereo optics and uses the recorded 154.98 mm ECM tip-to-pivot distance for
length. This does not determine its optical angle. The straight-scope convention,
diameter, optical hand-eye and third fixture opening remain unmeasured.

Geometric clearance audits sample distal jaw surfaces across all 597 supported
frames. Minima below distinguish Euclidean nearest-surface distance from signed
clearance to the topmost phantom surface along world Z or its base normal:

| Jaw | Nearest surface (mm) | World-Z clearance (mm) | Base-normal clearance (mm) |
| --- | ---: | ---: | ---: |
| PSM1 jaw 1 | 0.724 | 1.073 | 1.068 |
| PSM1 jaw 2 | 0.891 | 0.984 | 0.979 |
| PSM2 jaw 1 | 9.364 | 12.351 | 12.350 |
| PSM2 jaw 2 | 6.228 | 10.396 | 10.287 |

No negative sampled surface or table clearances were found. These measurements
do not certify exact mesh contact or grasp success.
A held needle can span a legitimate tool-to-pad gap; the rigid-needle
visualization does not independently resolve metric depth. Exact PSM2 instrument
CAD and the physical jaw geometry under needle load are unavailable; the static pad also cannot
reproduce local recorded deformation. These limitations remain visible in the
[actual frame-300 mesh overlay](media/fem_suturebot_alignment.jpg).

The additional source-only shaft review contains 21 short wrist-image segments
(63 original-pixel points), with 6–9 pixel uncertainty. It reserves frames
300 and 525; PSM1 frame 300 has no reliable observed edge and is excluded.
No stereo long-cylinder tangent is inferred from a dark vignette or a visible
milled connector. The observed edges belong to the reflective outer shaft,
not its narrow central reflection. Since each wrist camera moves with that
shaft, these repeated observations test mount/lens consistency rather than
independently measuring global tool height.

The partial-mask review adds 35 records over fitting frames 150/450 and reserved
frames 300/525: 213 points on 51 external contour fragments, 70 known-foreground
points and 60 known-background points. Native-resolution masks label only small
reviewed point disks; all other pixels remain ignored. Jaw observations are
audited separately rather than fitted until their geometry/opening is verified.
PSM2 silhouette residuals include the documented surrogate-CAD discrepancy.
The independent full-scene USD render gives reserved head-contour RMS of
8.27/5.66 px in PSM1/PSM2 wrist views and 10.31/7.53 px in left/right stereo
views. Reviewed head foreground points lie inside the rendered arms, but some
stereo background points are incorrectly covered; those remaining mask errors
are reported separately. Jaw checks remain diagnostic, including a PSM2
reserved contour RMS of 8.59 px and missed PSM1 jaw foreground points. These
actual-mesh checks use full-arm visibility and original points, so they differ
from the fit's head-only, mixed contour/foreground/background objective.

The two existing dataset point labels are rendered as an ordinary USD UV
texture on the original tetrahedral exterior triangles. The nearest visible
surface hit determines each triangle and barycentric binding; face-varying UVs
preserve the source mask coverage without changing reference vertices or triangle
indices. The rendered markers use bright green RGB `(0, 255, 0)` for visibility;
the original mask remains unchanged, with source and display policies recorded
separately. Before the pad translation, an actual native-resolution camera render
recovered all 162 labelled source pixels. After translation, the same material
labels move with their triangles: 159 green pixels are visible in the left image,
with source-mask IoU 0.0355. That remaining image mismatch is reported explicitly.
Strict material-point color checks still pass on CPU and CUDA. The replay does not evolve
the FEM vertices; the material bindings can follow the same vertices if they
are deformed.

The needle uses recorded jaw hysteresis for grasp/release. A fresh open-to-closed
transition translates a free needle's nearest centerline point to the grasp
center without rotating it in the world. Each later release and fresh closure
can repeat this assistance. The resulting tool-to-needle transform stays fixed
while held, and the public Newton MuJoCo solver advances it after release.
Nearby closed-jaw capture and handoff remain available. Closure events are
consumed while another tool holds the needle, preventing a stale closure from
causing delayed capture. This is imposed replay assistance, not a physical
contact-grasp or image-alignment result. Needle dynamics use its CPU backend by default, including during CUDA
rendering. No recorded needle trajectory is stored or consumed. Tests cover
release velocity and gravity, collision support, nearby closed-jaw capture,
handoff, repeatable backward seeking and clock mismatch rejection. The physical
CAD end labels are corrected: the tapered endpoint is the sharp tip. Its image
placement remains explicitly unaccepted. The only static environment collider
is the flat rectangular backing beneath the raised tissue, approximately
133.19 × 132.48 × 4.45 mm. Raised tissue, tools, fixture, endoscope and table
have no needle collision shapes. A held needle follows its imposed grasp and
can pass through the raised tissue; after release, MuJoCo lets it fall onto
the backing. No tissue puncture forces, deformation or suture dynamics are
modeled. Passing these tests does not establish the recorded grasp,
needle-to-pad contact or suturing success.

Rendering checks cover asymmetric intrinsics, rotated/translated optical
frames, nearest-surface occlusion, moving geometry, unavailable frames and
the absence of artificial edges at coplanar region seams. Before the later
fresh-closure assistance change, all 21 then-existing replay,
camera, UV and needle tests pass against the extracted wheel on CPU (6.313 s)
and CUDA (7.007 s), including both actual-texture and material-binding tests. The
portable-package test copies the entire sample before loading it, checks that
USD dependencies stay inside the copy, and rejects workstation paths.
Seven preparation camera tests additionally verify inverse-lens rays, source-image
dimensions, fixed camera mounts, reserved curve splits and physical constraints.
They independently resample the source Cartesian poses and verify that one
constant arm transform preserves every supported relative motion increment.
Together with six projection/constraint tests, five actual-mesh mask tests and
two input-preservation regressions, all 20 preparation tests pass. Three actual
source frames (150, 300 and 525) were rendered through all four cameras and
reviewed separately; the independent boundary audit covers seven frames. The
fresh wheel includes 2,478 bundle files: all 2,404 source JPEGs, 35 reviewed mask
rasters, the UV texture and the rigid needle assets. Every bundled asset is
byte-identical to the published sample. It passes all 597 supported replay
frames from an extracted copy outside the repository, with every USD and
texture dependency resolving inside the bundle.
The later green-marker update changes the atlas colors and display metadata;
source images, source masks, USD bytes, material bindings, motion and cameras
remain unchanged. Both focused UV tests pass again on CPU and CUDA, and five
preparation tests pass, including exact preservation of atlas coverage during
recoloring. The refreshed wheel also passes the dependency and full supported-frame
replay checks. Later pad-placement checks verify unchanged local geometry,
UVs, tool/camera arrays and needle grasp. The new backing regression checks
one actual USD box, free fall through raised tissue, and a stopped needle on
that backing after 400 ms of MuJoCo dynamics. All 597 frames replay with the
initial grasp held through frame 303, release at 304, and deterministic backward
seeking in that historical run. Its proximity-only rule did not produce a later
capture. The current fresh-closure assistance and its new audit are described
separately below.

Four additional source episodes (tissue3_212958, tissue4_211216, tissue5_102359
and tissue7_200743) were prepared and rendered in the MSc workbench. All were
reviewed as region 1; this does not validate automatic classification of all six
dataset regions. Their remaining tool and pad discrepancies fail image
alignment acceptance, so none is included in the example package. Successful
export, coordinate consistency and UV rendering are separate from calibration
acceptance.

The full CUDA suite passed all 162 then-existing tests in 286.227 seconds after
the pad translation, including the retained FEM contact regressions. The later
backing-only asset change and added regression are covered by the 21 focused
tests above; no full 163-test run is claimed. Repository-wide
Ruff checks and formatting also pass. Earlier runs with CUDA-context memory
errors and the interrupted full CPU run remain recorded as historical checks;
they are not substituted for this completed run.

The [full viewer recording](media/fem_suturebot.mp4) contains 1,202 frames at
1280 × 720 and 60 fps, with four simultaneous overlays. Its
[capture metadata](media/fem_suturebot.json) confirms all 597 supported source
frames were displayed; the regenerated capture took 50.880 seconds with CUDA rendering and the
MuJoCo CPU needle backend. The close-up from 3 to 14.17 seconds makes both green
surface marks and the frame-127 approach visible in the main OpenGL viewport.
Encoded frames at 4.1 and 10 seconds were inspected separately from
the camera overlays. Source frames 444/448/464/494 additionally show the free-to-held
transition and later tool following. Hashes of the final MP4, GIF, poster and metadata are
recorded in the validation JSON. Playback rate is not a dynamics or live-performance
measurement. The self-contained sample is documented in its
[README](../src/newton_solver_fem/examples/assets/suturebot/recordings/tissue3_215300/README.md).
Preparation commands and full audit arrays remain separately under the MSc
workbench in `docs/recording-replay.md`; the example does not require it.

```bash
PYTHONPATH=tests uv run --locked --extra examples python -m unittest test_suturebot_recording test_suturebot_cameras test_suturebot_point_labels test_suturebot_needle -v
NEWTON_TEST_DEVICE=cuda:0 uv run --locked --extra examples python -m unittest discover -s tests -v
uv run --locked --extra examples python tools/record_examples.py fem_suturebot --width 1280 --height 720
```


## SutureBot bounded approach replays (2026-11-03)

Two additional portable samples are included in the example selector:
`tissue3_222119_approach` and `tissue3_224616_approach`. The
[structured approach report](validation/suturebot-approaches.json) records their
current bundle hashes, scoped image residuals and portable checks separately
from the default needle example above.

| Sample | Source frames, inclusive | Duration | Reserved image frames | Historical incision RMS in left / right / PSM1 / PSM2, px |
| --- | --- | ---: | --- | --- |
| `tissue3_222119_approach` | 5–141 | 4.527 s | 33, 76 | 2.05 / 1.72 / 1.75 / 12.64 |
| `tissue3_224616_approach` | 4–106 | 3.401 s | 27, 61 | 4.03 / 4.59 / 9.45 / 5.66 |

The user now confirms the entire `224616` recording is public region 1,
including both visible incision fragments. Nearby feature A is tentatively
region 2. This supersedes the earlier region-3 interpretation. The bundled
placement and green label bindings at that identity-review stage still came
from the historical region-3 fit and were marked invalid for the corrected
identity. The current bundle subsequently received a region-1 pose and rebound
labels; its alignment remains unaccepted. The table's
`224616` residuals describe that old fit; they do not validate a region-1
placement. All twelve native meshes and their canonical IDs remain unchanged.
The correction and exact prior metadata are retained in `audit/region_identity/`.

Source-only inspection defined these initial approaches before visible tissue
deformation, independently of the fit residuals. The full 575/461 frame arrays,
original timestamps and reserved-frame roles remain intact. Normal playback and
the frame slider stop at the declared bounds. Later frames remain available for
diagnostics; the complete recovery is not validated against a static phantom.

Both historical fits passed the original incision/cap phase screen: at least two reserved incision frames per
camera, incision RMS at most 25 px and median at most 12 px, plus separate cap
and source-motion checks. These are engineering thresholds, not pixel-exact
agreement. The superseded region-3 fit for 224616 has a 20.38 px maximum PSM1
incision residual and 12.27 px maximum right-image residual. Full-clip reserved stereo incision RMS
remains above the declared approach errors; those full-clip values remain in
the bundled reports. Own PSM1 cap centers are not observable and are not invented.

The original four-camera reviews identify corresponding tool neighborhoods and
incision direction while retaining visible jaw, head and groove-bank differences.
PSM2 uses surrogate CAD. Both `alignment.accepted` and
`absolute_metric_accuracy_validated` remain false. The pad bottom is approximately
20.2 mm above the estimated fixture-foot table plane for 222119 and 17.8 mm for
224616; these dimensions are not independently measured or a physical mounting
validation. No exact needle contact, complete suturing or tissue dynamics is claimed.

**Recording 4 boundary correction (2026-11-04):** independent native-resolution
USD silhouette checks found reserved outer-edge RMS of **193.94 px in PSM1 and
159.86 px in PSM2** at source frames 27 and 61. These fail the 12 px RMS / 30 px
maximum thresholds and the 80% pad/background-side recall checks. The earlier
incision-only publication result is superseded: the bundle now records a failed
alignment check and is retained for inspection. Geometry, recorded motion,
cameras and region numbering were not changed by this status correction.
The bundle's `audit/pad_boundary/` contains the native observations, rendered
review and numerical report. New publication requires this separate boundary
check; a passing incision trace cannot qualify the pad rectangle as aligned.

The two dataset labels are bound to the mesh and rendered green in all four
views. Recording `224616` reproduces all 162 source-mask pixels after rebinding
to region 3. Recording `222119` preserves its original material bindings through
the new pad translation: at annotation frame 574, outside the replay interval,
the two points move about 14.8/15.2 px from their source pixels and mask IoU is
0.0. Rendering success is separate from this source-label mismatch. The
original mask files, clicks and mesh regions are preserved. The final-frame
annotation is a static material target, not a measured contact event during
the approach.

Recording `222119` now contains an estimated needle after jointly fitting its
shape and fixed grasp, PSM1 optics/mount and a constant pad translation. Original
reserved frames 33/76 remain excluded from fitting. Needle reserved RMS is
4.66 px and maximum error 11.48 px, within the original thresholds. All 137 held
poses replay deterministically with at least 5.60 mm clearance from the backing.
Native renders still show a thicker needle and approximately 10 px endpoint
error at frame 139. The tip also has shallow 0.025–0.077 mm raised-surface
occlusion at frames 138–140; complete surface-contact alignment is unaccepted.

The PSM1 first-frame head RMS improves from 8.54 to 7.75 px and shaft RMS from
15.47 to 11.61 px. Held-out head/shaft checks improve too, while opposite-tool
cap RMS worsens from 8.88 to 11.41 px and PSM2 incision RMS from 7.27 to 12.64 px.
These tradeoffs and visible CAD differences remain explicit; passing the existing
thresholds does not establish full camera alignment. The earlier needle fit is
retained as predecessor evidence. `224616` still has no needle because its fit
retained roughly 25–30 px early sharp-tip discrepancies. Penetration through
raised tissue is permitted; only the flat backing is a collision surface.

Before the region correction, the same batch preparation command was repeated with both cache hits and all
nine recorded geometry, motion, calibration, clock and screening artifacts
byte-identical per sample. Source-input preservation, coordinate checks and
actual camera rendering pass. Portable publication retains relative USD/texture
paths, original image clocks and bounded playback metadata. The full FEM CUDA suite passes all 164 tests in 216.751 seconds with the
default backing sample and both approach bundles. The structured report records
this run separately from the earlier default-only checks and the isolated-wheel
validation. All 105 final workbench recording tests pass in 10.646 seconds,
including source-label preservation, pad-report regeneration, stale-feature
guards and phase/publication policy regressions.

Preparation remains in `MSc/SutureBot-workbench`, using
`config/recording_batch/newton_approach_examples.json`. The Newton runtime loads
prepared data without refitting cameras or importing the preparation checkout.


### Full supported pickup replay

`tissue3_215248_pickup` is also bundled. It retains all 282 original source
entries and replays all supported frames 4–281 (9.226 seconds) from
`tissue_3/1_needle_pickup/20250428-215248-974340`. The entire supported motion
precedes visible tissue deformation; there is no excluded later tail. Source
frames 0–3 remain unavailable under the recorded robot clock.

Independent actual-mesh reviews cover reserved frames 39, 94, 149, 204 and 259
in all four cameras. Reserved incision RMS is 6.48/8.27/10.04/10.27 px in
left/right/PSM1/PSM2. The source-defined phase and original reservations remain
unchanged. Tool neighborhoods and incision direction follow the images, while
jaw shape/opening, near-shaft contours and groove-bank widths retain visible
mismatches. The estimated pad bottom is 18.15 mm above the fixture-foot table
plane; metric mounting accuracy is not validated.

This recording has no source clicked-point CSV or mask. Its manifest records
`source_labels.available: false`; there are no invented UV markers and no needle
assets. It replays tools used during pickup, not a validated needle grasp.
Calibration remains `accepted: false`. Source preservation, metadata repair and
unchanged-input cache proofs are retained alongside the actual render review.
The structured report includes its current bundle hashes and final isolated-wheel
checks; the full 164-test CUDA run above predates this third asset addition.


Before the fresh-closure needle change, the validated wheel contained the
default sample and all three new pose replays. It was 126,689,059 bytes with SHA256
`f2b7ceb0c7b07eb0c15e13e359974afc02e178f298b8a7e6b18d4e554ab9b4b6`.
For that wheel, an isolated `python -I` process loaded the extracted package
outside the checkout, checked all 5,448 files across the three new bundles byte-for-byte, and rendered
all four cameras on CUDA at each replay's first, last and original reserved
frames inside its scope. All passed without the source dataset or preparation
checkout. The post-pickup CUDA recording suite also passes all 11 tests in
11.685 seconds across all four bundles, covering source clocks, all-frame
replay, sample switching and copied-package independence. Final Ruff checks,
formatting of 122 Python files and `git diff --check` pass. These runtime and
portability checks do not change any sample's estimated calibration status.


### Publication image gate

The three new scoped bundles each pass all 28 required publication image checks.
The gate rechecks the declared numerical limits and reserved observations in
all four views, visual-review readiness, exact reviewed overlay hashes and the
unchanged geometry, camera and motion inputs. Runtime or rendering success alone
cannot substitute for a missing or failed image review.
`--publish-review-required` only permits a clearly labeled absolute-calibration
estimate after those checks; it cannot waive image failures or include a failed
needle initialization. Only the original `tissue3_215300` reference without automatic-fit metadata
retains its legacy estimate status; every other new recording requires the
four-camera image scope. All three new samples still have `alignment.accepted: false`.


### Repeated needle capture on fresh jaw closure

The default recording now releases the needle at source frame 304 and captures
it again at frame 446. The new closure translates its nearest centerline point
8.436046 mm to the PSM1 grasp center while preserving world orientation. The
measured grasp-point error is 1.31e−9 m. Across all 597 supported frames, the
needle is held for 455 frames and free for 142; backward replay reproduces the
poses exactly. All 12 audited geometry, initial-needle, camera and motion assets
remain byte-identical.

Eight focused needle tests cover multiple release/closure cycles across both
arms, preserved capture orientation, fixed held transforms, consumed stale
closures, nearby handoff, free dynamics/backing collision and the actual
frame-446 replay. The initial needle image placement remains unaccepted.
Repeated closure translation is deliberate replay assistance, not evidence of
physical grasp contact, calibrated needle position or suturing success.


After the repeated-capture change, the complete FEM CUDA suite passes all
167 tests in 240.473 seconds. All 113 preparation tests pass in 11.701 seconds,
including strict four-camera publication checks and the explicit original-reference
exception. The current 1,202-frame viewer video covers all 597 supported source
frames and records the 4/304/446 initial-grasp, release and recapture events.
Current runtime, media and unchanged-input hashes are in
[the default recording validation](validation/suturebot-recording.json); these
supersede the older proximity-only needle audits retained as history.


The wheel built after the capture change, before the region-3 correction, is
126,693,990 bytes, SHA256
`0832189164641f07babdbce0840d19f338af941731ba9b8333567a0466b4a782`.
Isolated CPU checks import that needle runtime from its extracted copy,
verify all 7,928 files across the four bundles byte-for-byte, pass eight needle
tests including actual capture at frame 446, and render 20 source frames through
all four cameras across the three pose-only samples. Runtime needs no source
checkout. The final-wheel CUDA attempt was incomplete because its context
allocation ran out of memory; that failure is retained separately. The successful
167-test repository CUDA suite is separate evidence, not a claim that the final
extracted wheel completed CUDA checks. No other GPU process was changed.
Those wheel checks do not validate the current region-1 identity correction,
the superseded region-3 placement, or a replacement calibration.

### Current portable examples (2026-11-04)

The [current pre-push report](validation/suturebot-prepush.json) records 33 passing
CPU replay, camera, needle, CLI and documentation tests, plus two material-label
tests for each corrected recording. The new wheel imports independently and
matches all 7,958 bundled SutureBot files byte-for-byte. Ruff and formatting pass.
CUDA context creation failed because of GPU memory pressure. The full CPU run
was terminated during the legacy active-tissue press test after 156 passes and
five skips; the remaining recording tests pass in the focused run. A separate
legacy press rerun had not finished at commit review, so a completed full-suite
run is not claimed for this revision. These checks do not certify camera alignment.
