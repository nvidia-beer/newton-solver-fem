# Interactive phantom regions

The `fem_phantom_regions` example loads the bundled phantom USD with one region
numbering scheme: regions 1–6 are the wounds in [SutureBot's setup figure](https://suturebot.github.io/),
and regions 7–12 cover the remaining pad areas. Region 1 is the dataset's
training wound; regions 2–6 are its generalization wounds.

One original connected practice feature is divided into three selectable
regions: wound 5 and its thin upper rim are region 5, the remainder including
the lower-left tail is region 8, and the upper slit between wounds 3 and 4 is
region 12. This lets
FEM simulate these areas independently. The other nine original region payloads
and all tetrahedral geometry are preserved. No SutureBot checkout is needed to
run the example. Install the `examples` extra for USD loading and the OpenGL
viewer.

The default 3D camera looks from the native +Y lettering edge, so “3-D Med”
faces the viewer. The USD coordinates and the established upright 2D map
orientation are unchanged.

```bash
./docker/run.sh fem_phantom_regions
# Local equivalent:
uv run --locked --extra examples python -m newton_solver_fem.examples fem_phantom_regions
```

Click a region on the upright 2D pad map to activate FEM for that region.
Selecting another region replaces the current selection, so only one region
is active at a time. Clicking the current region leaves its deformation intact;
use **None** to disable FEM. The map, active selection and command-line
`--regions` option all use the same numbers, 1–12. The 3D pad remains
uniform pink, inspired by the [3-Dmed soft tissue suture pad](https://www.3-dmed.com/product/soft-tissue-suture-pad/),
and inactive material stays fixed. The displayed active tetrahedron count is
the actual FEM model size. Changing selection rebuilds the model from the
reference shape and clears existing holds.

Camera translation is scaled to the pad size (about 68 mm/s with WASD), and
mouse orbit, pan and zoom use a pivot near the pad. Use **Frame pad** to restore
this view.

Right-drag an active region to pull it with the same mouse tool as the Baymax
cutting example. Release to keep a spring hold, middle-click its marker to
release it, or use **Clear holds**. The inactive reference cannot be grabbed.
The selected region includes its local base. Nodes on interfaces with inactive
regions are fixed, and the pad underside is supported as well.

```bash
# Start with the dataset's training wound; click the map to change selection.
uv run --locked --extra examples python -m newton_solver_fem.examples fem_phantom_regions --regions 1
# Activate only the newly isolated area around dataset wound 5.
uv run --locked --extra examples python -m newton_solver_fem.examples fem_phantom_regions --regions 5
# Activate the upper slit between dataset wounds 3 and 4.
uv run --locked --extra examples python -m newton_solver_fem.examples fem_phantom_regions --regions 12
```

The default selection is region 1.

The material controls are demonstration parameters, not values fitted to the
tissue videos. The example retains the source tetrahedral geometry and native
coordinates. Prepared polygons assign the slit/oval and its thin upper rim to
region 5, the upper slit to region 12, and the remaining material to region 8.
These partitions assign existing tetrahedra without changing their geometry.
See the [asset provenance](../src/newton_solver_fem/examples/assets/phantom_pad/README.md)
for the bundled USD and its source.

The [recorded demonstration](media/fem_phantom_regions.mp4) clicks the 2D map,
pulls wound 4, switches to wound 6, disables FEM, then returns to wound 4. Its script
sends mouse input through the map and attachment controls; deformation comes
from the FEM solve. Regenerate its MP4, animated preview and capture metadata
with:

```bash
uv run --locked --extra examples python tools/record_examples.py fem_phantom_regions --width 1280 --height 720
python3 tools/build_readme.py --require-videos
```

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
