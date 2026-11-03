# SutureBot recording bundle

**Scoped replay: Approach before tissue deformation. Original source frames 5–141, inclusive.** All 575 original timestamped entries and synchronized images remain packaged; source pose support remains 5–574. This bounded initial motion is not a complete needle throw or recovery. The later deforming tail is excluded from playback, not deleted or relabeled as valid rigid-pad motion. `source_phases.json` records the source-only phase decision; `alignment_screening.json` retains scoped checks and full-clip failures. `approach_visual_review.json` records the scoped four-camera review. Publication preserves these review decisions and does not certify absolute metric accuracy.

Copy this directory as a unit. It is self-contained: replay needs no MSc checkout, preparation scripts, original videos, or external data paths. `recording.json` is the entry point.

```bash
python -m newton_solver_fem.examples fem_suturebot --sample path/to/recording.json
```

| File or directory | Contents |
| --- | --- |
| `recording.json` | Relative asset paths, body identities, camera intrinsics/distortion/mounts and timing |
| `scene.usda`, `rig.usdc`, `phantom.usdc` | Tool/camera poses, source RCM cage, endoscope, shared table and unchanged 12-region phantom surface |
| `motion.npz` | Source timestamps, prepared body poses and validity |
| `camera_poses.npz` | Four optical camera trajectories on the same timestamps |
| `cameras/` | Prepared camera rays and validity masks |
| `images/` | Synchronized JPEGs for left, right, psm1 and psm2 |
| `calibration.json`, `alignment.json`, `*_fit_metrics.json`, `evidence/` | Frame transforms, measured image residuals, physical scene constraints and reviewed annotations |
| `provenance.json`, `validation.json` | Logical source identities, hashes and export checks |

Positions are metres in the fixed calibrated scene frame, with Z up. Body poses store xyz followed by quaternion xyzw. Camera matrices map OpenCV coordinates (+X right, +Y down, +Z forward) into that world frame.

`motion.npz` exposes `timestamps_ns` int64[575], `time_s` float64[575], `body_poses` float64[575,14,7], and `body_valid` bool[575]. `camera_poses.npz` exposes matching timestamps, `camera_poses_cv` float64[575,4,4,4] and `camera_valid` bool[575,4]. Camera order is left, right, psm1, psm2. Frames 5–574 are supported; the fitted clock correction is already applied to prepared poses.

Recorded tools use position replay with estimated visual alignment. Fixture ports anchor the calibrated arm bases. The table follows the fixture feet; the parallel phantom's independently fitted displacement is recorded in the rig report. The transparent endoscope remains rigidly attached to the calibrated stereo optics, with recorded ECM insertion length. Unmeasured housing dimensions and port-midpoint residuals are explicit in `recording.json` `rig_geometry`. `alignment.json` records residuals at original image resolution; absolute metric accuracy is unvalidated. Source names and preparation-script hashes in `provenance.json` document derivation only.

`pad_placement.json` records a separate constant rigid phantom placement with tools, cameras, fixture, table and any existing needle grasp held fixed. `phantom.T_world_from_native` applies to the unchanged regional geometry. Any pad collision support follows its placement. Earlier calibration fit reports describe the predecessor pad placement. Current placement is assessed only within the explicit source-defined playback scope; see its screening and visual-review reports above.
Material-label bindings were transported with the pad; their source-pixel displacement is retained in `point_labels.json`.

`automatic_fit_metrics.json` records episode-specific image registration, region identification and validation; inherited camera parameters remain explicit priors.

Dataset point labels are projected to the preserved tetrahedral exterior and rendered through its USD UV texture. `point_labels.json` records the pixel-to-surface bindings and display color. The original source mask is retained unchanged; marker display colors improve visibility without changing label coverage or bindings.

The needle is a separate rigid MuJoCo body. Jaw opening and proximity control attachment/release; while held it follows the gripper at a fixed relative pose. No recorded needle trajectory drives it, and no tissue puncture physics is simulated.

The needle's only static environment collider is the flat rectangular backing beneath the raised tissue (`needle_backing_collision.json`). Raised tissue, table, fixture and tools do not collide with the needle. A held needle follows its prescribed grip; after release, MuJoCo lets it pass through the raised tissue and land on the backing.
