# SutureBot recording bundle

**Scoped replay: Pickup motion before tissue interaction. Original source frames 4–281, inclusive.** All 282 original timestamped entries and synchronized images remain packaged; source pose support remains 4–281. Playback covers the complete supported static-pad motion interval; no later source frames are excluded. This does not validate tissue interaction or suturing. `source_phases.json` records the source-only phase decision; `alignment_screening.json` retains scoped checks and full-clip failures. `audit/pickup_visual_review.json` records the scoped four-camera review. This recording has no source clicked-point annotations, so no material labels are included. No needle model is included in this pose replay. Publication preserves these review decisions and does not certify absolute metric accuracy.

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

`motion.npz` exposes `timestamps_ns` int64[282], `time_s` float64[282], `body_poses` float64[282,14,7], and `body_valid` bool[282]. `camera_poses.npz` exposes matching timestamps, `camera_poses_cv` float64[282,4,4,4] and `camera_valid` bool[282,4]. Camera order is left, right, psm1, psm2. Frames 4–281 are supported; the fitted clock correction is already applied to prepared poses.

Recorded tools use position replay with estimated visual alignment. Fixture ports anchor the calibrated arm bases. The table follows the fixture feet; the parallel phantom's independently fitted displacement is recorded in the rig report. The transparent endoscope remains rigidly attached to the calibrated stereo optics, with recorded ECM insertion length. Unmeasured housing dimensions and port-midpoint residuals are explicit in `recording.json` `rig_geometry`. `alignment.json` records residuals at original image resolution; absolute metric accuracy is unvalidated. Source names and preparation-script hashes in `provenance.json` document derivation only.

`pad_placement.json` records a separate constant phantom translation with tools, cameras, fixture, table and any existing needle grasp held fixed. `phantom.T_world_from_native` applies to the unchanged regional geometry. Any pad collision support follows its placement. Earlier calibration fit reports describe the predecessor pad placement. Current placement is assessed only within the explicit source-defined playback scope; see its screening and visual-review reports above.

`automatic_fit_metrics.json` records episode-specific image registration, region identification and validation; inherited camera parameters remain explicit priors.
