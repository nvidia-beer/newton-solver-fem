# SutureBot recording bundle

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

`motion.npz` exposes `timestamps_ns` int64[601], `time_s` float64[601], `body_poses` float64[601,14,7], and `body_valid` bool[601]. `camera_poses.npz` exposes matching timestamps, `camera_poses_cv` float64[601,4,4,4] and `camera_valid` bool[601,4]. Camera order is left, right, psm1, psm2. Frames 4–600 are supported; the fitted clock correction is already applied to prepared poses.

Recorded tools use position replay with estimated visual alignment. Fixture ports anchor the calibrated arm bases. The table follows the fixture feet; the parallel phantom's independently fitted displacement is recorded in the rig report. The transparent endoscope remains rigidly attached to the calibrated stereo optics, with recorded ECM insertion length. Unmeasured housing dimensions and port-midpoint residuals are explicit in `recording.json` `rig_geometry`. `alignment.json` records residuals at original image resolution; absolute metric accuracy is unvalidated. Source names and preparation-script hashes in `provenance.json` document derivation only.

`constrained_fit_metrics.json` records the joint camera/fixture fit and its residuals. The recorded tool motion is preserved with one constant base registration per arm, anchored to its RCM fixture port. Each wrist optical axis is parallel to its tool shaft. The unchanged phantom is parallel to the fixture support plane; its displacement is fitted. Partial tool-mask labels use known foreground/background and leave ambiguous pixels ignored. These are estimated parameters under explicit assumptions, not a measured optical calibration.

`pad_placement.json` records a separate constant phantom translation with tools, cameras, fixture and table held fixed. `phantom.T_world_from_native` applies to the unchanged regional geometry. Earlier calibration fit reports describe the predecessor pad placement. The current translation remains an unvalidated visual estimate.
Material-label bindings were transported with the pad; their source-pixel displacement is retained in `point_labels.json`.

Dataset point labels are projected to the preserved tetrahedral exterior and rendered through its USD UV texture. `point_labels.json` records the pixel-to-surface bindings and display color. The original source mask is retained unchanged; marker display colors improve visibility without changing label coverage or bindings.

Needle and thread models are excluded from this example. Their appearances in the original source images are retained. Tools, pad, cameras, fixture, recorded motion and available point labels are unchanged; existing calibration limitations still apply. Historical feature assets and reports were archived separately under the provenance archive ID.
