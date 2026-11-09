**Source identity corrected to region 1; refit required.** The user confirms the entire recording, including both visible incision fragments, is region 1. Nearby feature A is tentatively region 2. The bundled pad placement and green marker bindings still come from the superseded region-3 fit and are not valid for this corrected identity. They remain available for inspection and have not been renamed to region 1. See `audit/region_identity/user_region_correction.json` and its retained historical metadata.

The rectangular pad and background boundary also fail alignment checks in both wrist views. Actual USD silhouette checks on reserved frames 27 and 61 give 193.94 px RMS in PSM1 and 159.86 px in PSM2 for this historical placement. Its earlier passing incision/cap check did not cover that boundary. This is not an aligned example. See `audit/pad_boundary/pad_boundary_review.jpg` and `audit/pad_boundary/screening.json`.

The standard USD cameras now correctly encode the JSON principal points, including the pixel-center convention. This changes eight aperture-offset attributes in `rig.usdc`; see `audit/usd_camera/correction.json`. The replay overlays already use the JSON inverse-ray maps, so this export correction leaves their pad alignment unchanged. All twelve original regional meshes and canonical IDs are unchanged.

# SutureBot recording bundle

**Scoped replay: Approach before tissue deformation. Original source frames 4–106, inclusive.** All 461 original timestamped entries and synchronized images remain packaged; source pose support remains 4–460. This bounded initial motion is not a complete needle throw or recovery. The later deforming tail is excluded from playback, not deleted or relabeled as valid rigid-pad motion. `source_phases.json` records the source-only phase decision; `alignment_screening.json` retains scoped checks and full-clip failures. `approach_visual_review.json` records the scoped four-camera review. No needle model is included in this pose replay. Publication preserves these review decisions and does not certify absolute metric accuracy.

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

`motion.npz` exposes `timestamps_ns` int64[461], `time_s` float64[461], `body_poses` float64[461,14,7], and `body_valid` bool[461]. `camera_poses.npz` exposes matching timestamps, `camera_poses_cv` float64[461,4,4,4] and `camera_valid` bool[461,4]. Camera order is left, right, psm1, psm2. Frames 4–460 are supported; the fitted clock correction is already applied to prepared poses.

Recorded tools use position replay with estimated visual alignment. Fixture ports anchor the calibrated arm bases. The table follows the fixture feet; the parallel phantom's independently fitted displacement is recorded in the rig report. The transparent endoscope remains rigidly attached to the calibrated stereo optics, with recorded ECM insertion length. Unmeasured housing dimensions and port-midpoint residuals are explicit in `recording.json` `rig_geometry`. `alignment.json` records residuals at original image resolution; absolute metric accuracy is unvalidated. Source names and preparation-script hashes in `provenance.json` document derivation only.

`pad_placement.json` records a separate constant rigid phantom placement with tools, cameras, fixture, table and any existing needle grasp held fixed. `phantom.T_world_from_native` applies to the unchanged regional geometry. Any pad collision support follows its placement. Earlier calibration fit reports describe the predecessor pad placement. Current placement is assessed only within the explicit source-defined playback scope; see its screening and visual-review reports above.
Point labels were baked onto the superseded region-3 placement; `point_labels.json` records their unchanged source pixels, historical surface bindings and geometry/motion hashes, with `binding_valid: false`. They require reprojection after a corrected region-1 placement is validated.

`automatic_fit_metrics.json` records episode-specific image registration, region identification and validation; inherited camera parameters remain explicit priors.

The historical dataset point labels are rendered through the tetrahedral exterior's USD UV texture. `point_labels.json` records the invalidated pixel-to-surface bindings and display color. The original source mask remains unchanged. Visible green texels are not evidence of a correct current region-1 placement.
