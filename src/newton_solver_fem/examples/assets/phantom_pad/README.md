# SutureBot regions and selective simulation

The public selection numbers are **1–12**. Regions 1–6 match the six wound
locations in the [SutureBot setup figure](https://suturebot.github.io/static/images/scene_setup.jpg).
Regions 7–12 retain the base and other practice features. Wound 1 is used for
training; wounds 2–6 are the generalization locations. `suturebot_region_map.json`
provides these labels and surface anchors without renumbering the original USD
payloads. The example uses the public numbers for both the map and `--regions`.

Only the original connected region containing the slit, spiral and long groove
was subdivided. Public **region 5** includes the lower-left slit and its thin
upper rim: **18,790 tetrahedra, 4,689 vertices**. Public
**region 12** is the upper narrow slit between regions 3 and 4: **9,801
tetrahedra, 2,652 vertices**. The central spiral and broad long groove remain
visible as region 8 with 60,323 tetrahedra. All nine other original payloads
are **byte-for-byte unchanged**.

The complete pad retains exactly 97,489 vertices and 413,405 tetrahedra, with
no coordinate, connectivity, or USD-root rotation changes. The 2D map retains
its upright presentation; the 3D camera views from the +Y lettering side.

For provenance, the public labels map to stored USD IDs as follows:

| Public label | Stored USD ID | Domain |
|---|---:|---|
| 1 | 6 | Training wound |
| 2 | 2 | Upper-right wound |
| 3 | 5 | Left wavy wound |
| 4 | 7 | Small parallel wound |
| 5 | 10 | Lower-left slit, split from original USD region 1 |
| 6 | 9 | Lower-right parallel wound |
| 7 | 0 | Base and lettering |
| 8 | 1 | Original region 1 remainder: spiral and long groove |
| 9 | 3 | Lower-right curved feature |
| 10 | 4 | Central paired grooves |
| 11 | 8 | Upper-right paired strips |
| 12 | 11 (split from 1) | Upper narrow slit |

The wound-five outline follows the source shoulder and inner bank of the
neighboring groove, including the requested upper rim only. The lower-left wrap remains in region 8. Its accepted lower
boundary is locked during repair (32,173 tetrahedra retain prior ownership). The upper slit
is cut at the curved shoulder above the central spiral. These are visually
matched simulation volumes with local support, **not dataset-provided masks**.
The original surface label joins all these features and supplies no separate
internal semantic boundaries. A topology repair transfers 25 existing
tetrahedra only between the three pieces of that original connected region.
No elements from the nine other regions are transferred. All 12 individual
regions pass closed-manifold edge and vertex-link checks. Combining the three
split pieces exactly restores the original connected region.

All preparation is in `/home/beer/Dev/MSc/SutureBot-workbench`. The current
output is `outputs/simulation/suturebot_regions_upper_rim`, including
`wounds_preview.png`, `region_refinement_comparison.png`, and validation reports. The
configuration is `config/suturebot_wounds.json`; the original runtime bytes
are preserved in `outputs/simulation/suturebot_original_runtime` with their
source commit in `baseline.json`. Reproduce into a fresh output directory:

```sh
cd /home/beer/Dev/MSc/SutureBot-workbench
python3 tools/snapshot_original_phantom.py
PYTHONPATH=/home/beer/Dev/Robotics/NewtonSolvers/newton-solver-fem/.venv/lib/python3.12/site-packages \
  .venv-tissue/bin/python -m suturebot.tissue.suturebot_wounds \
  --output outputs/simulation/suturebot_regions_rebuild
PYTHONPATH=. .venv-tissue/bin/python tools/preview_suturebot_wounds.py \
  --catalog outputs/simulation/suturebot_regions_rebuild
PYTHONPATH=. .venv-tissue/bin/python tools/preview_suturebot_refinement.py \
  --catalog outputs/simulation/suturebot_regions_rebuild
```

The exporter checks exact geometry/rest-array inheritance, element coverage,
and manifold boundaries; it writes a geometry-only `runtime_assets/` bundle.
`--package-output` optionally copies that completed bundle into this directory.
Payloads contain geometry, exact double precision reference coordinates,
original node/element IDs, colors, and boundary topology. Material coefficients
and applied constraints are computed by the example. Selected neighboring
volumes are welded, and only remaining cut boundaries are clamped. As with
the original catalog, a combination touching only at an edge or vertex can
be rejected by the selector even when each individual domain is manifold.

The original phantom came from
[`surgical_robotics_challenge`](https://github.com/surgical-robotics-ai/surgical_robotics_challenge),
commit `26bb2f6`, according to the source SutureBot README. That README records
no upstream LICENSE file for this mesh. The package's Apache-2.0 license does
not establish a license for the third-party phantom. See `provenance.json`
for original and bundled hashes and the metadata stripped during packaging.
Original dataset and reconstruction files were not modified.
