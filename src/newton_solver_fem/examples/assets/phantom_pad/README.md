# Grouped phantom pad

`phantom_regions.usd` and the adjacent `phantom_regions_payloads/` directory
contain the ten approved pad regions from SutureBot-workbench's
`lowered_quality_groups_v3` result. The full conforming volume contains 97,489
vertices and 413,405 tetrahedra. The base includes the “3-D Med” lettering;
region 9 combines the upper purple and light-blue strips.

Each payload contains geometry, exact double precision reference coordinates,
original node and element IDs, colors, and boundary topology. It contains no
constitutive coefficients or applied constraints. The example chooses its FEM
material and applies constraints to the remaining cut boundary after joining
the selected regions. Internal interfaces between selected neighbors are
welded and are not clamped.

The original phantom came from
[`surgical_robotics_challenge`](https://github.com/surgical-robotics-ai/surgical_robotics_challenge),
commit `26bb2f6`, according to the source SutureBot README. That README records
no upstream LICENSE file for this mesh. The package's Apache-2.0 license does
not establish a license for the third-party phantom. See `provenance.json`
for source and bundled file hashes and the exact metadata removed during
packaging. Original dataset and reconstruction files were not modified.
