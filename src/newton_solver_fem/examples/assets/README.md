# FEM geometry assets

`tet_box`, `tet_sphere`, `tet_baymax`, `hex_cube`, `hex_pyramid`, and `hex_tire`
contain volume connectivity and authored boundary triangles for the showcase.
`bear.usd` contains 1,986 vertices and 5,354 tetrahedra for gait optimization.
These assets retain the Newton Developers' Apache-2.0 attribution.

`tet_grid`, `hex_grid`, and `tet_cantilever` are static numerical-test fixtures.
Materials are assigned by simulation code. Only interactive cutting changes
mesh topology during simulation; no asset generator runs at startup.
