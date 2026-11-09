# Example guide

The example launcher discovers all runnable demos. Each includes a recorded
preview and launch command in the [main page's example gallery](../README.md#examples).

| Example | What it demonstrates | Details |
| --- | --- | --- |
| `linear_fem` | Six Tet4/Hex8 solids deform under gravity and floor contact, comparing tetrahedral and hexahedral discretizations. | [Source and options](../src/newton_solver_fem/examples/example_linear_fem.py) |
| `linear_fem_baymax_cut` | Drill, slice or carve material, update sparse FEM after topology changes, and use mouse spring forces to lift and hold separated pieces. | [Source and controls](../src/newton_solver_fem/examples/example_linear_fem_baymax_cut.py) |
| `diffsim_bear` | Optimize tetrahedral activation through the fixed-matrix adjoint, then run fresh controller inference to produce a walking gait. | [Source and training options](../src/newton_solver_fem/examples/diffsim/example_diffsim_bear.py) |
| `fem_phantom_regions` | Simulate one selected Tet4 region with fixed interfaces to inactive material; use mouse spring forces to deform it. | [Controls, region mapping and geometry](phantom-regions.md) |
| `fem_suturebot` | Replay measured tool poses and inspect four camera overlays against source videos. This is a static-scene kinematic replay; it does not advance FEM or rigid-body dynamics. | [Recording setup and calibration scope](suturebot-recording.md) |

Install the `examples` extra for the viewer and USD dependencies:

```bash
uv sync --locked --extra examples
uv run --locked python -m newton_solver_fem.examples
```

For a particular example, run `./docker/run.sh <example>` or
`uv run --locked python -m newton_solver_fem.examples <example>`.
Append `--help` to see its available options. The phantom and recording examples
include their prepared assets; no source-dataset checkout is needed for playback.
