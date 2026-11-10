# Example guide

The launcher discovers the three runnable examples below. Their geometry is
bundled under `src/newton_solver_fem/examples/assets/`; no external source
checkout is required. See the [main example gallery](../README.md#examples)
for recorded demonstrations. Movie playback rate is not live simulation speed.

| Example | What it demonstrates | Source |
| --- | --- | --- |
| `linear_fem` | Six Tet4/Hex8 solids deform under gravity and floor contact. | [Example and options](../src/newton_solver_fem/examples/example_linear_fem.py) |
| `linear_fem_baymax_cut` | Drill, slice or carve material and lift separated pieces with spring holds. | [Example and controls](../src/newton_solver_fem/examples/example_linear_fem_baymax_cut.py) |
| `diffsim_bear` | Optimize tetrahedral activation through the fixed-matrix adjoint, then simulate the controller. | [Example and training options](../src/newton_solver_fem/examples/diffsim/example_diffsim_bear.py) |

## Setup

Install the `examples` extra for the OpenGL viewer and USD loading:

```bash
uv sync --locked --extra examples
uv run --locked python -m newton_solver_fem.examples
```

For Docker, use `./docker/build.sh`, then `./docker/run.sh --list` or
`./docker/run.sh <example>`. Visual Docker runs require a local X11 display and
readable `XAUTHORITY`. For supported CPU/headless scenes, use
`./docker/run.sh --cpu --headless <example>`.

Locally, choose an example with
`uv run --locked python -m newton_solver_fem.examples <example>` and append
`--help` for its options. Use `--device cpu` to select CPU execution, or
`--viewer null --num-frames N` for a finite run without OpenGL. These options do
not remove the simulation or, for the bear's default mode, its training work.

## Tet4 and Hex8 solids

`linear_fem` defaults to the six-body showcase. `--mode tet` or `--mode hex`
selects one element family. Materials and numerical budgets are illustrative;
"linear" refers to the shape functions, not a linear constitutive force.
The ordinary viewer controls handle camera movement and pause.

```bash
uv run --locked python -m newton_solver_fem.examples linear_fem --mode hex
```

## Cutting and material holds

The Baymax example starts with the cylinder tool. Left-click to drill through
the body. Choose **Line**, left-drag and release for an incision; choose
**Sphere** and hold the left button to carve along the initial viewing
direction. Right-drag to lift material. Releasing creates a spring hold at a
green marker; middle-click the marker or choose **Clear holds** to release it.
**Reset body and cuts** restores the initial body. The connectivity debug view
can display the separated material pieces and grip connections.

Cuts retain the parent Tet4 affine fields with duplicated disconnected nodes,
volume weights and sparse topology updates. Capacity and distortion checks can
reject an operation. The demo does not guarantee freedom from inversion or
contact penetration under arbitrary interaction; see [the method](method.md).

## Differentiable bear

The default run performs one Adam update on a 300-frame training trajectory,
then displays newly simulated controller inference. It does not replay stored
training positions. Frame checkpoints reduce trajectory storage by recomputing
substeps during the backward pass. The 80 physical substeps per frame and
backward solve make training more expensive than inference.

```bash
uv run --locked python -m newton_solver_fem.examples diffsim_bear \
  --train-iters 1 --sim-steps 300 --save-weights /tmp/bear-controller.npz
uv run --locked python -m newton_solver_fem.examples diffsim_bear \
  --inference --load-weights /tmp/bear-controller.npz
```

`--train-iters`, `--sim-steps`, `--load-weights`, `--save-weights` and `--no-graph`
control the optimization or numerical execution. Gradient support is limited
to the documented fixed-matrix contract; it does not include topology creation.
The [validation guide](validation.md) separates forward simulation, training,
rendering and historical timing results.
