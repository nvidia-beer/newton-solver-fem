# Numerical validation

These retained measurements were collected on 2026-10-28 with Newton 1.7.0,
Warp 1.17.0, Python 3.12, Linux ARM64 and NVIDIA GB10. They describe those
configurations and hardware; they are not fresh timing measurements from a
documentation build. Rendering and cold compilation are excluded unless stated.

## Forward simulation

| Scene | Physical configuration | Measured result |
| --- | --- | --- |
| Baymax cutting | 8,965 nodes, 38,730 tets; 60 Hz; 4 substeps; 30 CG iterations; 300 frames; three cuts | Ordinary median 11.783 ms, p95 13.460 ms; cut-frame median 18.721 ms, max 21.424 ms |
| Six-object showcase | 10,030 particles, 39,800 Tet4, 385 Hex8, 54,392 edge springs; 10 substeps; 50 iterations | 46.758 ms/frame over 300 warmed frames; 0.356× realtime |
| Differentiable bear inference | 1,986 particles, 5,354 tets, 9,323 edge springs; 80 substeps; 300 frames | Median 26.836 ms, p95 28.330 ms; 0.619× realtime |

Cutting retained captured storage, committed all three incisions, and ended
with 40,315 fragments/146,350 active BSR blocks in 1,720,180 reserved blocks.
Physical mass remained 7.51521 kg to float32 accumulation accuracy; minimum
Jacobian was approximately 0.329. Every frame was finite with positive element
volume. Ordinary cutting frames met 60 Hz; editing frames did not. The full
showcase and checked bear were slower than realtime. These are physical stepping
measurements, not video playback rates or guarantees of all-scene stability.

A separate unchanged-scene before/after differentiation measurement gave
48.308/48.342 ms per showcase frame. The 0.07% difference is smaller than a
meaningful effect established by single samples; ordinary simulation records
no tape. See `validation/showcase-{before,after}-differentiation.json`.

## Derivatives and optimization

The CPU suite ran 62 tests: 58 passed, four CUDA-only skips (187.537 s).
The CUDA suite passed all 62 (7.522 s). Coverage includes Tet4/Hex8 state/force
VJPs, Tet activation/controller gradients, fixed contact branches, repeated and
clamped stretches, owned snapshots, revision guards, checkpoint/full-Tape
agreement, sparse cut updates, rollback, holds and surface growth. A 120-frame
CPU Tet/Hex fixture passed. No derivative through cuts is claimed.

One genuine Adam update on the 300-frame bear trajectory reduced the objective
227.444229 → 197.617142 (gradient norm 15.003851). Forward/transpose residual guards
passed, updated frame checkpoints had minimum Jacobian 0.786158, and 300 fresh
inference frames remained finite. Initial forward including setup took 8.229 s;
backward with recomputation/setup took 18.207 s; updated forward took 8.044 s.
Frame checkpoints used 14,346,864 bytes plus an 81-state recomputation segment;
that is not total solver memory. Training adds backward/recomputation cost.

A small force fit reduced loss 0.372905731 → 1.68e-12 in four accepted updates,
recovering -0.200000033 N for a -0.2 N target. Its fixed support is an explicit
differentiable projection after each step. Historical results are retained in
`validation/force-fit-differentiable.json`, `validation/bear.json`,
`validation/cut.json`, and `validation/showcase.json`.

## Reproduce checks

```bash
uv sync --locked --extra examples
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -m unittest discover -s tests -v
NEWTON_TEST_DEVICE=cuda:0 uv run --locked python -m unittest discover -s tests -v
uv run --locked python tests/benchmark_showcase.py --device cuda:0 --warmup 30 --frames 300
uv run --locked python tests/benchmark_cut.py --frames 300 --warmup 10 --validate_every_frame --output /tmp/fem-cut.json
uv run --locked python -m newton_solver_fem.examples diffsim_bear --device cuda:0 --train-iters 1 --sim-steps 300 --viewer null --num-frames 300 --save-weights /tmp/bear-policy.npz --test
```

GPU reductions and contact switches can change long trajectories. Timing
comparisons require unchanged scenes, budgets, synchronized warm execution and
an otherwise idle device. Refer to [method](method.md) for the gradient scope.

## Cleanup verification (2026-10-28)

The final standalone suite ran with `python -m unittest discover -s tests -v`:

| Device | Tests | Result | Wall time |
| --- | --- | --- | --- |
| CPU | 72 | Passed; four CUDA-only skips | 211.386 s |
| CUDA (`NEWTON_TEST_DEVICE=cuda:0`) | 72 | All passed | 34.598 s |

The CUDA run used `WARP_CACHE_PATH=/tmp/newton-solvers-cleanup-warp`; test time
includes compilation and setup, so it is not a solver performance measurement.
Lint and formatting passed for all 77 Python files. Wheel and source-archive
builds succeeded. The Docker image built and passed imports, example discovery
and native USD loading in an isolated container with no network or host mounts.
The Docker launcher also passed a two-frame CPU headless example with `--test`.
These checks validate the cleanup and retained behavior; they do not establish
a real-time guarantee.
