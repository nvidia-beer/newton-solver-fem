# Papers mapped to this implementation

| Paper | Implemented portion and code |
| --- | --- |
| Smith, de Goes and Kim, [Stable Neo-Hookean Flesh Simulation (2018)](https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf) | The regularized isotropic energy and rest correction are used by the Tet4 force kernels in [kernels.py](../src/newton_solver_fem/_src/solvers/fem/kernels.py) and its Q1 `eval_hexahedra` kernel. Stretch/determinant clamps and the approximate PSD tangent are implementation choices; this is not the paper's full nonlinear solve. |
| Sifakis, Der and Fedkiw, [Arbitrary Cutting of Deformable Tetrahedralized Objects (2007)](https://pages.cs.wisc.edu/~sifakis/papers/cutting.pdf), §3 | [LinearTetCut](../src/newton_solver_fem/_src/cutting/tetrahedral_cut.py) clips embedded material regions; [CutModel](../src/newton_solver_fem/_src/cutting/model.py) retains parent affine basis functions with duplicated nodes, fragment volume weights and barycentric mass integrals. |
| Acharya and Hyde, [STA-FEM (2026)](https://arxiv.org/abs/2605.11673), §3.4 and Algorithm 1 | [StreamingTetAssembly](../src/newton_solver_fem/_src/solvers/fem/streaming.py) maintains additive element blocks under local topology changes. It adapts the element-assembly idea to reserved copies of basis nodes; it does not implement the paper's graph-Laplacian proxy or benchmark. |

## Streaming assembly pseudocode to code

| STA-FEM Algorithm 1 step | Current implementation |
| --- | --- |
| 1–2: initialize persistent connectivity and matrix state | [SolverFEM.prepare_tetrahedral_updates](../src/newton_solver_fem/_src/solvers/fem/solver.py) constructs `StreamingTetAssembly` before graph capture. It reserves candidate BSR connections for a finite pool of copied original nodes. Incision geometry itself is chosen at runtime. |
| 4–5: receive edits and update connectivity | [LinearTetCut.propose*](../src/newton_solver_fem/_src/cutting/tetrahedral_cut.py) stages clipped geometry, [FragmentConnectivityGPU.launch](../src/newton_solver_fem/_src/cutting/fragment_connectivity_gpu.py) identifies separated basis regions, and [CutModel.update](../src/newton_solver_fem/_src/cutting/model.py) writes the accepted topology/masses. Capacity checks precede publication. |
| 6–15: increment only affected contributions | [StreamingTetAssembly.update](../src/newton_solver_fem/_src/solvers/fem/streaming.py) stages changed elements; `_remove_elements`, `_reuse_blocks`, and `_add_elements` subtract cached old blocks and add new ones. This corresponds to §3.4's additive element extension, **not** Algorithm 1's six-edge multiplicity thresholding. |
| 16: finalize the sparse operator | [_compact_rows](../src/newton_solver_fem/_src/solvers/fem/streaming.py), `_map_compact_slots`, and `_restore_matrix` assemble active rows into persistent storage; mass terms and the block-diagonal preconditioner are refreshed. |
| 17: solve the current system | [SolverFEM.implicit_integration](../src/newton_solver_fem/_src/solvers/fem/solver.py) runs the existing sparse Krylov solve. No dense inverse, Cholesky rank update or Woodbury solve is added. |

Here element tangents depend on deformation and material as well as connectivity;
unchanged blocks are reused only when those inputs agree. Tests in
[test_reference_streaming.py](../tests/test_reference_streaming.py) compare the
maintained matrix with a full rebuild. This verifies the implemented sparse
update, not the paper's full fracture/refinement/merge performance results.
See [method](method.md) for time integration and differentiation scope.
