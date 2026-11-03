# Papers mapped to this implementation

| Paper | Implemented portion and code |
| --- | --- |
| Smith, de Goes and Kim, [Stable Neo-Hookean Flesh Simulation (2018)](https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf) | The regularized isotropic energy and rest correction are used by the Tet4 force kernels in [kernels.py](../src/newton_solver_fem/_src/solvers/fem/kernels.py) and its Q1 `eval_hexahedra` kernel. Stretch/determinant clamps and the approximate PSD tangent are implementation choices; this is not the paper's full nonlinear solve. |
| Sifakis, Der and Fedkiw, [Arbitrary Cutting of Deformable Tetrahedralized Objects (2007)](https://pages.cs.wisc.edu/~sifakis/papers/cutting.pdf), §3 | [LinearTetCut](../src/newton_solver_fem/_src/cutting/tetrahedral_cut.py) clips embedded material regions; [CutModel](../src/newton_solver_fem/_src/cutting/model.py) retains parent affine basis functions with duplicated nodes, fragment volume weights and barycentric mass integrals. |
| Acharya and Hyde, [STA-FEM (2026)](https://arxiv.org/abs/2605.11673), §3.4 and Algorithm 1 | [StreamingTetAssembly](../src/newton_solver_fem/_src/solvers/fem/streaming.py) maintains additive element blocks under local topology changes. It adapts the element-assembly idea to reserved copies of basis nodes; it does not implement the paper's graph-Laplacian proxy or benchmark. |
| Haworth et al., [SutureBot (2025)](https://suturebot.github.io/static/SutureBot_NeurIPS_2025.pdf), Fig. 2 and §2.1 | [fem_suturebot](../src/newton_solver_fem/examples/example_fem_suturebot.py) replays the recorded tool motion for `tissue_3/2_needle_throw/20250428-215300-489212` over the original static phantom, with four model/source-video overlays. Prepared USD geometry, link poses, optical poses and lens rays come from the separate MSc preparation pipeline. Region 1 identifies the training wound; the regional FEM example presents wounds 1–6 and other areas 7–12. Camera/phantom registration remains a fitted estimate. Tool motion is an alignment replay. A separate rigid needle uses an estimated initial pose, jaw-controlled attachment and MuJoCo after release; this addition does not reproduce the paper's policy, tissue puncture or suturing results. |

## SutureBot data and implementation scope

| Source concept | Current implementation |
| --- | --- |
| Fig. 2: six numbered pad wounds | [fem_phantom_regions](../src/newton_solver_fem/examples/example_fem_phantom_regions.py) uses one visible numbering scheme, 1–12. Offline preparation partitions one connected original feature into wound 5 with its thin upper rim, remainder 8 including the lower-left tail, and upper slit 12; the other nine original payloads and native tetrahedral coordinates are preserved. |
| §2.1: recorded tool motion and four camera views | [Recording](../src/newton_solver_fem/examples/_suturebot_recording.py) reads a structured JSON/USD/NPZ package and preserves exact source timestamps. The replay selects one source index for tool poses, camera poses and all four images, holding each sample until its successor. Derived link poses are assigned kinematically; no dynamics or force measurement is inferred. |
| Checking model placement against recorded pixels | [FourCameraOverlay](../src/newton_solver_fem/examples/_suturebot_cameras.py) renders actual Newton model geometry through prepared inverse-lens rays and camera-to-world poses, then blends visible surfaces and outlines onto indexed source frames. This rendering/registration workflow is an implementation addition, not a claimed algorithm or calibration result from the paper. Numerical projection tests do not establish accuracy of the fitted recording alignment. |
| Visible tool-shaft alignment | Offline preparation compares independently observed wrist-image cylinder edges with finite perspective tangents of the retained shaft CAD, and checks the result against actual triangle renders. The [shaft objective](method.md#suturebot-recorded-motion-and-camera-overlays) estimates mount/lens alignment; an own-shaft view cannot independently determine global tool/table height. This is an implementation addition, not a calibration procedure or measured optical specification supplied by the paper. |
| Fig. 2: pad, RCM fixture and endoscope arrangement | Offline preparation ties each constant arm-base calibration to a fixture port, constrains wrist optical axes parallel to shafts and the pad normal parallel to fixture support, and fits the pad's independent displacement. The source fixture feet define the table plane. A transparent endoscope housing follows the stereo optics; its optical-to-shaft angle is unmeasured. These scene-registration constraints are implementation choices, not measured hand-eye calibration supplied by the paper. |
| Dataset point targets and rigid needle | Existing final-frame clicked points bind to the unchanged tet exterior through barycentric coordinates and a USD UV atlas. A fixed tool-to-needle transform controls held motion; jaw opening releases it, and each fresh closure can translate the free needle to the gripper without rotating it. Nearby closed-jaw handoff remains available, with MuJoCo dynamics while free. These are simulator additions, not recorded needle trajectories or a reproduced SutureBot controller. |
| Earlier approximate physical rig demonstration | [_suturebot_contact_demo.py](../src/newton_solver_fem/examples/_suturebot_contact_demo.py) retains the MuJoCo tools/free needle and sampled two-way FEM contact experiment. Available instrument CAD includes a documented DeBakey surrogate. The [legacy contact method](method.md#legacy-suturebot-rigid-and-deformable-contact-experiment) and press validation are separate from the default recording replay. |

All fitting, source-image decoding and USD preparation remain under the MSc
workbench; the solver package consumes published assets. The package's alignment
report records the status of fitted camera mounts, phantom registration and
timing; [validation](validation.md#suturebot-recorded-placement-replay-2026-11-03)
reports the numerical image errors. Native phantom geometry and recorded
relative tool motion are preserved. Only one pad side is observed;
other sides/corners and the third fixture opening are not independently
registered. PSM2 uses surrogate instrument CAD, and the endoscope housing/table
dimensions include visual assumptions. Measured tool motion therefore does not
make the fitted camera/phantom poses measured calibration. No absolute metric
accuracy or reproduction of the paper's results is claimed.

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
