# Sparse FEM method

`SolverFEM` advances position-only Tet4 and Q1 Hex8 solids in SI units.
Tet4 uses constant reference gradients; Hex8 uses $2\times2\times2$ Gauss
quadrature. Both assemble $3\times3$ blocks into a sparse BSR matrix. “Linear”
refers to the shape functions; the constitutive force is nonlinear.

The [paper-to-code mapping](papers.md) identifies the implemented portions of
Stable Neo-Hookean elasticity, embedded cutting and STA-FEM streaming assembly.

## Elasticity and stepping

For deformation gradient $F$, $I_C=\mathrm{tr}(F^T F)$, $J=\det F$, and
Lamé coefficients $\mu,\lambda>0$, the smooth-branch energy is

$$
\psi(F)=\frac\mu2(I_C-3)-\frac\mu2\log(I_C+1)
 +\frac\lambda2(J-\alpha)^2,\qquad
\alpha=1+\frac{3\mu}{4\lambda}.
$$

Its stress is $P=\mu(1-1/(I_C+1))F+\lambda(J-\alpha)\mathrm{cof}\,F$.
The force evaluates clamped principal stretches (default $[0.05,20]$) and
replaces $J$ by $\max(J,0.01)$ in the volumetric stress. Thus the displayed
energy derivative applies on the unclamped branch. Viscous forces are explicit.
These regularizations do not guarantee positive integrated element volumes.

With lumped mass $M$, step $h$, approximate stiffness $\widetilde K$, and
assembled spring damping $C_s$, each substep solves

$$
A=M+h^2\widetilde K+hC_s,\qquad
A\Delta v=h(f_{\rm elastic}+f_{\rm viscous}+f_{\rm contact}+f_{\rm ext}+Mg)
-h^2\widetilde K v,
$$

$$
v^+=v+\Delta v,\qquad x^+=x+h v^+.
$$

The omitted geometric Hessian terms make this a linearly implicit update,
not a converged nonlinear equilibrium. Block-Jacobi Krylov solves never form
a dense global matrix. The showcase refreshes the tangent each frame and uses
10 substeps/50 CG iterations; cutting uses 4/30 at 60 Hz. Tet showcase/bear
meshes also carry unique-edge springs. Hex showcase masses use a node-0/node-6
volume estimate and per-node floor $15h\,\mathrm{kg/s}$; cutting instead
integrates occupied volume. These example mass prescriptions differ.

The fast-path Krylov relative residual tolerance is the public
`linear_solver_tolerance` setting, defaulting to $10^{-2}$, with zero absolute
tolerance. Each solve retains its caller-selected iteration cap. On supported
CUDA devices, `linear_solver_use_cuda_graph=True` and `check_every=0` use Warp's
conditional graph loop to stop at the same residual target without a host
synchronization per iteration. Unsupported devices or an explicit opt-out use
the ordinary path. Persistent Krylov workspaces are allocated without running
a discarded solve. The differentiable branch retains its separate checked
fixed-matrix solve contract.

The non-differentiable CPU CG path exposes the same operator as one public
Warp reduction batch spanning all scalar degrees of freedom. This avoids
redundant lane work in the installed Warp CPU tiled reduction; the matrix,
preconditioner, right-hand side, iteration cap and tolerance stay unchanged.
Existing batched operators, other Krylov methods and CUDA retain their original
paths. Reduction order can differ, so parity is checked against the physical
solution and true residual rather than bitwise equality.

Ground contact is an analytic penalty with inward normal damping and velocity
friction capped by both Coulomb force and particle weight. Surface contact
uses evolving boundary geometry. External forces enter through
`step(..., external_forces=forces)`; incoming `particle_f` is workspace.

Stationary boundaries use `SolverFEM(..., fixed_particle_indices=indices)`.
With $P$ the projector onto free nodes, the solve is
$(PAP+I-P)\Delta v=Pb$. Boundary positions remain at the model's construction
positions and boundary velocities are zero. Physical masses and the original
matrix remain intact, including the constrained columns needed by the elastic
velocity predictor. Boundary state entries are restored before force evaluation
and after integration; tangent/topology reassembly also restores fixed positions.
This option currently excludes differentiable stepping,
streaming topology and component scaling; no gradient support is claimed for it.

On CUDA, a separate sparse value buffer stores $PAP+I-P$ for Warp's native
sparse multiply. Its values refresh once per solve, including captured solves;
changed sparse storage rebuilds the cached operator. The original $A$ remains
available to the predictor. The projected value buffer shares the original
index buffers. CPU and unsupported sparse layouts retain the masked multiply.

`refresh_tangent(positions, dt)` updates stiffness in place after its first
sparse-pattern preparation. For a non-differentiable, non-streaming solver,
changing `dt` outside tapes/capture also rebuilds mass, spring and triangle
terms and clears the numerical velocity-increment guess. Subsequent steps must
use that timestep; caller-owned CUDA graphs must be recaptured. Differentiable
and streaming solvers retain their construction-timestep contract. A changed
model allocation requires a new solver, while same-storage sparse updates
must preserve captured buffer lifetime.

### Prepared FEM mechanics

`SolverFEM.prepare_step(state, control, contacts, h)` exposes one attempt's
affine mechanics through
[PreparedFEMStep](../src/newton_solver_fem/_src/solvers/fem/prepared_step.py).
Preparation copies the accepted state, evaluates the ordinary forces once,
and retains the elastic velocity predictor. The prepared equation is
$A_D\Delta v=b(f_{\rm ext})$, with the same constrained sparse operator,
load-before-predictor arithmetic, fixed boundaries and integration as `step`.
`base_rhs` is the zero-additional-load impulse in N s; `rhs(forces)` accepts
additional nodal forces in N. `apply_operator`, `apply_preconditioner`, and
`reconstruct` are GPU operations that do not advance physical history.

The caller refreshes the material tangent separately, as for ordinary stepping,
and prepares again after any accepted state, timestep, material or support
change. One owned workspace is reused per FEM solver. Optional `prepare_solve`
and `solve` supply a reference Krylov solution; a simultaneous integrator may
instead solve the prepared equation directly. These operations currently
exclude differentiation, streaming topology and component scaling. They do
not change the existing `SolverFEM.step` or fixed-matrix adjoint contract.

## Regional Tet4 geometry

`list_mesh_regions(path)` reads a USD catalog without loading volume payloads.
`load_mesh_regions(path, region_ids)` loads only the selected regions and returns
a `RegionMesh` in world-space metres with Z up. `None` uses authored active
regions; an empty iterable returns an empty mesh. Visibility does not select
physics, and explicit selection uses an in-memory session layer without
modifying the source asset.

The default prim carries a JSON string `fem:regionManifest`. Each entry has
`id`, `name`, `prim_path`, `color_srgb` (three values in [0,255]),
`tetrahedron_count` and `vertex_count`. Each region's `Volume` child is a
`TetMesh` containing:

| Attribute | Meaning |
| --- | --- |
| `newton:referencePoints` or `points` | Reference vertices; the exact embedding is preferred when present |
| `tetVertexIndices` | Local four-vertex connectivity |
| `surfaceFaceVertexIndices` | Authored regional boundary triangles |
| `fem:globalVertexIndices` | Unique nonnegative original vertex IDs within the payload, shared across adjoining regions |
| `fem:globalTetrahedronIndices` | Unique nonnegative original cell IDs; selected regions may not overlap |
| `fem:boundaryIsCutInterface` | Boolean per boundary triangle identifying an interface to another region |

The reader applies USD transforms, stage units and Y/Z-up conversion, welds
matching global vertices and checks coordinate consistency, cell orientation,
authored boundaries and manifold surface topology. Matching selected interface
faces disappear from the union. Exposed interface vertices are returned as
`fixed_vertex_indices`; these are labels, not automatic constraints. Callers
choose whether to pass them to `SolverFEM`. Materials, masses and physical
state are outside this geometry API. See
[regions.py](../src/newton_solver_fem/_src/geometry/regions.py) for the schema
validation and [region tests](../tests/test_mesh_regions.py) for synthetic
selection and welding checks.

## Cuts and differentiation

Cuts clip material into convex fragments while preserving the parent affine
basis. Disconnected regions receive copied basis nodes; fragment volumes weight
energy and barycentric first moments determine mass. Cached element blocks are
subtracted/added in reserved sparse adjacency, and active rows are compacted
without moving captured storage. Capacity failures roll back. Tiny components
use mass scaling; a bounded volume guard limits distortion. Neither the guard
nor the penalty contact guarantees inversion/intersection freedom.

For a fixed assembled $A$, differentiable mode records the force/predictor and
state update while using the sparse linear-solve adjoint

$$
\Delta v=A^{-1}b(x,v,u),\qquad
A^Tz=\overline{\Delta v},\qquad \bar b\mathrel{+}=z.
$$

Supported derivatives are initial position/velocity, external nodal force, and
Tet activation. Geometry, mass, material, $h$, contact coefficients, and matrix
assembly are fixed; cut topology and contact creation are discrete. Clamps and
contact switches have branch-local derivatives. Checked forward/transpose
solves require $\|Ax-b\|\le2\cdot10^{-5}\|b\|+10^{-10}$; failures raise errors.
This differentiates a converged fixed-matrix solve, not finite CG iterations.
Use unique taped states; rebuild graphs after model changes. Ordinary untaped
stepping allocates no adjoint/trajectory storage. `SolverLinearFEM` is an alias.

The 1,986-node/5,354-tet bear uses 80 substeps at 60 Hz. Eight sinusoidal inputs
produce $u_e(t)=0.3\tanh(\sum_k W_{ek}\sin(5t+2\pi k/8))$; Adam minimizes
$\sum_n(|\bar v_y|+|\bar v_z|-\bar v_x)$ with arithmetic mean vertex velocity.
Frame checkpoints recompute substeps during backward; display runs fresh
inference. Material parameters stay fixed. See [validation](validation.md)
for gradient checks, measured performance and reproducible commands.
