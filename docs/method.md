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

Ground contact is an analytic penalty with inward normal damping and velocity
friction capped by both Coulomb force and particle weight. Surface contact
uses evolving boundary geometry. External forces enter through
`step(..., external_forces=forces)`; incoming `particle_f` is workspace.

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
