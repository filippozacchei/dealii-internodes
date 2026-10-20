# RBF radius study (Geometry-B, P2/P2, 768 processes)

Effect of the RBF radius factor r_f (r = r_f · h_avg per subdomain) on the
accuracy, on two shell meshes: `shell54` (shell refinement master 5 / slave
4, 1.8e6 DoFs; the paper's original Test 5) and `shell65` (6 / 5, 1.43e7
DoFs; the current Test 5). One run each on Galileo100, 768 processes,
commit 1675c73, Φ factored with ScaLAPACK (r_f = 10: `auto`; the others
forced with `INTERNODES_PHI_SOLVER=scalapack`). The interface sphere has
diameter 1.5.

| r_f | shell54 broken H1 | shell65 broken H1 | ratio | order | radii shell65 (master / slave) | Φ density shell65 (master / slave) |
|---|---|---|---|---|---|---|
| 10  | 0.0521  | 0.0358   | 1.46 | 0.54 | 0.21 / 0.59 | 2.1% / 16% |
| 20  | --      | 0.0189   |      |      | 0.43 / 1.19 | 8.2% / 63% |
| 50  | 0.00389 | 0.000781 | 4.98 | 2.32 | 1.07 / 2.97 | 51% / 100% |
| 100 | 0.00541 | 0.000990 | 5.47 | 2.45 | 2.14 / 5.93 | 100% / 100% |

GMRES iterations: 11 (shell54) and 10 (shell65) for every r_f. With r_f
fixed the support shrinks with the mesh (stationary RBF interpolation) and
the transfer error saturates (order 0.5 at r_f = 10); a support comparable
to the interface restores the optimal order 2. r_f = 100 is slightly less
accurate than 50: Φ is worse conditioned (rcond 2.8e-12 vs 8.3e-11 for the
shell65 master).

The `.prm`/`.sh` files were moved here from `rf_study_galileo100/` and
`scalability_galileo100-final/` (tag `results-history`) with their paths
rewritten; the logs still show the original paths.
