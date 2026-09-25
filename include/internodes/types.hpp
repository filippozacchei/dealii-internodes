#ifndef INTERNODES_TYPES_HPP
#define INTERNODES_TYPES_HPP

namespace internodes
{
  /**
   * @brief Geometrical space dimension.
   *
   * The original lifex-based implementation configured this as a CMake
   * build-time constant (LIFEX_DIM). All the paper's numerical results are
   * 3D, so it is fixed here; templating this on the dimension (as deal.II
   * tutorials typically do) is a natural follow-up if 2D support is ever
   * needed.
   */
  constexpr unsigned int dim = 3;

  /**
   * @brief Stopping criteria of one inner CG solve (deal.II's
   * ReductionControl): the solve stops when the residual norm falls below
   * @p tolerance (absolute) or below @p reduction times the initial residual
   * norm (relative), whichever comes first.
   */
  struct CGTolerance
  {
    double tolerance = 1e-13;
    double reduction = 1e-11;
  };

  /**
   * @brief Tolerances of the inner linear solves of the Schur-complement
   * algorithm. The defaults are the values of the paper's runs; they are far
   * tighter than the outer GMRES tolerance, so they can be relaxed to trade
   * some accuracy of the inner solves for speed (see MultiDomainProblem::
   * set_solver_tolerances()).
   */
  struct InnerSolverTolerances
  {
    /// CG on the internal-internal block $A_{k,k}$ of each subdomain (Steps
    /// 1 and 4, and inside every Schur matrix-vector product).
    CGTolerance subdomain{1e-13, 1e-11};
    /// CG on the auxiliary master-subdomain problem of the Schur
    /// preconditioner (one per preconditioner application).
    CGTolerance preconditioner{1e-13, 1e-11};
    /// CG on the slave interface mass matrix $M_{\Gamma_2}$ (residual
    /// transfer).
    CGTolerance interface_mass{1e-13, 1e-11};
    /// CG on the RBF matrix $\Phi$ (every RL-RBF interpolation).
    CGTolerance rbf{1e-12, 1e-10};
  };
} // namespace internodes

#endif // INTERNODES_TYPES_HPP
