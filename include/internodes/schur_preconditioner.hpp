#ifndef INTERNODES_SCHUR_PRECONDITIONER_HPP
#define INTERNODES_SCHUR_PRECONDITIONER_HPP

#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include "internodes/multi_domain_problem.hpp"

#include <memory>
#include <vector>

namespace internodes
{
  /**
   * @brief Optimal Schur-complement preconditioner $P$ (paper's
   * "Optimal preconditioner" section): applies $P^{-1}$ by solving an
   * auxiliary Neumann-to-Dirichlet problem on the master subdomain, using
   * the *same bilinear form* as the master's own primal problem (via
   * `problem->master->localBilinearForm()`), then extracting the trace of
   * the solution back onto the interface.
   *
   * Note on fidelity to the original: the original lifex-based
   * implementation hardcoded this auxiliary problem's bilinear form as
   * mass + stiffness ($\int \phi_i\phi_j + \nabla\phi_i\cdot\nabla\phi_j$)
   * directly in assembly(), rather than calling through to the master
   * subproblem's own localBilinearForm(). For the paper's actual test
   * problem ($Lu=-\Delta u+u$) these coincide, so it happened to be
   * correct there, but it silently stops matching the claimed "same
   * bilinear form as the primal problem" for any other PDE model. Fixed
   * here to call `problem->master->localBilinearForm()` directly, matching
   * both the paper's stated construction and Subscriptor-style
   * polymorphism already used elsewhere in this codebase.
   */
  class SchurPreconditioner : public Subscriptor
  {
  public:
    void
    initialize(const std::shared_ptr<MultiDomainProblem> &problem);

    void
    vmult(TrilinosWrappers::MPI::Vector       &dst,
          const TrilinosWrappers::MPI::Vector &src) const;

    void
    assembly();

  private:
    std::shared_ptr<MultiDomainProblem> problem;

    TrilinosWrappers::PreconditionAMG preconditioner_problem;

    TrilinosWrappers::SparseMatrix        matrix;
    mutable TrilinosWrappers::MPI::Vector rhs;
    mutable TrilinosWrappers::MPI::Vector dst_global;

    std::vector<types::global_dof_index> owned_indices;
    std::vector<types::global_dof_index> interface_indices;
  };
} // namespace internodes

#endif // INTERNODES_SCHUR_PRECONDITIONER_HPP
