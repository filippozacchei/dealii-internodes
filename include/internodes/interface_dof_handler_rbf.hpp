#ifndef INTERNODES_INTERFACE_DOF_HANDLER_RBF_HPP
#define INTERNODES_INTERFACE_DOF_HANDLER_RBF_HPP

#include <deal.II/base/process_grid.h>

#include <deal.II/lac/lapack_full_matrix.h>
#ifdef DEAL_II_WITH_SCALAPACK
#  include <deal.II/lac/scalapack.h>
#endif
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>

#include "internodes/interface_dof_handler.hpp"
#include "internodes/rtree_handler.hpp"

#include <map>
#include <memory>
#include <string>

namespace internodes
{
  /**
   * @brief Interface DoF handler for geometrically non-conforming
   * interfaces, using rescaled localized radial basis function (RL-RBF)
   * interpolation (Deparis et al. 2014) in place of Lagrange interpolation.
   *
   * Implements the $R_{12}$/$R_{21}$ operators of the paper's Eq.
   * \eqref{eq:interp}: given the RBF matrix $\Phi_{kk}$ (entries
   * $\phi_j^{(k)}(x_i^{\Gamma_k})$, assembled here as `Phi`) and its action
   * on the constant function 1 (the `scaling_factors` vector, i.e.
   * $\Phi_{kk}^{-1}\mathbf 1$), interpolate() evaluates
   * $(R_{lk})_{ij} = (\Phi_{lk}\Phi_{kk}^{-1})_{ij} /
   * (\Phi_{lk}\Phi_{kk}^{-1}\mathbf 1)_i$ implicitly, without ever forming
   * $\Phi_{kk}^{-1}$ explicitly.
   */
  class InterfaceDoFHandlerRBF : public InterfaceDoFHandler
  {
  public:
    /// RBF kernel choice; see evaluate_rbf().
    enum class Mode
    {
      Wendland, ///< Wendland $C^2$ compactly-supported kernel (the paper's
                ///< Eq. \eqref{eq:Wendland}; default).
      Gaussian, ///< Gaussian kernel.
      IMQ,      ///< Inverse multiquadric kernel.
    };

    InterfaceDoFHandlerRBF(
      const std::shared_ptr<const DoFHandler<dim>> &dof_handler,
      const std::shared_ptr<const MeshHandler>     &triangulation,
      const std::set<types::boundary_id>           &interface_id,
      const std::set<types::boundary_id>           &neumann_id,
      const std::set<types::boundary_id>           &dirichlet_id,
      double                                          radius,
      Mode                                            mode = Mode::Wendland);

    InterfaceDoFHandlerRBF() : InterfaceDoFHandler() {}

    /// Evaluates the chosen RBF kernel at @p distance (i.e. at $x -
    /// x_j^{\Gamma_k}$ for the node the kernel is centered at), with
    /// support radius `radius`.
    double
    evaluate_rbf(const Point<dim> &distance) const;

    void
    interpolate(TrilinosWrappers::MPI::Vector       &dst,
                const TrilinosWrappers::MPI::Vector &src,
                const std::vector<Point<dim>>       &points) const override;

    /// Computes the RBF weights (sparse, in `destination_points_map`) of the
    /// destination points owned by this rank. Unlike the Lagrange base
    /// class, this needs no geometric search: the weights of a destination
    /// point depend only on the (globally replicated) source support
    /// points, so every rank can compute the weights of the points it owns
    /// on its own, without a global gather of the full map.
    void
    setup_destination_points(const std::vector<Point<dim>> &points,
                             const IndexSet                &destination_owned) override;

    /// Number of nonzeros of the RBF matrix `Phi` (a diagnostic: its density
    /// relative to n_interface^2 is what drives the cost of the Phi solves).
    types::global_dof_index
    phi_n_nonzero_elements() const
    {
      return Phi.n_nonzero_elements();
    }

  private:
    /// Name under which this handler's Phi solves are recorded in
    /// cg_statistics() (by interface size, to tell master from slave).
    std::string
    cg_tag() const
    {
      return "rbf_phi_n" + std::to_string(interface_dofs_owned().size());
    }

    void
    DoFs_values(const Point<dim>                                    &p,
                Vector<double>                                       &rhs_vector,
                const std::vector<std::pair<Point<dim>, unsigned int>> &results) const;

    double radius;
    Mode   mode = Mode::Wendland;

    SparsityPattern                 sp;
    TrilinosWrappers::SparseMatrix  Phi;
    TrilinosWrappers::MPI::Vector   scaling_factors;

    /// CG preconditioner for `Phi`. Back to AMG -- the original choice --
    /// after two cluster-only failures of sparse alternatives tried at
    /// Test 5's r_f=10 (see git history for both): an *exact* factorization
    /// (Amesos) blew up in fill-in and didn't finish in 10+ minutes; an
    /// incomplete one (Ifpack ILU-T) crashed outright
    /// ("Ifpack_LocalFilter.cpp" / "Ifpack_ILUT.cpp", error -1). Root cause
    /// for both, found only after the second failure: at r_f=10, the
    /// *slave* handler's support radius (1.1855, logged as "RBF radii:
    /// ... slave ...") is comparable to the whole interface sphere's
    /// diameter (radius 0.75, so diameter 1.5) -- every point's support
    /// ball covers nearly every other point, so Phi there is not sparse
    /// with a wide bandwidth, it is close to fully *dense*. Every
    /// technique above assumes meaningful sparsity (bounded fill-in,
    /// block-local structure); none of that holds here, which is almost
    /// certainly why both broke specifically on Test 5 and nowhere else.
    /// AMG+CG is slow in this regime (thousands of iterations) but it is
    /// the one approach actually validated at this scale, so it stays
    /// until a fix designed for the genuinely-dense case (e.g. a real
    /// distributed dense factorization, not a sparse one) is validated
    /// off the critical path, not iterated live against cluster jobs.
    TrilinosWrappers::PreconditionAMG preconditioner_Phi; // unused, kept for the doc above

    /// Phi solve strategy actually in use (see the constructor).
    std::string phi_solver;
    /// CG preconditioner for Phi (amg / amg_sub / none strategies).
    std::unique_ptr<TrilinosWrappers::PreconditionBase> phi_preconditioner;
    /// Replicated dense Cholesky factor of Phi ("dense" strategy).
    mutable LAPACKFullMatrix<double> Phi_dense;
#ifdef DEAL_II_WITH_SCALAPACK
    /// "scalapack" strategy: Phi^{-1}, dense, block-cyclically distributed
    /// over all ranks (only its lower triangle is valid after invert()).
    std::shared_ptr<Utilities::MPI::ProcessGrid> phi_grid;
    std::unique_ptr<ScaLAPACKMatrix<double>>     Phi_inv;
#endif

    /// Solves Phi x = b with the selected strategy; also fills @p x_full
    /// with the whole solution on every rank. @return CG iterations (0 for
    /// the direct strategy).
    unsigned int
    solve_Phi(TrilinosWrappers::MPI::Vector       &x,
              const TrilinosWrappers::MPI::Vector &b,
              Vector<double>                      &x_full) const;

    mutable TrilinosWrappers::MPI::Vector temp;
    mutable Vector<double>                temp_data;
    mutable Vector<double>                rhs;

    std::shared_ptr<RTreeHandler> rtree;

    /// Copy of `scaling_factors` as a serial Vector, for fast local dot
    /// products against sparse per-point RBF weight vectors in
    /// interpolate() (avoids repeated distributed-vector indexing).
    Vector<double> scaling_factor_vectors;
  };
} // namespace internodes

#endif // INTERNODES_INTERFACE_DOF_HANDLER_RBF_HPP
