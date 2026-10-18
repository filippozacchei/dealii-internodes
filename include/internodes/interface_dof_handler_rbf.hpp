#ifndef INTERNODES_INTERFACE_DOF_HANDLER_RBF_HPP
#define INTERNODES_INTERFACE_DOF_HANDLER_RBF_HPP

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_solver.h>

#include "internodes/interface_dof_handler.hpp"
#include "internodes/rtree_handler.hpp"

#include <map>
#include <memory>

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

  private:
    void
    DoFs_values(const Point<dim>                                    &p,
                Vector<double>                                       &rhs_vector,
                const std::vector<std::pair<Point<dim>, unsigned int>> &results) const;

    double radius;
    Mode   mode = Mode::Wendland;

    SparsityPattern                 sp;
    TrilinosWrappers::SparseMatrix  Phi;
    TrilinosWrappers::MPI::Vector   scaling_factors;

    /// Direct factorization of `Phi`, computed once in the constructor and
    /// reused (no refactorization) by every interpolate() call. `Phi` is an
    /// RBF Gram matrix, not a discretized elliptic operator, so AMG is a
    /// poor preconditioner for it: at large support radii (e.g. the paper's
    /// r_f=10 regime) CG+AMG needs thousands of iterations per call and
    /// that cost barely improves with more MPI ranks (it's iteration-
    /// latency-bound, not work-bound). `Phi` itself stays modest in size
    /// even at large radius (it is assembled from a 2D interface mesh), so
    /// a one-time sparse direct factorization (gathered internally by
    /// Amesos) is cheap and turns every subsequent solve into a back/
    /// forward substitution. Verified to reproduce CG+AMG's solution to
    /// round-off.
    mutable TrilinosWrappers::SolverDirect solver_Phi;

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
