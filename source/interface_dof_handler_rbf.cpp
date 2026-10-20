#include "internodes/interface_dof_handler_rbf.hpp"

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparsity_tools.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace internodes
{
  InterfaceDoFHandlerRBF::InterfaceDoFHandlerRBF(
    const std::shared_ptr<const DoFHandler<dim>> &dof_handler,
    const std::shared_ptr<const MeshHandler>     &triangulation,
    const std::set<types::boundary_id>           &interface_id,
    const std::set<types::boundary_id>           &neumann_id,
    const std::set<types::boundary_id>           &dirichlet_id,
    double                                          radius_,
    Mode                                            mode_)
    : InterfaceDoFHandler(dof_handler,
                          triangulation,
                          interface_id,
                          neumann_id,
                          dirichlet_id)
    , radius(radius_)
    , mode(mode_)
    , rtree(std::make_shared<RTreeHandler>(support_points_global_, radius_))
  {
    TimerOutput::Scope timer_section(timer_output(),
                                     "setup: RBF operators (Phi, AMG, scaling)");
    rhs.reinit(this->interface_dofs_owned().size());
    temp_data.reinit(this->interface_dofs_owned().size());
    temp.reinit(this->interface_dofs_owned());

    pcout() << "Assembling RBF sparsity pattern..." << std::endl;
    const unsigned int n_interface(interface_dofs_owned().size());
    DynamicSparsityPattern dsp(n_interface,
                               n_interface,
                               interface_dofs_relevant());

    // interface_dofs_owned() returns a copy: take it once, not per row.
    const IndexSet owned_interface = this->interface_dofs_owned();

    std::vector<types::global_dof_index> columns;
    std::vector<double>                  values;
    for (std::size_t i = 0; i < owned_interface.n_elements(); ++i)
      {
        const unsigned int idx_i = owned_interface.nth_index_in_set(i);
        const auto         results = rtree->query(this->support_points()[i]);
        columns.clear();
        for (const auto &entry : results)
          columns.push_back(entry.second);
        std::sort(columns.begin(), columns.end());
        dsp.add_entries(idx_i, columns.begin(), columns.end(), true);
      }
    dsp.compress();
    SparsityTools::distribute_sparsity_pattern(dsp,
                                                owned_interface,
                                                mpi_comm,
                                                interface_dofs_relevant());
    sp.copy_from(dsp);
    Phi.reinit(owned_interface, sp, mpi_comm);

    pcout() << "Assembling RBF matrix Phi..." << std::endl;
    // One row at a time: the row's nonzeros are the kernel at the support
    // points within the radius -- exactly the columns of the pattern above,
    // so no per-entry sp.exists() (a linear search of the row, O(row^2) per
    // row) and a single insertion per row instead of one per entry.
    for (std::size_t i = 0; i < owned_interface.n_elements(); ++i)
      {
        const Point<dim>   point   = this->support_points()[i];
        const unsigned int idx_i   = owned_interface.nth_index_in_set(i);
        const auto         results = rtree->query(point);
        columns.clear();
        values.clear();
        for (const auto &entry : results)
          {
            Point<dim> node_distance = point;
            node_distance -= this->support_points_global()[entry.second];
            columns.push_back(entry.second);
            values.push_back(evaluate_rbf(node_distance));
          }
        Phi.set(idx_i, columns, values);
      }
    Phi.compress(VectorOperation::insert);

    // Phi solve strategy (experimental, INTERNODES_PHI_SOLVER):
    //   none      CG, no preconditioner. Measured on the real
    //             Test 5 matrices (8 ranks): 113 s vs 345 s (master Phi) and
    //             74 s vs 172 s (slave Phi) per solve against amg -- ML AMG
    //             does not help an RBF Gram matrix (all-positive
    //             off-diagonals, constant unit diagonal), it even raises
    //             the slave's iteration count.
    //   amg       CG + ML AMG, the original/lifex choice
    //   amg_sub   CG + ML AMG with the subdomain-solve settings
    //   scalapack Phi^{-1} computed once as a distributed dense matrix
    //             (ScaLAPACK Cholesky + inverse); each solve is then a
    //             distributed dense mat-vec. Falls back to none if deal.II
    //             was built without ScaLAPACK.
    //   dense     replicated dense Cholesky (LAPACK) on every rank, for
    //             matrices up to INTERNODES_PHI_DENSE_MAX rows (default
    //             12000); falls back to amg above that.
    //   auto      (default) scalapack if deal.II has ScaLAPACK and Phi is
    //             at least INTERNODES_PHI_DENSE_MIN_DENSITY dense (default
    //             0.01), else none. Dense factorization only pays off for
    //             dense-ish Phi (large radius, e.g. Test 5: 8% / 63%); for
    //             small radii Phi is large but very sparse (Tests 2-4:
    //             ~0.05%, up to ~1e5 rows), where a dense factorization
    //             would cost tens of GB and minutes against <1 s of CG.
    const char *const env_solver = std::getenv("INTERNODES_PHI_SOLVER");
    phi_solver = env_solver ? env_solver : "auto";
    if (phi_solver == "auto")
      {
        const char *const env_min = std::getenv("INTERNODES_PHI_DENSE_MIN_DENSITY");
        const double      min_density = env_min ? std::stod(env_min) : 0.01;
        const double      phi_density =
          double(Phi.n_nonzero_elements()) / (double(n_interface) * n_interface);
#ifdef DEAL_II_WITH_SCALAPACK
        phi_solver = (phi_density >= min_density) ? "scalapack" : "none";
#else
        (void)phi_density;
        (void)min_density;
        phi_solver = "none";
#endif
      }
    const char *const env_dense_max = std::getenv("INTERNODES_PHI_DENSE_MAX");
    const unsigned int dense_max = env_dense_max ? std::stoul(env_dense_max) : 12000;
    if (phi_solver == "dense" && n_interface > dense_max)
      phi_solver = "amg";
#ifndef DEAL_II_WITH_SCALAPACK
    if (phi_solver == "scalapack")
      {
        pcout() << "  (deal.II has no ScaLAPACK: Phi solver falls back to none)" << std::endl;
        phi_solver = "none";
      }
#endif
    double phi_rcond = -1.;

    Timer prec_timer(mpi_comm, true);
    if (phi_solver == "dense")
      {
        // Every rank holds all support points and the R-tree, so it can
        // build the whole matrix itself, with no communication.
        Phi_dense.reinit(n_interface, n_interface);
        Vector<double> row(n_interface);
        for (unsigned int i = 0; i < n_interface; ++i)
          {
            const auto results = rtree->query(this->support_points_global()[i]);
            DoFs_values(this->support_points_global()[i], row, results);
            for (const auto &entry : results)
              Phi_dense(i, entry.second) = row(entry.second);
          }
        Phi_dense.set_property(LAPACKSupport::symmetric);
        Phi_dense.compute_cholesky_factorization();
      }
#ifdef DEAL_II_WITH_SCALAPACK
    else if (phi_solver == "scalapack")
      {
        const unsigned int block = 64;
        phi_grid = std::make_shared<Utilities::MPI::ProcessGrid>(
          mpi_comm, n_interface, n_interface, block, block);
        Phi_inv = std::make_unique<ScaLAPACKMatrix<double>>(n_interface,
                                                            phi_grid,
                                                            block,
                                                            LAPACKSupport::symmetric);
        // Each process fills its own block-cyclic block directly from the
        // (replicated) support points: no communication.
        if (phi_grid->is_process_active())
          for (unsigned int lr = 0; lr < Phi_inv->local_m(); ++lr)
            {
              const unsigned int gi = Phi_inv->global_row(lr);
              for (unsigned int lc = 0; lc < Phi_inv->local_n(); ++lc)
                {
                  const unsigned int gj = Phi_inv->global_column(lc);
                  Point<dim> d = this->support_points_global()[gi];
                  d -= this->support_points_global()[gj];
                  // same cut-off as the R-tree query of the sparse Phi
                  Phi_inv->local_el(lr, lc) =
                    (d.norm_square() <= radius * radius) ? evaluate_rbf(d) : 0.;
                }
            }
        const double a_norm = Phi_inv->linfty_norm();
        Phi_inv->compute_cholesky_factorization();
        phi_rcond = Phi_inv->reciprocal_condition_number(a_norm);
        Phi_inv->invert();
        if (phi_grid->is_process_active())
          {
            phi_inv_rows.resize(Phi_inv->local_m());
            phi_inv_cols.resize(Phi_inv->local_n());
            for (unsigned int lr = 0; lr < Phi_inv->local_m(); ++lr)
              phi_inv_rows[lr] = Phi_inv->global_row(lr);
            for (unsigned int lc = 0; lc < Phi_inv->local_n(); ++lc)
              phi_inv_cols[lc] = Phi_inv->global_column(lc);
          }
      }
#endif
    else if (phi_solver == "none")
      phi_preconditioner = std::make_unique<TrilinosWrappers::PreconditionIdentity>();
    else
      {
        auto amg = std::make_unique<TrilinosWrappers::PreconditionAMG>();
        TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
        amg_data.smoother_type = "Chebyshev";
        amg_data.coarse_type   = "Amesos-KLU";
        if (phi_solver == "amg_sub")
          {
            amg_data.higher_order_elements = false;
            amg_data.smoother_sweeps       = 2;
            amg_data.aggregation_threshold = 1e-3;
          }
        else
          {
            amg_data.higher_order_elements = true;
            amg_data.smoother_sweeps       = 1;
            amg_data.aggregation_threshold = 1e-2;
          }
        amg->initialize(Phi, amg_data);
        phi_preconditioner = std::move(amg);
      }
    prec_timer.stop();

    // Rescaling: solve Phi * scaling_factors = 1 (the paper's g(x) = 1 in
    // the rescaled interpolant, Eq. before \eqref{eq:interp}).
    TrilinosWrappers::MPI::Vector ones(this->interface_dofs_owned());
    ones.add(1.);
    scaling_factors.reinit(this->interface_dofs_owned());
    Timer solve_timer(mpi_comm, true);
    const unsigned int its = solve_Phi(scaling_factors, ones, scaling_factor_vectors);
    solve_timer.stop();
    record_cg_solve(cg_tag() + " (scaling)", its);

    const double density = double(Phi.n_nonzero_elements()) / (double(n_interface) * n_interface);
    pcout() << "  Phi: n=" << n_interface << " nnz=" << Phi.n_nonzero_elements()
            << " density=" << density << " solver=" << phi_solver
            << " | setup " << prec_timer.wall_time() << "s"
            << [&] {
                 // own stream: setprecision on pcout would leak into every
                 // later print (it did -- the H1 error lost its digits)
                 std::ostringstream o;
                 if (phi_rcond >= 0.)
                   o << " rcond=" << std::scientific << std::setprecision(2) << phi_rcond;
                 return o.str();
               }()
            << " | scaling solve " << its << " its, " << solve_timer.wall_time() << "s"
            << std::endl;
  }

  unsigned int
  InterfaceDoFHandlerRBF::solve_Phi(TrilinosWrappers::MPI::Vector       &x,
                                    const TrilinosWrappers::MPI::Vector &b,
                                    Vector<double>                      &x_full) const
  {
    if (phi_solver == "dense")
      {
        x_full = b; // gathers the whole right-hand side on every rank
        Phi_dense.solve(x_full);
        for (const auto i : x.locally_owned_elements())
          x(i) = x_full(i);
        x.compress(VectorOperation::insert);
        return 0;
      }
#ifdef DEAL_II_WITH_SCALAPACK
    if (phi_solver == "scalapack")
      {
        // x = Phi^{-1} b with the distributed inverse: each process applies
        // its local block (lower triangle only, mirrored), then one sum.
        x_full = b; // the whole right-hand side on every rank
        Vector<double> y(x_full.size());
        // Column by column (ScaLAPACK local storage is column-major), with the
        // global indices cached at setup.
        if (phi_grid->is_process_active())
          for (unsigned int lc = 0; lc < phi_inv_cols.size(); ++lc)
            {
              const unsigned int gj  = phi_inv_cols[lc];
              const double       xj  = x_full(gj);
              double             acc = 0.;
              for (unsigned int lr = 0; lr < phi_inv_rows.size(); ++lr)
                {
                  const unsigned int gi = phi_inv_rows[lr];
                  if (gi < gj)
                    continue;
                  const double a = Phi_inv->local_el(lr, lc);
                  y(gi) += a * xj;
                  if (gi > gj)
                    acc += a * x_full(gi);
                }
              y(gj) += acc;
            }
        Utilities::MPI::sum(y, mpi_comm, y);
        x_full = y;
        for (const auto i : x.locally_owned_elements())
          x(i) = x_full(i);
        x.compress(VectorOperation::insert);
        return 0;
      }
#endif
    ReductionControl                        control(100000, 1e-12, 1e-10);
    SolverCG<TrilinosWrappers::MPI::Vector> solver(control);
    solver.solve(Phi, x, b, *phi_preconditioner);
    x_full = x;
    return control.last_step();
  }

  double
  InterfaceDoFHandlerRBF::evaluate_rbf(const Point<dim> &distance) const
  {
    const double norm_p_over_r = distance.norm() / radius;

    if (mode == Mode::Wendland)
      return std::pow(std::max(0., 1. - norm_p_over_r), 4) *
             (1 + 4 * norm_p_over_r);
    else if (mode == Mode::Gaussian)
      return std::exp(-norm_p_over_r * norm_p_over_r);
    else // Mode::IMQ
      return 1.0 / (radius * std::sqrt(norm_p_over_r * norm_p_over_r + 1));
  }

  void
  InterfaceDoFHandlerRBF::DoFs_values(
    const Point<dim>                                        &p,
    Vector<double>                                            &rhs_vector,
    const std::vector<std::pair<Point<dim>, unsigned int>> &results) const
  {
    rhs_vector = 0;
    for (const auto &entry : results)
      {
        Point<dim> node_distance = p;
        node_distance -= this->support_points_global()[entry.second];
        rhs_vector[entry.second] = evaluate_rbf(node_distance);
      }
  }

  void
  InterfaceDoFHandlerRBF::setup_destination_points(
    const std::vector<Point<dim>> &points,
    const IndexSet                &destination_owned)
  {
    TimerOutput::Scope timer_section(timer_output(), "Setup destination points");

    destination_points_map.clear();
    destination_owned_ = destination_owned;

    // Unlike the Lagrange base class, a destination point's RBF weights
    // don't depend on which rank's cell contains it -- only on the
    // (globally replicated) source support points and the R-tree over
    // them, which every rank already holds. So each rank computes the
    // weights of exactly the destination points it owns and keeps them,
    // with no communication at all (it used to compute a load-balanced
    // slice of *all* points and then all_gather the result onto every
    // rank -- hundreds of MB per rank at large support radii, and a cost
    // that doesn't shrink with more ranks since the full map has to land
    // everywhere regardless of how many there are).
    for (const types::global_dof_index i : destination_owned_)
      {
        const Point<dim> point   = points[i];
        const auto       results = rtree->query(point);

        for (const auto &entry : results)
          {
            Point<dim> node_distance = point;
            node_distance -= this->support_points_global()[entry.second];
            const double weight = evaluate_rbf(node_distance);
            if (weight > 1e-15)
              {
                destination_points_map[i].first.push_back(entry.second);
                destination_points_map[i].second.push_back(weight);
              }
          }
      }
  }

  void
  InterfaceDoFHandlerRBF::interpolate(TrilinosWrappers::MPI::Vector       &dst,
                                       const TrilinosWrappers::MPI::Vector &src,
                                       const std::vector<Point<dim>> & /*points*/) const
  {
    AssertThrow(dst.locally_owned_elements() == destination_owned_,
                ExcMessage("The destination vector must own exactly the entries "
                           "passed as destination_owned to "
                           "setup_destination_points()."));

    // Step 1: temp = Phi^{-1} * src (the source-side RBF coefficients gamma
    // in the paper's notation).
    {
      TimerOutput::Scope timer_section(timer_output(), "  interpolate: solve Phi");
      record_cg_solve(cg_tag(), solve_Phi(temp, src, temp_data));
    }
    TimerOutput::Scope timer_section(timer_output(), "  interpolate: evaluate at points");

    // Step 2: evaluate the rescaled interpolant at each destination point
    // owned here: value = (weights . gamma) / (weights . scaling_factors),
    // using the sparse weights computed in setup_destination_points()
    // directly (no dense O(n_interface) pass per point).
    for (const types::global_dof_index i : destination_owned_)
      {
        double     value        = 0.;
        double     scale_factor = 0.;
        const auto point_data   = destination_points_map.find(i);
        if (point_data != destination_points_map.end())
          {
            const auto &indices = point_data->second.first;
            const auto &weights = point_data->second.second;
            for (std::size_t k = 0; k < indices.size(); ++k)
              {
                value += weights[k] * temp_data[indices[k]];
                scale_factor += weights[k] * scaling_factor_vectors[indices[k]];
              }
          }
        dst(i) = value / scale_factor;
      }
    dst.compress(VectorOperation::insert);
  }
} // namespace internodes
