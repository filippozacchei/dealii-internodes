/**
 * @file coupled_diffusion.cpp
 *
 * Small, reviewer-runnable example: solves the reaction-diffusion problem
 * $-\Delta u + u = f$ on a domain split into two non-conforming subdomains,
 * using the INTERNODES method, and compares against the manufactured exact
 * solution $u_{ex}(x,y,z) = e^{x+y+z}$ (the same problem used in the
 * paper's Sect. "Comparison with Monodomain").
 *
 * Geometry, boundary conditions, discretization, and interpolation are all
 * read from a parameter file (see coupled_diffusion.prm for the default,
 * self-documenting one) -- nothing here needs recompiling to try a
 * different configuration.
 */
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include "internodes/internodes_schur_complement.hpp"
#include "internodes/multi_domain_problem.hpp"
#include "internodes/sub_problem_diffusion_reaction.hpp"
#include "internodes/utilities.hpp"

#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace internodes;

// =========================================================================
// Manufactured solution: u_ex(x,y,z) = e^{x+y+z}, so that
//   Lu = -Delta u + u = -3u + u = -2u  =>  f = -2 u_ex.
// =========================================================================

class ExactSolution : public Function<dim>
{
public:
  double
  value(const Point<dim> &p, unsigned int = 0) const override
  {
    return std::exp(p[0] + p[1] + p[2]);
  }

  Tensor<1, dim>
  gradient(const Point<dim> &p, unsigned int = 0) const override
  {
    const double        u = value(p);
    Tensor<1, dim> grad;
    for (unsigned int d = 0; d < dim; ++d)
      grad[d] = u;
    return grad;
  }
};

class Forcing : public Function<dim>
{
public:
  double
  value(const Point<dim> &p, unsigned int = 0) const override
  {
    return -2.0 * std::exp(p[0] + p[1] + p[2]);
  }
};

/// Neumann data: assembly() calls fun_neumann->gradient(x) . normal, so this
/// need only implement gradient() -- it matches the exact solution's own
/// gradient, giving the exact co-normal derivative g_N = grad(u_ex) . n.
class NeumannData : public Function<dim>
{
public:
  Tensor<1, dim>
  gradient(const Point<dim> &p, unsigned int = 0) const override
  {
    return ExactSolution().gradient(p);
  }
};

// =========================================================================
// Parameters
// =========================================================================

enum class GeometryType
{
  adjacent_boxes,
  half_hyper_shells
};

struct Parameters
{
  GeometryType geometry = GeometryType::adjacent_boxes;

  // -- adjacent_boxes --
  // Omega = (-2,2) x (-1,1) x (-1,1), split at x=0 into Omega_1=(-2,0)x...
  // (master) and Omega_2=(0,2)x... (slave), matching the paper's Sect. 4.1
  // setup exactly.
  std::vector<unsigned int> subdivisions_master = {8, 4, 4};
  std::vector<unsigned int> subdivisions_slave  = {8, 4, 4};

  // -- half_hyper_shells --
  double inner_radius     = 0.5;
  double interface_radius = 0.75;
  double outer_radius     = 1.0;
  unsigned int shell_refinement_master = 2;
  unsigned int shell_refinement_slave  = 2;

  unsigned int master_degree = 1;
  unsigned int slave_degree  = 1;

  /// RBF support radius for the slave's interface handler; 0 => Lagrange
  /// interpolation (only valid/meaningful for geometrically conforming
  /// interfaces, i.e. adjacent_boxes without a deliberate mismatch).
  double rbf_radius = 0.0;

  std::set<types::boundary_id> dirichlet_ids_master;
  std::set<types::boundary_id> neumann_ids_master;
  std::set<types::boundary_id> interface_id_master;
  std::set<types::boundary_id> dirichlet_ids_slave;
  std::set<types::boundary_id> neumann_ids_slave;
  std::set<types::boundary_id> interface_id_slave;

  double gmres_tolerance    = 1e-8;
  unsigned int gmres_max_it = 1000;

  static std::set<types::boundary_id>
  parse_ids(const std::string &s)
  {
    std::set<types::boundary_id> result;
    for (const auto &token : Utilities::split_string_list(s))
      if (!token.empty())
        result.insert(static_cast<types::boundary_id>(Utilities::string_to_int(token)));
    return result;
  }

  static std::vector<unsigned int>
  parse_subdivisions(const std::string &s)
  {
    std::vector<unsigned int> result;
    for (const auto &token : Utilities::split_string_list(s))
      result.push_back(static_cast<unsigned int>(Utilities::string_to_int(token)));
    if (result.size() != dim)
      throw std::runtime_error("Expected " + std::to_string(dim) +
                               " comma-separated subdivision counts, got " +
                               std::to_string(result.size()));
    return result;
  }

  void
  declare(ParameterHandler &prm)
  {
    prm.enter_subsection("Geometry");
    {
      prm.declare_entry("Type",
                        "adjacent_boxes",
                        Patterns::Selection("adjacent_boxes|half_hyper_shells"),
                        "Which pair of subdomains to build: two adjacent "
                        "boxes with a flat interface (matches the paper's "
                        "Geometry-A), or two half hyper-shells with a "
                        "curved, geometrically non-conforming interface "
                        "(matches the paper's Geometry-B).");
      prm.declare_entry("Subdivisions master",
                        "8,4,4",
                        Patterns::List(Patterns::Integer(1), dim, dim, ","),
                        "[adjacent_boxes only] Number of mesh subdivisions "
                        "per direction on the master box (-2,0)x(-1,1)x(-1,1).");
      prm.declare_entry("Subdivisions slave",
                        "8,4,4",
                        Patterns::List(Patterns::Integer(1), dim, dim, ","),
                        "[adjacent_boxes only] Same, for the slave box "
                        "(0,2)x(-1,1)x(-1,1). Use different values from "
                        "the master to get a discretization-non-conforming "
                        "interface.");
      prm.declare_entry("Inner radius", "0.5", Patterns::Double(0));
      prm.declare_entry("Interface radius", "0.75", Patterns::Double(0));
      prm.declare_entry("Outer radius", "1.0", Patterns::Double(0));
      prm.declare_entry("Shell refinement master", "2", Patterns::Integer(0));
      prm.declare_entry("Shell refinement slave", "2", Patterns::Integer(0));
    }
    prm.leave_subsection();

    prm.enter_subsection("Discretization");
    {
      prm.declare_entry("Master degree", "1", Patterns::Integer(1));
      prm.declare_entry("Slave degree", "1", Patterns::Integer(1));
      prm.declare_entry("RBF radius",
                        "0.0",
                        Patterns::Double(0),
                        "0 selects Lagrange interpolation (geometrically "
                        "conforming interfaces only); >0 selects RL-RBF "
                        "interpolation with this support radius.");
    }
    prm.leave_subsection();

    // Default boundary ids below match deal.II's colorization for each
    // geometry (see build_*_geometry() for what each id means); override
    // freely to try a different Dirichlet/Neumann split, exactly as the
    // paper's Table with the four accuracy-test configurations does.
    prm.enter_subsection("Boundary conditions");
    {
      prm.declare_entry("Dirichlet ids master", "0", Patterns::List(Patterns::Integer()));
      prm.declare_entry("Neumann ids master", "2,3,4,5", Patterns::List(Patterns::Integer()));
      prm.declare_entry("Interface id master", "1", Patterns::List(Patterns::Integer()));
      prm.declare_entry("Dirichlet ids slave", "1", Patterns::List(Patterns::Integer()));
      prm.declare_entry("Neumann ids slave", "2,3,4,5", Patterns::List(Patterns::Integer()));
      prm.declare_entry("Interface id slave", "0", Patterns::List(Patterns::Integer()));
    }
    prm.leave_subsection();

    prm.enter_subsection("Solver");
    {
      prm.declare_entry("GMRES tolerance", "1e-8", Patterns::Double(0));
      prm.declare_entry("GMRES max iterations", "1000", Patterns::Integer(1));
    }
    prm.leave_subsection();
  }

  void
  parse(ParameterHandler &prm)
  {
    prm.enter_subsection("Geometry");
    {
      geometry = prm.get("Type") == "half_hyper_shells" ?
                   GeometryType::half_hyper_shells :
                   GeometryType::adjacent_boxes;
      subdivisions_master   = parse_subdivisions(prm.get("Subdivisions master"));
      subdivisions_slave    = parse_subdivisions(prm.get("Subdivisions slave"));
      inner_radius          = prm.get_double("Inner radius");
      interface_radius      = prm.get_double("Interface radius");
      outer_radius          = prm.get_double("Outer radius");
      shell_refinement_master = prm.get_integer("Shell refinement master");
      shell_refinement_slave  = prm.get_integer("Shell refinement slave");
    }
    prm.leave_subsection();

    prm.enter_subsection("Discretization");
    {
      master_degree = prm.get_integer("Master degree");
      slave_degree  = prm.get_integer("Slave degree");
      rbf_radius    = prm.get_double("RBF radius");
    }
    prm.leave_subsection();

    prm.enter_subsection("Boundary conditions");
    {
      dirichlet_ids_master = parse_ids(prm.get("Dirichlet ids master"));
      neumann_ids_master   = parse_ids(prm.get("Neumann ids master"));
      interface_id_master  = parse_ids(prm.get("Interface id master"));
      dirichlet_ids_slave  = parse_ids(prm.get("Dirichlet ids slave"));
      neumann_ids_slave    = parse_ids(prm.get("Neumann ids slave"));
      interface_id_slave   = parse_ids(prm.get("Interface id slave"));
    }
    prm.leave_subsection();

    prm.enter_subsection("Solver");
    {
      gmres_tolerance = prm.get_double("GMRES tolerance");
      gmres_max_it    = prm.get_integer("GMRES max iterations");
    }
    prm.leave_subsection();
  }
};

// =========================================================================
// Geometry construction
// =========================================================================

/// Geometry-A: Omega split at x=0 into two boxes. Colorized boundary ids
/// (deal.II's subdivided_hyper_rectangle convention, 3D): 0/1 = -x/+x faces,
/// 2/3 = -y/+y, 4/5 = -z/+z. The shared interface is the master's +x face
/// (id 1) and the slave's -x face (id 0) -- hence the default "Interface
/// id master = 1" / "Interface id slave = 0" above.
void
build_adjacent_boxes(MeshHandler                     &mesh,
                     bool                              is_master,
                     const std::vector<unsigned int> &subdivisions)
{
  const Point<dim> p1 = is_master ? Point<dim>(-2, -1, -1) : Point<dim>(0, -1, -1);
  const Point<dim> p2 = is_master ? Point<dim>(0, 1, 1) : Point<dim>(2, 1, 1);
  GridGenerator::subdivided_hyper_rectangle(
    mesh.get(), subdivisions, p1, p2, /* colorize = */ true);
}

/// Geometry-B: two half hyper-shells (annuli split by a plane through the
/// origin), sharing the curved interface at radius `interface_radius`.
/// deal.II's half_hyper_shell colorizes: 0 = inner surface, 1 = outer
/// surface, 2 = the flat cut face(s). Since master/slave are independently
/// meshed half-shells of different radial extent, the shared spherical
/// interface is geometrically non-conforming even though it is
/// mathematically the same sphere -- this is the paper's Geometry-B case,
/// and requires RBF interpolation (RBF radius > 0), not Lagrange.
void
build_half_hyper_shells(MeshHandler        &mesh,
                        bool                is_master,
                        double              inner_radius,
                        double              interface_radius,
                        double              outer_radius,
                        unsigned int        refinement)
{
  const double r_min = is_master ? inner_radius : interface_radius;
  const double r_max = is_master ? interface_radius : outer_radius;

  GridGenerator::half_hyper_shell(
    mesh.get(), Point<dim>(), r_min, r_max, /* n_cells = */ 0, /* colorize = */ true);
  mesh.get().refine_global(refinement);
}

// =========================================================================
// main
// =========================================================================

int
main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      const std::string prm_file = (argc > 1) ? argv[1] : "coupled_diffusion.prm";

      Parameters       parameters;
      ParameterHandler prm;
      parameters.declare(prm);
      prm.parse_input(prm_file);
      parameters.parse(prm);

      pcout() << "=== dealii-internodes: coupled_diffusion example ===" << std::endl;
      pcout() << "Parameter file: " << prm_file << std::endl;

      auto mesh_master = std::make_shared<MeshHandler>();
      auto mesh_slave  = std::make_shared<MeshHandler>();

      if (parameters.geometry == GeometryType::adjacent_boxes)
        {
          build_adjacent_boxes(*mesh_master, true, parameters.subdivisions_master);
          build_adjacent_boxes(*mesh_slave, false, parameters.subdivisions_slave);
        }
      else
        {
          build_half_hyper_shells(*mesh_master,
                                  true,
                                  parameters.inner_radius,
                                  parameters.interface_radius,
                                  parameters.outer_radius,
                                  parameters.shell_refinement_master);
          build_half_hyper_shells(*mesh_slave,
                                  false,
                                  parameters.inner_radius,
                                  parameters.interface_radius,
                                  parameters.outer_radius,
                                  parameters.shell_refinement_slave);
        }

      const auto forcing_term = std::make_shared<Forcing>();
      const auto dirichlet    = std::make_shared<ExactSolution>();
      const auto neumann      = std::make_shared<NeumannData>();

      const std::map<std::string, std::set<types::boundary_id>> boundary_tags_master = {
        {"Dirichlet", parameters.dirichlet_ids_master},
        {"Neumann", parameters.neumann_ids_master},
        {"Interface", parameters.interface_id_master}};
      const std::map<std::string, std::set<types::boundary_id>> boundary_tags_slave = {
        {"Dirichlet", parameters.dirichlet_ids_slave},
        {"Neumann", parameters.neumann_ids_slave},
        {"Interface", parameters.interface_id_slave}};

      const std::map<std::string, double> coefficients; // unused by this PDE model

      // Both master and slave need their own RBF-capable interface handler
      // when the interface is non-conforming: InternodesSchurComplement's
      // vmult() interpolates *from* the master's own handler *onto* the
      // slave's points (Q21) and vice versa via MultiDomainProblem's
      // interpolateResidual() (Q12) -- each side is a "source" for one
      // transfer direction, so each needs its own RBF machinery, not just
      // one of them. The original scales each side's radius off the
      // *other* side's mesh diameter (see main.cpp's r_master/r_slave);
      // simplified here to a single shared radius rather than replicating
      // that per-side heuristic, since diameter-based auto-scaling isn't
      // needed for a small illustrative example -- users who want it can
      // set "RBF radius" relative to their own mesh spacing directly.
      pcout() << "Building master subproblem..." << std::endl;
      auto master = std::make_shared<SubProblemDiffusionReaction>(
        mesh_master,
        parameters.master_degree,
        dirichlet,
        neumann,
        forcing_term,
        boundary_tags_master,
        coefficients,
        parameters.rbf_radius);

      pcout() << "Building slave subproblem..." << std::endl;
      auto slave = std::make_shared<SubProblemDiffusionReaction>(mesh_slave,
                                                                 parameters.slave_degree,
                                                                 dirichlet,
                                                                 neumann,
                                                                 forcing_term,
                                                                 boundary_tags_slave,
                                                                 coefficients,
                                                                 parameters.rbf_radius);

      pcout() << "Coupling master and slave..." << std::endl;
      auto problem = std::make_shared<MultiDomainProblem>(master, slave);

      SolverControl solver_control(parameters.gmres_max_it, parameters.gmres_tolerance);
      InternodesSchurComplement solver(problem, solver_control);

      pcout() << "Solving..." << std::endl;
      solver.solve();
      pcout() << "GMRES converged in " << solver.get_n_iterations() << " iterations."
               << std::endl;

      // Broken H1-norm error against the exact solution, matching the
      // paper's own accuracy metric (Sect. 4.1). solver.sol_master()/
      // sol_slave() are indexed by the *internal-block-local* numbering
      // (0..n_internal-1, from the internal/interface split in
      // SubProblemBase::setupSystem()), not the DoFHandler's own global
      // numbering that VectorTools::integrate_difference expects -- so the
      // internal and interface pieces first need reassembling into one
      // full, DoFHandler-indexed vector via internal_global_dof()/
      // interface_global_dof(), the same maps used throughout the solver
      // itself to move between the two numberings.
      auto reconstruct_full_solution =
        [](const SubProblemBase                &sub,
           const TrilinosWrappers::MPI::Vector &internal_part,
           const TrilinosWrappers::MPI::Vector &interface_part) {
          TrilinosWrappers::MPI::Vector full(sub.owned_dofs, mpi_comm);

          const auto &idh            = *sub.interface_dofHandler_ptr;
          const IndexSet internal_owned  = idh.internal_dofs_owned();
          const IndexSet interface_owned = idh.interface_dofs_owned();

          for (auto it = internal_owned.begin(); it != internal_owned.end(); ++it)
            full[idh.internal_global_dof(*it)] = internal_part[*it];
          for (auto it = interface_owned.begin(); it != interface_owned.end(); ++it)
            full[idh.interface_global_dof(*it)] = interface_part[*it];
          full.compress(VectorOperation::insert);

          // integrate_difference() needs ghost values for DoFs on locally
          // owned cells that are owned by a neighboring rank.
          TrilinosWrappers::MPI::Vector full_ghosted(sub.owned_dofs,
                                                     sub.relevant_dofs,
                                                     mpi_comm);
          full_ghosted = full;
          return full_ghosted;
        };

      const TrilinosWrappers::MPI::Vector solution_master =
        reconstruct_full_solution(*master, solver.sol_master(), solver.lambda_master_());
      const TrilinosWrappers::MPI::Vector solution_slave =
        reconstruct_full_solution(*slave, solver.sol_slave(), solver.lambda_slave_());

      double error_squared = 0.0;
      for (const auto &[sub, sol, degree] :
           {std::tuple{master.get(), &solution_master, parameters.master_degree},
            std::tuple{slave.get(), &solution_slave, parameters.slave_degree}})
        {
          Vector<double> difference_per_cell(sub->slice->get().n_active_cells());
          VectorTools::integrate_difference(*sub->dof_handler,
                                            *sol,
                                            ExactSolution(),
                                            difference_per_cell,
                                            QGauss<dim>(degree + 2),
                                            VectorTools::H1_norm);
          const double local_error =
            VectorTools::compute_global_error(sub->slice->get(),
                                              difference_per_cell,
                                              VectorTools::H1_norm);
          error_squared += local_error * local_error;
        }

      pcout() << "Broken H1 error vs. exact solution: " << std::sqrt(error_squared)
               << std::endl;
    }
  catch (const std::exception &exc)
    {
      std::cerr << "Exception: " << exc.what() << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << "Unknown exception." << std::endl;
      return 1;
    }

  return 0;
}
