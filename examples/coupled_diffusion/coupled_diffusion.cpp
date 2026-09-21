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
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/vector_tools.h>

#include "internodes/internodes_schur_complement.hpp"
#include "internodes/multi_domain_problem.hpp"
#include "internodes/sub_problem_diffusion_reaction.hpp"
#include "internodes/utilities.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
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
  // Coarse mesh of each box, refined globally afterwards (see
  // MeshHandler::create()): cells per direction = subdivisions * 2^refinement.
  std::vector<unsigned int> subdivisions_master = {8, 4, 4};
  std::vector<unsigned int> subdivisions_slave  = {8, 4, 4};
  unsigned int global_refinement_master = 0;
  unsigned int global_refinement_slave  = 0;

  // -- half_hyper_shells --
  double inner_radius     = 0.5;
  double interface_radius = 0.75;
  double outer_radius     = 1.0;
  unsigned int shell_refinement_master = 2;
  unsigned int shell_refinement_slave  = 2;

  unsigned int master_degree = 1;
  unsigned int slave_degree  = 1;

  /// Absolute RBF support radius; 0 => Lagrange interpolation (only
  /// valid/meaningful for geometrically conforming interfaces, i.e.
  /// adjacent_boxes without a deliberate mismatch).
  double rbf_radius = 0.0;

  /// If > 0, overrides rbf_radius: each subdomain's radius is
  /// rbf_radius_factor * (average cell diameter of that subdomain's mesh),
  /// i.e. r = r_f h as in the paper (and lifex's "RBF radius scaling factor").
  double rbf_radius_factor = 0.0;

  std::set<types::boundary_id> dirichlet_ids_master;
  std::set<types::boundary_id> neumann_ids_master;
  std::set<types::boundary_id> interface_id_master;
  std::set<types::boundary_id> dirichlet_ids_slave;
  std::set<types::boundary_id> neumann_ids_slave;
  std::set<types::boundary_id> interface_id_slave;

  double gmres_tolerance    = 1e-8;
  double gmres_reduction    = 0.;
  bool gmres_right_preconditioning = true;
  unsigned int gmres_basis_size    = 1000;
  unsigned int gmres_max_it = 1000;

  /// [adjacent_boxes] Cell type of each subdomain: hexahedra, or tetrahedra
  /// obtained by splitting the hexahedra of the same (refined) box.
  bool tets_master = false;
  bool tets_slave  = false;
  bool use_schur_preconditioner = true;

  /// "coupled": the INTERNODES solve of the two subdomains. "monolithic": a
  /// single-domain reference solve on the union of the two boxes (adjacent_
  /// boxes only), at the master's resolution.
  bool monolithic = false;

  /// If not empty, a JSON file with sizes, errors, iterations and per-phase
  /// timings is written here (by rank 0).
  std::string results_file;

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
                        "[adjacent_boxes only] Number of subdivisions per "
                        "direction of the *coarse* mesh of the master box "
                        "(-2,0)x(-1,1)x(-1,1); it is then refined "
                        "'Global refinement master' times.");
      prm.declare_entry("Subdivisions slave",
                        "8,4,4",
                        Patterns::List(Patterns::Integer(1), dim, dim, ","),
                        "[adjacent_boxes only] Same, for the slave box "
                        "(0,2)x(-1,1)x(-1,1). Use different resulting "
                        "resolutions on master and slave to get a "
                        "discretization-non-conforming interface.");
      prm.declare_entry("Cell type master",
                        "hex",
                        Patterns::Selection("hex|tet"),
                        "[adjacent_boxes only] hex, or tet: the refined "
                        "hexahedral box split into tetrahedra.");
      prm.declare_entry("Cell type slave",
                        "hex",
                        Patterns::Selection("hex|tet"),
                        "[adjacent_boxes only] Same, for the slave.");
      prm.declare_entry("Global refinement master",
                        "0",
                        Patterns::Integer(0),
                        "[adjacent_boxes only] Uniform refinements applied "
                        "to the distributed master mesh (cells per direction "
                        "= subdivisions * 2^refinements).");
      prm.declare_entry("Global refinement slave",
                        "0",
                        Patterns::Integer(0),
                        "[adjacent_boxes only] Same, for the slave mesh.");
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
                        "Absolute support radius. 0 selects Lagrange "
                        "interpolation (geometrically conforming interfaces "
                        "only); >0 selects RL-RBF interpolation.");
      prm.declare_entry("RBF radius factor",
                        "0.0",
                        Patterns::Double(0),
                        "If >0, selects RL-RBF interpolation with a support "
                        "radius r = factor * h_avg on each subdomain, h_avg "
                        "being the average cell diameter of that subdomain's "
                        "mesh (the paper's r = r_f h). Overrides 'RBF radius'.");
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
      prm.declare_entry("GMRES right preconditioning",
                        "true",
                        Patterns::Bool(),
                        "Right (true, as in lifex) or left preconditioning.");
      prm.declare_entry("GMRES basis size",
                        "1000",
                        Patterns::Integer(3),
                        "Number of Krylov vectors kept before GMRES restarts "
                        "(the default, as in lifex, effectively never restarts).");
      prm.declare_entry("GMRES reduction",
                        "0",
                        Patterns::Double(0),
                        "If >0, GMRES also stops when the residual has been "
                        "reduced by this factor relative to the initial one "
                        "(the tolerance above stays an absolute threshold; "
                        "set it tiny to stop on the reduction only).");
      prm.declare_entry("Use Schur preconditioner",
                        "true",
                        Patterns::Bool(),
                        "Use the Dirichlet-Neumann-type preconditioner in "
                        "the interface GMRES solve; false gives the "
                        "unpreconditioned iteration counts.");
    }
    prm.leave_subsection();

    prm.enter_subsection("Run");
    {
      prm.declare_entry("Mode",
                        "coupled",
                        Patterns::Selection("coupled|monolithic"),
                        "coupled: INTERNODES on two subdomains. monolithic: "
                        "single-domain reference solve on the union of the "
                        "two boxes at the master's resolution (adjacent_boxes "
                        "with the default colorized boundary ids only).");
      prm.declare_entry("Results file",
                        "",
                        Patterns::Anything(),
                        "If not empty, a JSON file with sizes, errors, "
                        "iteration counts and per-phase timings.");
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
      tets_master = prm.get("Cell type master") == "tet";
      tets_slave  = prm.get("Cell type slave") == "tet";
      global_refinement_master = prm.get_integer("Global refinement master");
      global_refinement_slave  = prm.get_integer("Global refinement slave");
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
      rbf_radius_factor = prm.get_double("RBF radius factor");
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
      gmres_reduction = prm.get_double("GMRES reduction");
      gmres_right_preconditioning = prm.get_bool("GMRES right preconditioning");
      gmres_basis_size            = prm.get_integer("GMRES basis size");
      use_schur_preconditioner = prm.get_bool("Use Schur preconditioner");
    }
    prm.leave_subsection();

    prm.enter_subsection("Run");
    {
      monolithic   = prm.get("Mode") == "monolithic";
      results_file = prm.get("Results file");
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
///
/// Like lifex, only a coarse mesh is built here (identically on every rank);
/// MeshHandler::create() distributes it and refines it globally on the
/// distributed triangulation, which is what scales to large meshes.
void
build_adjacent_boxes(MeshHandler                     &mesh,
                     bool                              is_master,
                     const std::vector<unsigned int> &subdivisions,
                     const unsigned int                n_refinements,
                     const bool                        tetrahedra = false)
{
  const Point<dim> p1 = is_master ? Point<dim>(-2, -1, -1) : Point<dim>(0, -1, -1);
  const Point<dim> p2 = is_master ? Point<dim>(0, 1, 1) : Point<dim>(2, 1, 1);
  Triangulation<dim> serial_tria;
  GridGenerator::subdivided_hyper_rectangle(
    serial_tria, subdivisions, p1, p2, /* colorize = */ true);
  if (!tetrahedra)
    {
      mesh.create(serial_tria, n_refinements);
      return;
    }

  // deal.II cannot refine simplices: refine the hexahedra, then split them
  // into tetrahedra (boundary ids are preserved).
  serial_tria.refine_global(n_refinements);
  Triangulation<dim> simplex_tria;
  GridGenerator::convert_hypercube_to_simplex_mesh(serial_tria, simplex_tria);
  mesh.create(simplex_tria);
}

/// Geometry-B: two half hyper-shells (annuli split by a plane through the
/// origin), sharing the curved interface at radius `interface_radius`.
/// deal.II's half_hyper_shell colorizes: 0 = inner surface, 1 = outer
/// surface, 2 = the flat cut face(s). Since master/slave are independently
/// meshed half-shells of different radial extent, the shared spherical
/// interface is geometrically non-conforming even though it is
/// mathematically the same sphere -- this is the paper's Geometry-B case,
/// and requires RBF interpolation (RBF radius > 0), not Lagrange.
///
/// The paper's actual Geometry-B meshes are *tetrahedral*, generated
/// externally with Gmsh, not the hexahedral cells GridGenerator::
/// half_hyper_shell produces directly. Lacking the original Gmsh files,
/// this builds the hex half-shell on a temporary serial triangulation,
/// converts it to a genuinely tetrahedral mesh via
/// GridGenerator::convert_hypercube_to_simplex_mesh() (which preserves
/// boundary ids across the conversion), and only then distributes it into
/// the parallel::distributed::Triangulation MeshHandler owns -- so the
/// *cell type* matches the paper even though the exact mesh (vertex
/// positions, cell count) does not.
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

  Triangulation<dim> hex_tria;
  GridGenerator::half_hyper_shell(
    hex_tria, Point<dim>(), r_min, r_max, /* n_cells = */ 0, /* colorize = */ true);
  hex_tria.refine_global(refinement);

  Triangulation<dim> simplex_tria;
  GridGenerator::convert_hypercube_to_simplex_mesh(hex_tria, simplex_tria);

  // The hex half-shell carries a SphericalManifold (used above to place the
  // refined vertices on the sphere); the conversion keeps the manifold *ids*
  // on the cells but not the manifold object. The paper's Gmsh tetrahedra
  // are plain straight-sided linear tets, so drop to flat manifolds -- the
  // curved-vertex placement has already been done by the refinement.
  // (reset_all_manifolds() alone only removes the manifold *objects*; the
  // manifold *ids* on cells/faces would still refer to the removed
  // SphericalManifold, so they must be reset to flat explicitly too.)
  simplex_tria.set_all_manifold_ids(numbers::flat_manifold_id);
  simplex_tria.reset_all_manifolds();

  mesh.create(simplex_tria);
}

// =========================================================================
// Reports (JSON) and single-domain reference solve
// =========================================================================

/// Sizes and error of one subdomain (or of the single-domain reference).
struct SubdomainReport
{
  unsigned int             degree           = 0;
  types::global_dof_index  n_dofs           = 0;
  types::global_dof_index  n_interface_dofs = 0;
  types::global_cell_index n_cells          = 0;
  double                   h_avg            = 0.;
  double                   rbf_radius       = 0.;
  double                   error            = 0.;
};

std::string
json_escape(const std::string &s)
{
  std::string out;
  for (const char c : s)
    {
      if (c == '"' || c == '\\')
        out += '\\';
      out += c;
    }
  return out;
}

/// Per-phase timings as a JSON object {name: {"wall_max": s, "calls": n}}.
/// The wall time is the maximum over the MPI ranks, which is what a strong
/// scaling study reports. Collective.
std::string
timings_json()
{
  const auto wall =
    timer_output().get_summary_data(TimerOutput::OutputData::total_wall_time);
  const auto calls =
    timer_output().get_summary_data(TimerOutput::OutputData::n_calls);

  std::vector<double> values;
  for (const auto &entry : wall)
    values.push_back(entry.second);

  // All ranks execute the same (collective) timer scopes, so the sections
  // agree; fall back to rank 0's own numbers if they somehow do not.
  const unsigned int n = values.size();
  if (Utilities::MPI::min(n, mpi_comm) == Utilities::MPI::max(n, mpi_comm))
    {
      std::vector<double> maxima(n);
      MPI_Allreduce(values.data(), maxima.data(), n, MPI_DOUBLE, MPI_MAX, mpi_comm);
      values = maxima;
    }

  std::ostringstream out;
  out.precision(9);
  out << "{";
  unsigned int i = 0;
  for (const auto &entry : wall)
    {
      std::string name = entry.first;
      name.erase(0, name.find_first_not_of(' '));
      out << (i ? ", " : "") << "\n    \"" << json_escape(name)
          << "\": {\"wall_max\": " << values[i]
          << ", \"calls\": " << calls.at(entry.first) << "}";
      ++i;
    }
  out << "\n  }";
  return out.str();
}

void
write_results(const Parameters                                              &parameters,
              const std::vector<std::pair<std::string, SubdomainReport>>    &subdomains,
              const double                                                   total_error,
              const int                                                      n_iterations)
{
  if (parameters.results_file.empty())
    return;

  const std::string timings = timings_json(); // collective: call on all ranks
  if (Utilities::MPI::this_mpi_process(mpi_comm) != 0)
    return;

  std::ofstream out(parameters.results_file);
  AssertThrow(out, ExcMessage("Cannot write results file " + parameters.results_file));
  out.precision(12);
  out << "{\n";
  out << "  \"mode\": \"" << (parameters.monolithic ? "monolithic" : "coupled") << "\",\n";
  out << "  \"geometry\": \""
      << (parameters.geometry == GeometryType::adjacent_boxes ? "adjacent_boxes" :
                                                                 "half_hyper_shells")
      << "\",\n";
  out << "  \"n_mpi_ranks\": " << Utilities::MPI::n_mpi_processes(mpi_comm) << ",\n";
  out << "  \"interpolation\": \""
      << ((parameters.rbf_radius > 0. || parameters.rbf_radius_factor > 0.) ? "rbf" :
                                                                              "lagrange")
      << "\",\n";
  out << "  \"rbf_radius_factor\": " << parameters.rbf_radius_factor << ",\n";
  out << "  \"schur_preconditioner\": "
      << (parameters.use_schur_preconditioner ? "true" : "false") << ",\n";
  out << "  \"gmres_iterations\": " << n_iterations << ",\n";
  out << "  \"broken_H1_error\": " << total_error << ",\n";
  out << "  \"subdomains\": {";
  unsigned int i = 0;
  for (const auto &[name, r] : subdomains)
    {
      out << (i++ ? "," : "") << "\n    \"" << name << "\": {"
          << "\"degree\": " << r.degree << ", \"n_dofs\": " << r.n_dofs
          << ", \"n_interface_dofs\": " << r.n_interface_dofs
          << ", \"n_cells\": " << r.n_cells << ", \"h_avg\": " << r.h_avg
          << ", \"rbf_radius\": " << r.rbf_radius << ", \"H1_error\": " << r.error
          << "}";
    }
  out << "\n  },\n";
  out << "  \"timings\": " << timings << "\n}\n";
}

/// Broken/plain H1 error of a DoFHandler-indexed (ghosted) solution against
/// the exact solution, using the cell-type-aware mapping/quadrature of @p mesh.
double
h1_error(const MeshHandler                     &mesh,
         const DoFHandler<dim>                 &dof_handler,
         const TrilinosWrappers::MPI::Vector   &ghosted_solution,
         const unsigned int                     degree)
{
  const auto mapping    = mesh.get_linear_mapping();
  const auto quadrature = mesh.get_quadrature_gauss(degree + 2);

  Vector<double> difference_per_cell(mesh.get().n_active_cells());
  VectorTools::integrate_difference(*mapping,
                                    dof_handler,
                                    ghosted_solution,
                                    ExactSolution(),
                                    difference_per_cell,
                                    *quadrature,
                                    VectorTools::H1_norm);
  return VectorTools::compute_global_error(mesh.get(),
                                           difference_per_cell,
                                           VectorTools::H1_norm);
}

/// Single-domain reference solve of the same problem on the union of the two
/// boxes, (-2,2)x(-1,1)x(-1,1), at the master's resolution (so, for a
/// conforming coupled run, on exactly the same mesh). Dirichlet/Neumann ids
/// are the unions of the master's and slave's, i.e. deal.II's colorized ids
/// with the paper's default assignment. Plain CG with algebraic multigrid.
SubdomainReport
run_monolithic(const Parameters &parameters, unsigned int &n_cg_iterations)
{
  MeshHandler mesh;
  {
    Triangulation<dim>        coarse;
    std::vector<unsigned int> subdivisions = parameters.subdivisions_master;
    subdivisions[0] *= 2; // two boxes side by side along x
    GridGenerator::subdivided_hyper_rectangle(
      coarse, subdivisions, Point<dim>(-2, -1, -1), Point<dim>(2, 1, 1), true);
    mesh.create(coarse, parameters.global_refinement_master);
  }

  std::set<types::boundary_id> dirichlet_ids = parameters.dirichlet_ids_master;
  dirichlet_ids.insert(parameters.dirichlet_ids_slave.begin(),
                       parameters.dirichlet_ids_slave.end());
  std::set<types::boundary_id> neumann_ids = parameters.neumann_ids_master;
  neumann_ids.insert(parameters.neumann_ids_slave.begin(),
                     parameters.neumann_ids_slave.end());

  const unsigned int degree = parameters.master_degree;
  const auto         fe     = mesh.get_fe_lagrange(degree);
  const auto         mapping = mesh.get_linear_mapping();
  const QGauss<dim>     quadrature(degree + 1);
  const QGauss<dim - 1> face_quadrature(degree + 1);

  DoFHandler<dim> dof_handler(mesh.get());
  dof_handler.distribute_dofs(*fe);
  const IndexSet owned    = dof_handler.locally_owned_dofs();
  const IndexSet relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);

  AffineConstraints<double> constraints;
  constraints.reinit(owned, relevant);
  for (const auto id : dirichlet_ids)
    VectorTools::interpolate_boundary_values(dof_handler, id, ExactSolution(), constraints);
  constraints.close();

  TrilinosWrappers::SparseMatrix matrix;
  TrilinosWrappers::MPI::Vector  rhs(owned, mpi_comm);
  {
    TimerOutput::Scope timer_section(timer_output(), "monolithic: assembly");

    DynamicSparsityPattern dsp(relevant);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp, owned, mpi_comm, relevant);
    matrix.reinit(owned, owned, dsp, mpi_comm);

    FEValues<dim>     fe_values(*mapping,
                            *fe,
                            quadrature,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values(*mapping,
                                     *fe,
                                     face_quadrature,
                                     update_values | update_quadrature_points |
                                       update_normal_vectors | update_JxW_values);

    const unsigned int                   dofs_per_cell = fe->n_dofs_per_cell();
    FullMatrix<double>                   cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>                       cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
    const Forcing                        forcing;
    const NeumannData                    neumann;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          cell_matrix = 0.;
          cell_rhs    = 0.;

          for (unsigned int q = 0; q < quadrature.size(); ++q)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  cell_matrix(i, j) += (fe_values.shape_grad(i, q) *
                                          fe_values.shape_grad(j, q) +
                                        fe_values.shape_value(i, q) *
                                          fe_values.shape_value(j, q)) *
                                       fe_values.JxW(q);
                cell_rhs(i) += forcing.value(fe_values.quadrature_point(q)) *
                               fe_values.shape_value(i, q) * fe_values.JxW(q);
              }

          for (const auto &face : cell->face_iterators())
            if (face->at_boundary() && neumann_ids.count(face->boundary_id()))
              {
                fe_face_values.reinit(cell, face);
                for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                  {
                    const double g =
                      neumann.gradient(fe_face_values.quadrature_point(q)) *
                      fe_face_values.normal_vector(q);
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      cell_rhs(i) += g * fe_face_values.shape_value(i, q) *
                                     fe_face_values.JxW(q);
                  }
              }

          cell->get_dof_indices(dof_indices);
          constraints.distribute_local_to_global(
            cell_matrix, cell_rhs, dof_indices, matrix, rhs);
        }
    matrix.compress(VectorOperation::add);
    rhs.compress(VectorOperation::add);
  }

  TrilinosWrappers::MPI::Vector solution(owned, mpi_comm);
  {
    TimerOutput::Scope timer_section(timer_output(), "monolithic: solve");

    TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
    amg_data.higher_order_elements = degree > 1;
    TrilinosWrappers::PreconditionAMG preconditioner;
    preconditioner.initialize(matrix, amg_data);

    SolverControl control(20000, 1e-12 * rhs.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector> cg(control);
    cg.solve(matrix, solution, rhs, preconditioner);
    n_cg_iterations = control.last_step();
    constraints.distribute(solution);
  }

  TrilinosWrappers::MPI::Vector ghosted(owned, relevant, mpi_comm);
  ghosted = solution;

  SubdomainReport report;
  report.degree  = degree;
  report.n_dofs  = dof_handler.n_dofs();
  report.n_cells = mesh.get().n_global_active_cells();
  report.h_avg   = mesh.diameter_avg();
  report.error   = h1_error(mesh, dof_handler, ghosted, degree);
  return report;
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

      if (parameters.monolithic)
        {
          pcout() << "Mode: monolithic single-domain reference solve" << std::endl;
          unsigned int n_cg_iterations = 0;
          const SubdomainReport report = run_monolithic(parameters, n_cg_iterations);
          pcout() << "CG converged in " << n_cg_iterations << " iterations." << std::endl;
          pcout() << "H1 error vs. exact solution: " << report.error << std::endl;
          write_results(parameters, {{"monolithic", report}}, report.error, n_cg_iterations);
          return 0;
        }

      auto mesh_master = std::make_shared<MeshHandler>();
      auto mesh_slave  = std::make_shared<MeshHandler>();

      if (parameters.geometry == GeometryType::adjacent_boxes)
        {
          build_adjacent_boxes(*mesh_master,
                               true,
                               parameters.subdivisions_master,
                               parameters.global_refinement_master,
                               parameters.tets_master);
          build_adjacent_boxes(*mesh_slave,
                               false,
                               parameters.subdivisions_slave,
                               parameters.global_refinement_slave,
                               parameters.tets_slave);
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
      // transfer direction, so each needs its own RBF machinery. As in
      // lifex, with "RBF radius factor" r_f each side's radius is r_f times
      // the average cell diameter of *its own* mesh; otherwise the absolute
      // "RBF radius" is used on both sides (0 => Lagrange interpolation).
      const double radius_master = parameters.rbf_radius_factor > 0. ?
                                     parameters.rbf_radius_factor *
                                       mesh_master->diameter_avg() :
                                     parameters.rbf_radius;
      const double radius_slave = parameters.rbf_radius_factor > 0. ?
                                    parameters.rbf_radius_factor *
                                      mesh_slave->diameter_avg() :
                                    parameters.rbf_radius;
      if (radius_master > 0. || radius_slave > 0.)
        pcout() << "RBF radii: master " << radius_master << ", slave "
                << radius_slave << std::endl;

      pcout() << "Building master subproblem..." << std::endl;
      auto master = std::make_shared<SubProblemDiffusionReaction>(
        mesh_master,
        parameters.master_degree,
        dirichlet,
        neumann,
        forcing_term,
        boundary_tags_master,
        coefficients,
        radius_master);

      pcout() << "Building slave subproblem..." << std::endl;
      auto slave = std::make_shared<SubProblemDiffusionReaction>(mesh_slave,
                                                                 parameters.slave_degree,
                                                                 dirichlet,
                                                                 neumann,
                                                                 forcing_term,
                                                                 boundary_tags_slave,
                                                                 coefficients,
                                                                 radius_slave);

      pcout() << "Coupling master and slave..." << std::endl;
      auto problem = std::make_shared<MultiDomainProblem>(master, slave);

      ReductionControl solver_control(parameters.gmres_max_it,
                                      parameters.gmres_tolerance,
                                      parameters.gmres_reduction);
      InternodesSchurComplement solver(
        problem,
        solver_control,
        InternodesSchurComplement::GMRESData(parameters.gmres_basis_size,
                                             parameters.gmres_right_preconditioning));
      solver.set_use_schur_preconditioner(parameters.use_schur_preconditioner);

      pcout() << "Solving..." << std::endl;
      {
        TimerOutput::Scope timer_section(timer_output(), "driver: solve()");
        solver.solve();
      }
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

      std::vector<std::pair<std::string, SubdomainReport>> reports;
      double                                               error_squared = 0.0;
      for (const auto &[name, sub, sol, degree, radius] :
           {std::tuple{std::string("master"),
                       master.get(),
                       &solution_master,
                       parameters.master_degree,
                       radius_master},
            std::tuple{std::string("slave"),
                       slave.get(),
                       &solution_slave,
                       parameters.slave_degree,
                       radius_slave}})
        {
          SubdomainReport report;
          report.degree           = degree;
          report.n_dofs           = sub->dof_handler->n_dofs();
          report.n_interface_dofs =
            sub->interface_dofHandler_ptr->interface_dofs_global().n_elements();
          report.n_cells    = sub->slice->get().n_global_active_cells();
          report.h_avg      = sub->slice->diameter_avg();
          report.rbf_radius = radius;
          report.error      = h1_error(*sub->slice, *sub->dof_handler, *sol, degree);
          error_squared += report.error * report.error;
          reports.emplace_back(name, report);
        }

      const double total_error = std::sqrt(error_squared);
      pcout() << "Broken H1 error vs. exact solution: " << total_error << std::endl;

      write_results(parameters, reports, total_error, solver.get_n_iterations());
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
