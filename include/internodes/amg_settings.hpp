#ifndef INTERNODES_AMG_SETTINGS_HPP
#define INTERNODES_AMG_SETTINGS_HPP

#include <deal.II/lac/trilinos_precondition.h>

#include <cstdlib>
#include <string>

namespace internodes
{
  /// ML AMG settings for the volume (subdomain and Schur-preconditioner)
  /// solves. Experimental switch INTERNODES_VOLUME_AMG:
  ///   ho_agg (default) higher_order_elements + elliptic, aggregation_threshold
  ///                    0.02 (close to lifex's older, commented-out 1e-2).
  ///                    Measured (8 laptop ranks): full Test 5 (P2) Step 3
  ///                    248 -> 208 s, CG its/solve -15-17%; Test 1 shape (P1)
  ///                    -8%; Test 4 shape neutral; identical GMRES iterations
  ///                    and H1 errors in all cases.
  ///   lifex            the previous settings (lifex's active code):
  ///                    aggregation_threshold 1e-3, no higher-order flag
  ///   ho               ho_agg with aggregation_threshold 1e-3 (no change
  ///                    vs lifex: the higher-order flag alone does nothing)
  ///   ho_w             ho with W-cycles
  inline dealii::TrilinosWrappers::PreconditionAMG::AdditionalData
  volume_amg_data()
  {
    const char *const env     = std::getenv("INTERNODES_VOLUME_AMG");
    const std::string variant = env ? env : "ho_agg";

    dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
    amg_data.smoother_type = "Chebyshev";
    amg_data.coarse_type   = "Amesos-KLU";
    if (variant == "lifex")
      {
        amg_data.higher_order_elements = false;
        amg_data.aggregation_threshold = 1e-3;
        amg_data.smoother_sweeps       = 2;
      }
    else
      {
        amg_data.elliptic              = true;
        amg_data.higher_order_elements = true;
        amg_data.smoother_sweeps       = 2;
        amg_data.aggregation_threshold = (variant == "ho_agg") ? 2e-2 : 1e-3;
        amg_data.w_cycle               = (variant == "ho_w");
      }
    return amg_data;
  }
} // namespace internodes

#endif
