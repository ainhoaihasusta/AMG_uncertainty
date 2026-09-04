# AMG_uncertainty

# The repository contains
- Scripts to perform the uncertainty propagation in the AMG model (Clivot et al. 2019), either using an analytical method or a Monte Carlo approach.
- Data from QualiAgro exprimental site (Fujisaki et al., 2026; Levavasseur et al., 2021), yield from the French regional statistics Agreste (https://agreste.agriculture.gouv.fr/agreste-web/disaron/SAA-SeriesLongues/detail/).
- Inputs to perform the uncertainty propagation, i.e., vectors of mean and sigma values for input data.
- Outputs data generated from uncertainty propagation in the files 'uncertainty_propa_for_absolute_SOC_stock_and_Delta_SOC_stock.ipynb' and 'uncertainty_propa_for_SOC_stock_diff_CF_vs_dynamic_specific_baseline.ipynb'

# Workflow of the codes

- The script files 'uncertainty_carbon_inputs_sensitivity.ipynb' and 'uncertainty_mineralisation_sensitivity.ipynb' can be run individually, all the data required to run them are available here in data repertory.
- For the rest of the scripts :
  - Step (1) : Run the files 'generate_LUT_for_MC_all_scenario.py' and 'generate_LUT_for_MC_carbon_credit.py' to generate the Monte Carlo simulations required in the other scritps. The Monte Carlo simulations will be stored in the outputs repertory.
  - Step (2) : Run the files 'uncertainty_propa_for_SOC_stock_and_Delta_stock.ipynb', 'uncertainty_propa_for_SOC_stock_diff_CF_vs_dynamic_specific_baseline.ipynb' and 'uncertainty_propa_for_SOC_stock_diff_CF_vs_standardised_static_baseline.ipynb', that required the outputs from the scripts of step (1).
  - Step (3) : Run the file 'comparison_analytical_numerical_methods.ipynb'
