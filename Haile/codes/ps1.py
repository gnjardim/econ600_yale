# setup
import pyblp
import numpy as np
import pandas as pd
from pathlib import Path

pyblp.options.digits = 2
pyblp.options.verbose = False

# set home directory
HOME_DIR = "C:/Users/guilh/Dropbox/ECON 600 - Industrial Organization I/First Part/Problem Sets/"

# read data
df = pd.read_csv(Path(HOME_DIR, 'data.csv'))
df.rename(columns={'market_ID': 'market_ids', 
                   'product_ID': 'product_ids'}, inplace=True)

df_demandonly = df.copy()
df_demandonly.rename(columns={'w': 'demand_instruments0'}, inplace=True)

# 9. ---------------------------------------------------------------------
# define X1 and X2
X1_formulation = pyblp.Formulation('1 + prices + x + satellite')
X2_formulation = pyblp.Formulation('0 + satellite + wired')
product_formulations = (X1_formulation, X2_formulation)

# integration and optimization
mc_integration = pyblp.Integration('monte_carlo', size=500, specification_options={'seed': 0})

# estimate model with defaults (demand only)
prb = pyblp.Problem(product_formulations, df_demandonly, integration=mc_integration)
results_noopt = prb.solve(sigma = 0.5*np.eye((2)))

# optimal instruments
opt_instr = results_noopt.compute_optimal_instruments()
prb_opt = opt_instr.to_problem()
results_opt = prb_opt.solve(sigma = 0.5*np.eye((2)))


# with supply
df['firm_ids'] = df.product_ids
product_formulations_supply = (
   X1_formulation, X2_formulation,
   pyblp.Formulation('1 + w')
)

prb_supply = pyblp.Problem(product_formulations_supply, df, integration=mc_integration, costs_type='log')
results_noopt_supply = prb_supply.solve(sigma = 0.5*np.eye((2)), beta = [2, -1, 0.5, 0.05])

# optimal instruments
optimization_tr = pyblp.Optimization('trust-constr')

opt_instr_supply = results_noopt_supply.compute_optimal_instruments()
prb_opt_supply = opt_instr_supply.to_problem(demand_shifter_formulation = pyblp.Formulation('x'))
results_opt_supply = prb_opt_supply.solve(sigma = 0.5*np.eye((2)), beta = [2, -1, 0.5, 0.05],
                                          optimization = optimization_tr)


# export model results 
results_opt.to_pickle(Path(HOME_DIR, 'models/demand_only_blp.pkl'))
results_opt_supply.to_pickle(Path(HOME_DIR, 'models/full_blp.pkl'))
#results_opt_supply = pyblp.read_pickle(Path(HOME_DIR, 'models/full_blp.pkl'))


# 10. --------------------------------------------------------------------
# compute own-price elasticities
own_price_elasts = results_opt_supply.extract_diagonals(results_opt_supply.compute_elasticities())
pd.DataFrame(own_price_elasts).describe()

# compute diversion ratios
all_diversion_ratios = np.zeros((4, 4, 600))
for t in range(600):
   all_diversion_ratios[:, :, t] = results_opt_supply.compute_diversion_ratios(market_id = (t+1))


# extract only non-outside good diversion ratios (compare with previous results)
diversion_ratios = np.zeros((4, 4, 600))
for j in range(4):
   for k in range(4):
      for t in range(600):
         if j != k:
            diversion_ratios[j, k, t] = all_diversion_ratios[j, k, t]

diversion_ratios = diversion_ratios[diversion_ratios != 0]
pd.DataFrame(diversion_ratios).describe()


# 12. --------------------------------------------------------------------
# compute costs
costs = results_opt_supply.compute_costs()

# simulate merger 1-2
merger_df_12 = df.copy()
merger_df_12['merger_ids'] = merger_df_12['firm_ids'].replace(2, 1)

# post-merger equilibrium prices
changed_prices12 = results_opt_supply.compute_prices(
    firm_ids = merger_df_12['merger_ids'],
    costs = costs
)

changes_12 = changed_prices12 - merger_df_12[["prices"]]

# bootstrap (was taking too long)
#bootstrapped_results = results_opt_supply.bootstrap(draws=100, seed=0)
#
#bounds = np.percentile(
#    bootstrapped_results.compute_prices(
#      firm_ids = merger_df_12['merger_ids'],
#      costs = bootstrapped_results.compute_costs()
#   ) - merger_df_12[["prices"]],
#    q=[10, 90],
#    axis=0
#)

# 13. --------------------------------------------------------------------
# simulate merger 1-3
merger_df_13 = df.copy()
merger_df_13['merger_ids'] = merger_df_13['firm_ids'].replace(3, 1)

# post-merger equilibrium prices
changed_prices13 = results_opt_supply.compute_prices(
    firm_ids = merger_df_13['merger_ids'],
    costs = costs
)

changes_13 = changed_prices13 - merger_df_13[["prices"]]

# compare average across markets from 12 and 13
changes_12_reshaped = changes_12.values.reshape(600, 4, order='F')
changes_13_reshaped = changes_13.values.reshape(600, 4, order='F')

changes_12_averages = pd.DataFrame(changes_12_reshaped, columns=['Product1', 'Product2', 'Product3', 'Product4']).mean(axis=0)
changes_13_averages = pd.DataFrame(changes_13_reshaped, columns=['Product1', 'Product2', 'Product3', 'Product4']).mean(axis=0)


# 15. --------------------------------------------------------------------
merger_costs = costs.copy()
merger_costs[merger_df_12.merger_ids == 1] = 0.85*merger_costs[merger_df_12.merger_ids == 1]

# post-merger equilibrium prices
changed_prices = results_opt_supply.compute_prices(
    firm_ids = merger_df_12['merger_ids'],
    costs = merger_costs
)

changes_12 = changed_prices - merger_df_12[["prices"]]

# compare average across markets
changes_12_reshaped = changes_12.values.reshape(600, 4, order='F')
changes_12_averages = pd.DataFrame(changes_12_reshaped, columns=['Product1', 'Product2', 'Product3', 'Product4']).mean(axis=0)

# changes in shares
changed_shares = results_opt_supply.compute_shares(changed_prices)

# welfare
cs_m = results_opt_supply.compute_consumer_surpluses(prices=changed_prices)
cs_p = results_opt_supply.compute_consumer_surpluses(prices=df[["prices"]])
cs_change = np.sum(cs_m - cs_p)

profits = results_opt_supply.compute_profits(changed_prices, changed_shares, merger_costs)
profits_p = results_opt_supply.compute_profits(prices=df[["prices"]])
profits_change = np.sum(profits - profits_p) 
np.sum(cs_change) + np.sum(profits_change)