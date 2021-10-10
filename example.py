import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from continuous_time_models import AssetsDynamics, FirmClaimsRenegotiation

sns.set_style('white')
sns.set_context("paper", font_scale=1.1)

assets_dynamics = AssetsDynamics.debt_renegotation_optimal_theta(drift=0.01, volatility=0.25, r_free=0.04, tau=0.15,
                                                                 alpha=0.4, q=1, eta=0.5)
initial_cash_flow = 1
model = FirmClaimsRenegotiation(assets_dynamics)
# optimal renegotiation debt: optimal coupon and spread
opt_coupon = model.max_coupon(cash_flow=initial_cash_flow)
debt_yield = opt_coupon/model.debt(opt_coupon, initial_cash_flow)
spread = debt_yield - model.asset_dynamics.r_free
print('renegotiation boundary ' + str(model.debt_renegotiation(opt_coupon)))
print('opt_coupon ' + str(opt_coupon))
print('equity ' + str(model.equity(opt_coupon, initial_cash_flow)))
print('debt  ' + str(model.debt(opt_coupon, initial_cash_flow)))
print('yield ' + str(debt_yield))
print('spread ' + str(spread))

# plot the values of equity and debt depending on cash flow
cashflow_support = np.linspace(0.5, 4)
data = {'Equity': model.equity(opt_coupon, cashflow_support),
        'Debt': model.debt(opt_coupon, cashflow_support),
        'Cash Flow': cashflow_support}
df = pd.DataFrame(data)
melted_df = df.melt(id_vars=['Cash Flow'], var_name='Claims', value_name='Payoff')
scatter = sns.lineplot(data=melted_df,  y='Payoff', hue='Claims', x='Cash Flow')
#plt.savefig("scatter_firm_claims.pdf", bbox_inches='tight')
plt.show()
