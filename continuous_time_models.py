import numpy as np
import scipy.optimize as optimize
from functools import partial


class AssetsDynamics:

    def __init__(self, drift, volatility, r_free, tau, alpha, q=None, eta=None, theta=None):
        """

        :param drift: expected return of dX/X - Geometric Brownian Motion for assets dynamics
        :param volatility: volatility of dX/X - Geometric Brownian Motion for assets dynamics
        :param r_free: riskless return
        :param tau: taxes
        :param alpha: bankruptcy costs
        :param q: probability of default of the renegotiation
        :param eta: shareholders bargaining power in renegotiation
        :param theta: shareholders portion of assets if renegotiation happens (1-theta for debtholders)
        """
        self.drift = drift
        self.volatility = volatility
        self.r_free = r_free
        self.tau = tau
        self.alpha = alpha
        self.q = q
        self.eta = eta
        self.theta = theta

    def value(self, cash_flow):
        assets = cash_flow * (1-self.tau) / (self.r_free - self.drift)
        return assets

    @classmethod
    def debt_renegotation_optimal_theta(cls, drift, volatility, r_free, tau, alpha, q, eta):
        theta = eta * alpha
        return cls(drift, volatility, r_free, tau, alpha, q, eta, theta)


class FirmClaimsLiquidation:

    def __init__(self, assets_dynamics: AssetsDynamics):
        self.asset_dynamics = assets_dynamics

    def firm(self, coupon, cash_flow, debt_liquidation):
        return self.equity(coupon, cash_flow, debt_liquidation) + self.debt(coupon, cash_flow, debt_liquidation)

    def equity(self, coupon, cash_flow, debt_liquidation):
        risk_free_debt = coupon / self.asset_dynamics.r_free
        pv_cash_flow_no_default = self.asset_dynamics.value(cash_flow) - (1 - self.asset_dynamics.tau) * risk_free_debt
        option_to_default = self._arrow_debreu_default(cash_flow, debt_liquidation) * (
                self.asset_dynamics.value(cash_flow=debt_liquidation) - (1 - self.asset_dynamics.tau) * risk_free_debt)
        equity_value = pv_cash_flow_no_default - option_to_default
        return equity_value

    def debt(self, coupon, cash_flow, debt_liquidation):
        risk_free_debt = coupon / self.asset_dynamics.r_free
        debt_change =  self._arrow_debreu_default(cash_flow, debt_liquidation) * \
                       ((1 - self.asset_dynamics.alpha) * self.asset_dynamics.value(cash_flow=debt_liquidation) -
                        risk_free_debt)
        debt_value = risk_free_debt + debt_change
        return debt_value

    def _arrow_debreu_default(self, cash_flow, debt_limit):
        arrow_debreu_default = (cash_flow/debt_limit) ** self._beta2
        return arrow_debreu_default

    def _beta2(self):
        beta2 = 0.5 - self.asset_dynamics.drift/(self.asset_dynamics.volatility ** 2) - np.sqrt(
            (0.5 - self.asset_dynamics.drift/(self.asset_dynamics.volatility ** 2)) ** 2
            + 2 * self.asset_dynamics.r_free / (self.asset_dynamics.volatility ** 2)
        )
        return beta2

    def _max_coupon(self, cash_flow, debt_liquidation):
        func = - self.firm
        result = optimize.minimize(fun=func, x0=0.001, args=(cash_flow, debt_liquidation))
        coupon = result.x
        return coupon


class FirmClaimsRenegotiation:

    def __init__(self, assets_dynamics: AssetsDynamics):
        self.asset_dynamics = assets_dynamics

    def firm(self, coupon, cash_flow):
        return self.equity(coupon, cash_flow) + self.debt(coupon, cash_flow)

    def equity(self, coupon, cash_flow):
        risk_free_debt = coupon / self.asset_dynamics.r_free
        pv_cash_flow_no_default = self.asset_dynamics.value(cash_flow) - (1 - self.asset_dynamics.tau) * risk_free_debt
        option_to_default = self._arrow_debreu_default(cash_flow, self.debt_renegotiation(coupon)) * (
                self.asset_dynamics.value(cash_flow=self.debt_renegotiation(coupon)) *
                (1 - (1 - self.asset_dynamics.q) * self.asset_dynamics.theta) - (1 - self.asset_dynamics.tau) * risk_free_debt)
        equity_value = pv_cash_flow_no_default - option_to_default
        return equity_value

    def debt(self, coupon, cash_flow):
        risk_free_debt = coupon / self.asset_dynamics.r_free
        debt_change = self._arrow_debreu_default(cash_flow, self.debt_renegotiation(coupon)) * \
                      (self.asset_dynamics.q * (1 - self.asset_dynamics.alpha) *
                       self.asset_dynamics.value(cash_flow=self.debt_renegotiation(coupon)) +
                       (1 - self.asset_dynamics.q) * (1 - self.asset_dynamics.theta) *
                       self.asset_dynamics.value(cash_flow=self.debt_renegotiation(coupon)) -
                       risk_free_debt)
        debt_value = risk_free_debt + debt_change
        return debt_value

    def debt_renegotiation(self, coupon):
        risk_free_debt = coupon / self.asset_dynamics.r_free
        debt_renegotiation = (self.asset_dynamics.r_free - self.asset_dynamics.drift) * risk_free_debt * \
                             self._beta2()/((self._beta2() - 1) * (1 - (1 - self.asset_dynamics.q) *
                                                                   self.asset_dynamics.theta))
        return debt_renegotiation

    def _arrow_debreu_default(self, cash_flow, debt_limit):
        arrow_debreu_default = (cash_flow/debt_limit) ** self._beta2()
        return arrow_debreu_default

    def _beta2(self):
        beta2 = 0.5 - self.asset_dynamics.drift/(self.asset_dynamics.volatility ** 2) - np.sqrt(
            (0.5 - self.asset_dynamics.drift/(self.asset_dynamics.volatility ** 2)) ** 2
            + 2 * self.asset_dynamics.r_free / (self.asset_dynamics.volatility ** 2)
        )
        return beta2

    def _obj_max_coupon(self, coupon, cash_flow):
        return -1 * self.firm(coupon, cash_flow)

    def max_coupon(self, cash_flow):
        function = partial(self._obj_max_coupon, cash_flow=cash_flow)
        result = optimize.minimize(fun=function, x0=0.001)  # to try if it works or to modify the structure
        coupon = result.x[0]
        return coupon