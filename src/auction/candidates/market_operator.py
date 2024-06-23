import os
import sys
import random
from typing import Dict, List, Tuple, Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import pyomo.environ as pyo


sys.path.append(os.getcwd() + "\\P2P-simulator\\src\\")
if TYPE_CHECKING:
    from auction.components.strategy import AuctionContext
    from auction.components.model import AuctionModel
    from auction.candidates.participant import Participant
    from auction.candidates.optimizer import Optimizer


class MarketOperator(object):

    def __init__(
        self,
        initial_fee=0.05,
    ):

        self.initial_fee = (
            initial_fee  # TODO include in actions in the future, this can be optimized.
        )
        self.fee_space = np.linspace(0, 0.3, 31, endpoint=True)

    def get_fee_rate(self, maker: Any):
        raise NotImplementedError()


class ZeroOperator(MarketOperator):
    def get_fee_rate(self, maker: Any):
        fee_rate = self.fee_space[maker]
        return fee_rate


class LinearOperator(MarketOperator):
    def get_fee_rate(self, maker: Any):
        # maker is a function to get ask-bid spread
        if maker == 0.0:
            return 0

        fee_rate = max(0, min(round(1 - maker, 3), np.max(self.fee_space)))

        return fee_rate


class StackelBergOperator(MarketOperator):

    def bid_prob_with_fee_rate(self, fee_rate: float, a: int = 10, b: float = 0.5):
        # Revserse sigmoid function approximation
        # High fee rate leads low probability
        return 1 / (1 + np.exp(-a * (-fee_rate + b)))

    def calc_slope_and_intersection(self, x: List[float], y: List[float]):
        slope = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        intersection = y[:-1] - slope * x[:-1]
        return float(slope), float(intersection)

    def get_slope_intersections(
        self, num_segments: int, x_segments: np.array, b: float
    ):
        slope, intersection = [], []
        for i in range(num_segments):
            x_segment = x_segments[i : i + 2]  # Get the x values for the segment
            y_segment = self.bid_prob_with_fee_rate(
                fee_rate=x_segment, b=b
            )  # Sample from nonlinear environment
            slope_, intersection_ = self.calc_slope_and_intersection(
                x_segment, y_segment
            )
            slope.append(slope_)
            intersection.append(intersection_)
            # print(
            #     f"[+] Linear function of segment {i}: x*{slope[i]:.2f}+({intersection[i]:.2f})."
            # )

        return slope, intersection

    def get_fee_rate(self, maker: Dict[str, float]):
        solver = pyo.SolverFactory("appsi_highs")
        assert self.SOLVER.available(), f"Solver {solver} is not available."
        model = pyo.ConcreteModel("Fee rate Optimize")

        # Define the set of segments
        # Because reverse sigmoid is non-linear function,
        # in this problem we divides the sigmoid in several segments (10)
        # to make piecewise linear function. Therefore, we can apply linear programming.

        num_segments = 10
        lower_fee_rate = 0.0
        upper_fee_rate = 1.0
        x_segments = np.linspace(lower_fee_rate, upper_fee_rate, num_segments + 1)

        # Sigmoid bias is calcluated from bid prices
        biddings = np.array(list(maker.values()))
        min_max_range = np.max(biddings) - np.min(biddings)
        norms = (biddings - np.min(biddings)) / min_max_range
        sigmoid_bias = np.median(norms) + np.std(norms)

        slope, intersection = self.get_slope_intersections(
            num_segments, x_segments, sigmoid_bias
        )

        N = range(num_segments)
        model.segments = pyo.Set(initialize=N)
        model.fee_rate = pyo.Var(
            initialize=lower_fee_rate,
            bounds=(lower_fee_rate, upper_fee_rate),
            domain=pyo.Reals,
        )

        def tilde_bounds(m, i):
            lb = lower_fee_rate
            ub = upper_fee_rate
            return (lb, ub)

        def tilde_init(m, i):
            return x_segments[i]

        model.fee_tilde = pyo.Var(
            model.segments,
            initialize=tilde_init,
            bounds=tilde_bounds,
            domain=pyo.Reals,
        )

        # Define the binary variables
        def z_init(m, i):
            if i == 0:
                return 1
            else:
                return 0

        model.z = pyo.Var(model.segments, initialize=z_init, domain=pyo.Binary)

        model.revenue = pyo.Objective(
            expr=sum(
                [
                    (bid * slope[i] * model.fee_tilde[i] + model.z[i] * intersection[i])
                    for i in model.segments
                    for id, bid in maker.items()
                ]
            ),
            sense=pyo.minimize,
            # bid probabiltiy is correleated with negative fee rate so taht
            # minimize is equal to maximize fee_rate any given constraints.
        )

        K = 0.99  # minimum probability for bid
        model.c1 = pyo.Constraint(
            expr=sum(
                [
                    (slope[i] * model.fee_tilde[i] + model.z[i] * intersection[i])
                    for i in model.segments
                ]
            )
            >= K
        )
        model.c2 = pyo.Constraint(expr=sum([model.z[i] for i in model.segments]) == 1)

        # Define rules for the constraints that transform the bilinear terms
        def c3_rule(m, i):
            xmin = x_segments[i]
            return xmin * m.z[i] <= m.fee_tilde[i]

        def c4_rule(m, i):
            xmax = x_segments[i + 1]
            return m.fee_tilde[i] <= xmax * m.z[i]

        def c5_rule(m, i):
            xmax = upper_fee_rate
            return m.fee_rate - xmax * (1 - m.z[i]) <= m.fee_tilde[i]

        def c6_rule(m, i):
            return m.fee_tilde[i] <= m.fee_rate

        # Use the rules to write the constraints
        model.c3 = pyo.Constraint(model.segments, rule=c3_rule)
        model.c4 = pyo.Constraint(model.segments, rule=c4_rule)
        model.c5 = pyo.Constraint(model.segments, rule=c5_rule)
        model.c6 = pyo.Constraint(model.segments, rule=c6_rule)

        solver.solve(model)

        return max(0, round(min(model.fee_rate(), np.max(self.operator.fee_space)), 3))


class SingleReinforceOperator(MarketOperator):
    def get_fee_rate(self, maker: Any):
        return super().get_fee_rate(maker)
