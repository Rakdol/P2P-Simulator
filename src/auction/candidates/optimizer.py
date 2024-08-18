import random
from typing import List, Dict, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from auction.candidates.participant import (
        Participant,
        ZeroParticipant,
        LinearParticipant,
        StackelBergParticipant,
    )
    from auction.candidates.market_operator import (
        MarketOperator,
        ZeroOperator,
        LinearOperator,
        StackelBergOperator,
    )


import numpy as np
import pyomo.environ as pyo


"""
Optimizer acts as super class which means it depends on the participants and operator, 
but their actions (bid, fee rate) would be optimized in Optimizer.
AuctionEnv provides the state per participant of auction for each time step (1 hour), 
this would be used to make optimal desicions for participants and operator in AuctionEnv.
"""


class Optimizer(object):

    def get_actions(self, state: Dict[str, np.array]) -> Dict[str, int]:
        raise NotImplementedError()

    def get_optimal_bids(self, model):
        raise NotImplementedError()

    def optimize_auction(
        self,
        state: Dict[str, np.array],
    ):
        raise NotImplementedError()


class ZeroOptimizer(Optimizer):
    """
    ZeroOptimizer proposes to purchase (bid) or to sell (ask) randomly, subject to only to minimal constraints
    This solver is considered as baseline model. Operator's fee rate also can be randomly proposed.
    """

    def __init__(
        self,
        operator: "ZeroOperator",
        participants: List["ZeroParticipant"],
    ):
        self.operator = operator
        self.participants = participants

    def get_random_maker(self, a, b):
        return random.randint(a, b)

    def get_actions(self, state: Dict[str, np.array]) -> Tuple[Dict[str, int], float]:

        bids, fee_rate = self.optimize_auction(state=state)

        action_dict = {}
        for participant, bid in zip(self.participants, bids):
            action_dict[participant.id] = participant.action_space[bid]

        return action_dict, fee_rate

    def optimize_auction(
        self,
        state: Dict[str, np.array],
    ) -> Tuple[List[int], float]:

        fee_rate = self.operator.get_fee_rate(
            maker=self.get_random_maker(0, len(self.operator.fee_space) - 1)
        )
        bids = [
            participant.bid_action(
                maker=self.get_random_maker(0, len(participant.action_space) - 1)
            )
            for participant in self.participants
        ]
        return bids, fee_rate


class LinearOptimizer(Optimizer):
    """
    LinearOptimizer proposes to purchase (bid) or to sell (ask) using linear programming, subject to maximize the social welfare.
    The social welfare maximize the difference between bid and ask, and also maximize the difference between mid price and bid(or ask).
    Mid price will be determined using participant's own upper and lower bound price.
    """

    def __init__(
        self,
        operator: "LinearOperator",
        participants: List["LinearParticipant"],
        solver: str = "appsi_highs",
    ):

        self.operator = operator
        self.participants = participants
        self.SOLVER = pyo.SolverFactory(solver)
        assert self.SOLVER.available(), f"Solver {solver} is not available."

    def get_actions(self, state: Dict[str, np.array]) -> Dict[str, int]:

        model = self.optimize_auction(state=state)
        fee_rate = self.operator.get_fee_rate(self.get_ask_bid_spread(model))
        bids = self.get_optimal_bids(model)

        return bids, fee_rate

    def get_ask_bid_spread(self, model):

        bid_mean = 0
        ask_mean = 0

        bids = {}
        for k in model.BUYERS:
            # print(f"buyer {k} bid price: {model.bid[k].value}")
            bids[k] = model.bid[k].value
            bid_mean += model.bid[k].value

        for k in model.SELLERS:
            # print(f"seller {k} bid price: {model.ask[k].value}")
            bids[k] = model.ask[k].value
            ask_mean += model.ask[k].value

        bid_mean /= len(model.BUYERS)
        if not model.SELLERS:
            ask_mean = 0
        else:
            ask_mean /= len(model.SELLERS)

        ask_bid_ratio = ask_mean / bid_mean

        return ask_bid_ratio

    def optimize_auction(self, state: Dict[str, np.array]):
        model = pyo.ConcreteModel("Participant bid optimization")

        sellers, buyers = [], []

        for (key, val), participant in zip(state.items(), self.participants):
            if val[4] > 0:
                if participant.id == key:
                    participant.has_product = True
                sellers.append(key)

            if val[4] < 0:
                if participant.id == key:
                    participant.has_product = False
                buyers.append(key)

        model.SELLERS = pyo.Set(initialize=sellers)
        model.BUYERS = pyo.Set(initialize=buyers)

        @model.Param(model.BUYERS)
        def upper_bound(model, buyer):
            return state[buyer][3]

        @model.Param(model.SELLERS)
        def lower_bound(model, seller):
            return state[seller][2]

        @model.Param(model.SELLERS)
        def sell_price(
            model,
            seller,
        ):
            upper = state[seller][3]
            lower = state[seller][2]
            mid = (upper + lower) / 2
            return mid

        @model.Param(model.BUYERS)
        def buy_price(
            model,
            buyer,
        ):
            upper = state[buyer][3]
            lower = state[buyer][2]
            mid = (upper + lower) / 2
            return mid

        model.ask = pyo.Var(model.SELLERS, domain=pyo.NonNegativeReals)
        model.bid = pyo.Var(model.BUYERS, domain=pyo.NonNegativeReals)
        model.abs_diff = pyo.Var(
            model.SELLERS | model.BUYERS, within=pyo.NonNegativeReals
        )

        @model.Objective(sense=pyo.maximize)
        def welfare(model):
            """
            Social welfare maximize the difference between bid and ask (bid-ask spread)
            Maximize bid-ask spread, leading the highest surplus of seller and buyer.
            and also minimize the difference between participant's mid price and bid(or ask),
            which leads the price stability, converge to mid price of market
            """
            return pyo.quicksum(
                model.bid[b] - model.ask[s] for s in model.SELLERS for b in model.BUYERS
            ) - pyo.quicksum(model.abs_diff[p] for p in model.SELLERS | model.BUYERS)

        model.price_constraints = pyo.ConstraintList()
        # Inequality includes random process.
        for seller in model.SELLERS:
            model.price_constraints.add(
                pyo.inequality(
                    model.sell_price[seller] * (1.0 - random.uniform(0, 0.1)),
                    model.ask[seller],
                    model.sell_price[seller] * (1.0 + random.uniform(0, 0.1)),
                )
            )

        for buyer in model.BUYERS:
            model.price_constraints.add(
                pyo.inequality(
                    model.buy_price[buyer] * (1.0 - random.uniform(0, 0.1)),
                    model.bid[buyer],
                    model.buy_price[buyer] * (1.0 + random.uniform(0, 0.1)),
                )
            )

        # Absolute value constraints
        model.abs_constraints = pyo.ConstraintList()
        for p in model.SELLERS:
            model.abs_constraints.add(
                model.abs_diff[p] >= model.ask[p] - model.sell_price[p]
            )
            model.abs_constraints.add(
                model.abs_diff[p] >= model.sell_price[p] - model.ask[p]
            )
        for p in model.BUYERS:
            model.abs_constraints.add(
                model.abs_diff[p] >= model.buy_price[p] - model.bid[p]
            )
            model.abs_constraints.add(
                model.abs_diff[p] >= model.bid[p] - model.buy_price[p]
            )

        self.SOLVER.solve(model)

        return model

    def get_optimal_bids(self, model):
        bids = {}
        for participant in self.participants:
            bids[participant.id] = participant.bid_action(model)

        return bids


class StackelBergOptimizer(Optimizer):
    """
    Stackelberg leadership model is a strategic game in economics in which the leader firm moves first
    and then the follower firms move sequentially.
    In our problem, A leader (Operator) provides transaction fee rate,
    and the followers (participants) make their own bid prices.
    Linear Optimization is appiled to make optimal decisions.
    """

    def __init__(
        self,
        operator: "StackelBergOperator",
        participants: List["StackelBergParticipant"],
        solver: str = "appsi_highs",
    ):

        self.operator = operator
        self.participants = participants
        self.SOLVER = pyo.SolverFactory(solver)
        assert self.SOLVER.available(), f"Solver {solver} is not available."

    def get_actions(self, state: Dict[str, np.array]) -> Dict[str, int]:

        initial_fee = self.operator.initial_fee
        model = self.optimize_auction(state=state, fee_rate=initial_fee)
        bids = self.get_optimal_bids(model)
        optimal_fee_rate = self.operator.get_fee_rate(maker=bids)
        self.operator.initial_fee = optimal_fee_rate

        return bids, optimal_fee_rate

    def optimize_auction(self, state: Dict[str, np.array], fee_rate: float):
        model = pyo.ConcreteModel("Participant bid optimization")
        prob = self.operator.bid_prob_with_fee_rate(fee_rate)

        sellers, buyers = [], []
        for (key, val), participant in zip(state.items(), self.participants):
            if val[4] > 0:
                if participant.id == key:
                    participant.has_product = True
                sellers.append(key)

            if val[4] < 0:
                if participant.id == key:
                    participant.has_product = False
                buyers.append(key)

        model.SELLERS = pyo.Set(initialize=sellers)
        model.BUYERS = pyo.Set(initialize=buyers)

        @model.Param(model.BUYERS)
        def upper_bound(model, buyer):
            return state[buyer][3]

        @model.Param(model.SELLERS)
        def lower_bound(model, seller):
            return state[seller][2]

        @model.Param(model.SELLERS)
        def sell_price(
            model,
            seller,
        ):
            upper = state[seller][3]
            lower = state[seller][2]
            mid = (upper + lower) / 2
            return mid

        @model.Param(model.BUYERS)
        def buy_price(
            model,
            buyer,
        ):
            upper = state[buyer][3]
            lower = state[buyer][2]
            mid = (upper + lower) / 2
            return mid

        model.ask = pyo.Var(model.SELLERS, domain=pyo.NonNegativeReals)
        model.bid = pyo.Var(model.BUYERS, domain=pyo.NonNegativeReals)
        model.abs_diff = pyo.Var(
            model.SELLERS | model.BUYERS, within=pyo.NonNegativeReals
        )

        @model.Objective(sense=pyo.maximize)
        def welfare(model):
            """
            Social welfare maximize the difference between bid and ask (bid-ask spread)
            Maximize bid-ask spread, leading the highest surplus of seller and buyer.
            and also minimize the difference between participant's mid price and bid(or ask),
            which leads the price stability, converge to mid price of market
            """
            return pyo.quicksum(
                (model.bid[b] - model.ask[s]) * prob
                for s in model.SELLERS
                for b in model.BUYERS
            ) - pyo.quicksum(model.abs_diff[p] for p in model.SELLERS | model.BUYERS)

        model.price_constraints = pyo.ConstraintList()
        # Inequality includes random process.
        for seller in model.SELLERS:
            model.price_constraints.add(
                pyo.inequality(
                    model.sell_price[seller]
                    * (1.0 - fee_rate - random.uniform(0, 0.1)),
                    model.ask[seller],
                    model.sell_price[seller]
                    * (1.0 + fee_rate + random.uniform(0, 0.1)),
                )
            )

        for buyer in model.BUYERS:
            model.price_constraints.add(
                pyo.inequality(
                    model.buy_price[buyer] * (1.0 - fee_rate - random.uniform(0, 0.1)),
                    model.bid[buyer],
                    model.buy_price[buyer] * (1.0 + fee_rate + random.uniform(0, 0.1)),
                )
            )

        # Absolute value constraints
        model.abs_constraints = pyo.ConstraintList()
        for p in model.SELLERS:
            model.abs_constraints.add(
                model.abs_diff[p] >= model.ask[p] - model.sell_price[p]
            )
            model.abs_constraints.add(
                model.abs_diff[p] >= model.sell_price[p] - model.ask[p]
            )
        for p in model.BUYERS:
            model.abs_constraints.add(
                model.abs_diff[p] >= model.buy_price[p] - model.bid[p]
            )
            model.abs_constraints.add(
                model.abs_diff[p] >= model.bid[p] - model.buy_price[p]
            )

        self.SOLVER.solve(model)

        return model

    def get_optimal_bids(self, model: pyo.ConcreteModel):
        bids = {}
        for participant in self.participants:
            bids[participant.id] = participant.bid_action(model)

        return bids


class SingleReinforceOptimizer(Optimizer):
    def __init__(self):
        self.model = None

    def optimize_auction(
        self,
        state: Dict[str, np.array],
    ):
        return
