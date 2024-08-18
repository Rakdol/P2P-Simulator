"""

AuctionContoller
- which interfaces with AuctionModel and AuctionView

"""

from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from components.model import AuctionModel
    from components.view import AuctionView
    from components.strategy import AuctionContext
    from components.reward import Reward
    from src.network.dist_operator import DistOperator


class AuctionController(object):
    def __init__(
        self,
        model: "AuctionModel",
        view: "AuctionView",
        reward: "Reward",
        dms: "DistOperator" = None,
    ):
        self._model = model
        self._view = view
        self._reward = reward
        self._dms = dms

    def get_size(self):
        return self._model.get_size()

    def get_reward(self, tranactions: pd.DataFrame):
        return self._reward.get_reward(tranactions)

    def get_customer_list(self):
        return self._model.data_container.customer_ids

    def get_state(self):
        return self._model.get_state()

    def reset(self):
        self._model.reset()

    def get_obeservation_space(self):
        return self._model.get_obeservation_space()

    def get_action_space(self):
        return self._model.get_action_space()

    def run_auction(
        self,
        strategy: "AuctionContext",
        state: Dict[str, np.array],
        action: Tuple[Dict[str, int], float],
    ):
        transactions = self._model.run_auction(
            strategy=strategy, state=state, action=action
        )

        self._render_context = transactions

        return transactions



        

    def run_render(self):
        self._view.render(self._render_context)
