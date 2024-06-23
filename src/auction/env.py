# --- built in modules ---
import os
from typing import List, Dict, Tuple

# --- Third party modules ---
import numpy as np
import pandas as pd

# --- Custom modules ---
from auction.config import *
from auction.candidates.market_operator import MarketOperator
from auction.candidates.participant import Participant
from auction.components.model import AuctionModel, DataContainer
from auction.components.view import AuctionView
from auction.components.controller import AuctionController
from auction.components.strategy import AuctionContext, UniformAuction
from auction.components.scenario import BlackBox
from auction.components.reward import Reward


class AuctionEnv(object):
    def __init__(
        self,
        controller: AuctionController,
        strategy: AuctionContext,
    ):

        self.controller = controller
        self.strategy = strategy
        self.action_space = self.controller.get_action_space()
        self.observation_space = self.controller.get_obeservation_space()
        self.customer_list = self.controller.get_customer_list()
        self.max_count = self.controller.get_size()

        self.cuurent_state = None
        self.done = False
        self.counter = 0

    def step(self, action: Tuple[Dict, float]):

        transactions = self.controller.run_auction(
            self.strategy, self.current_state, action
        )

        self.counter += 1
        self.current_state = self.controller.get_state()

        reward = self.controller.get_reward(transactions)

        if self.counter == self.max_count:
            self.done = True

        return self.current_state, reward, self.done, {}

    def reset(self):
        self.counter = 0
        self.done = False
        self.controller.reset()
        self.current_state = self.controller.get_state()

        return self.current_state

    def render(self):
        self.controller.run_render()
