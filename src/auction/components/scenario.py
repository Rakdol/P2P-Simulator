from typing import Dict
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

"""
This module can be used to change market information scenario in the future
"""


class AuctionScenario(ABC):
    @abstractmethod
    def get_observation_space(self):
        pass

    @abstractmethod
    def get_action_space(self):
        pass

    @abstractmethod
    def get_state(self):
        pass


class BlackBox(AuctionScenario):
    """
    Represent Each participant's observation space
    BlackBox scenario can use participants' own consumption, generation, lower bound, upper bound, has_product
    that is, the shapce is (5,)
    """

    def get_observation_space(self):
        return (5,)

    def get_action_space(self):
        # Descrete action space
        # [0, 0.1, 0.2, ... 0.9, 1.0]
        # this can be used to convert an action of customer
        # between upper bound price and lower bound price

        return np.linspace(0, 1, 11, endpoint=True)

    def get_state(self, model):
        state = {}
        for id in model.data_container.customer_ids:

            cus_each = model.partitioned_dataset[id]
            smp_each = model.data_container.dataset.smp.iloc[model.counter]

            lower_bound = smp_each["smp"]
            upper_bound = model.data_container.dataset.price[
                (model.data_container.dataset.price["id"] == id)
                & (smp_each.month == model.data_container.dataset.price.month)
            ].price.iloc[0]

            consumption = cus_each["consumption"].iloc[model.counter]
            generation = cus_each["generation"].iloc[model.counter]

            has_product = model._is_has_product(generation, consumption)

            state[str(id)] = np.array(
                [
                    consumption,
                    generation,
                    lower_bound,
                    upper_bound,
                    has_product,
                ]
            )

        return state
