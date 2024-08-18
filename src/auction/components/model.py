import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd


PAKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent

sys.path.append(str(PAKAGE_ROOT))

from src.auction.config import (
    CUSTOMER_PATH,
    SMP_PATH,
    BLOCK_PATH,
    PRICE_PATH,
    DataPath,
    DataSet,
)

if TYPE_CHECKING:
    from components.scenario import AuctionScenario
    from components.strategy import AuctionContext


def partitioning(df: pd.DataFrame, index_col: str = "id") -> dict:
    # Fast constructing market state for each customer
    # DataFrame partitioned in dictionary

    part_dict = {}
    for i in df[index_col].unique():
        part_dict[i] = df[df[index_col] == i]

    return part_dict


class DataContainer(object):

    def __init__(
        self, customer_path: str, smp_path: str, price_path: str, block_path: str
    ):
        # --- Data Set up
        self.data_path = DataPath(
            customer_path=customer_path,
            smp_path=smp_path,
            price_path=price_path,
            block_path=block_path,
        )
        self.dataset = DataSet(
            customer=pd.read_csv(self.data_path.customer_path),
            smp=pd.read_csv(self.data_path.smp_path),
            price=pd.read_csv(self.data_path.price_path),
            block=pd.read_csv(self.data_path.block_path),
        )
        self._add_months(self.dataset.smp)

        self.customer_ids = self.get_customer_ids()

    def _add_months(self, smp: pd.DataFrame) -> None:
        smp["month"] = pd.to_datetime(smp["datetime"]).dt.month

    def get_size(self):
        return self.dataset.smp.shape[0] - 1
    
    def get_current_time(self, counter):
        return self.dataset.smp["datetime"].iloc[counter]

    def get_customer_ids(self):
        return list(set(self.dataset.customer.id))


class AuctionModel(object):

    def __init__(
        self,
        data_container: "DataContainer",
        scenario: "AuctionScenario" = None,
    ) -> None:
        self.data_container = data_container
        self.partitioned_dataset = partitioning(
            self.data_container.dataset.customer
        )  # For fast get state function

        # scenario for construting customer information to bid auction
        self.scenario = scenario

        # -- data containers in model
        self.state = {}
        self.counter = 0

    def get_obeservation_space(self):
        return self.scenario.get_observation_space()

    def get_action_space(self):
        return self.scenario.get_action_space()

    def _is_has_product(self, generation: float, consumption: float) -> int:
        if generation - consumption > 0:
            return 1  # Seller Position
        else:
            return -1  # Buyer Position

    def get_size(self) -> int:
        return self.data_container.get_size()
    
    def get_current_time(self):
        return self.data_container.get_current_time(self.counter)

    def get_state(self) -> Dict[str, np.array]:
        # get state for each customer
        self.state = self.scenario.get_state(self)
        return self.state

    def run_auction(
        self,
        strategy: "AuctionContext",
        state: Dict[str, np.array],
        action: Tuple[Dict[str, int], float],
    ) -> Optional[Dict[str, list]]:

        transactions = strategy.run_auction(state, action)
        if transactions is not None:
            transactions["datetime"] = self.get_current_time()

        self.counter += 1

        return transactions

    def reset(self):
        self.counter = 0
        self.state = {}

        return self.get_state()


if __name__ == "__main__":
    pass
