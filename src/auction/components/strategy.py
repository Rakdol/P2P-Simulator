from typing import Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AuctionStrategy(object):
    def __init__(self) -> None:
        self.matched_order = {
            "datetime": [],
            "seller": [],
            "buyer": [],
            "clearing": [],
            "quantity": [],
            "bid": [],
            "reserved": [],
        }
        self.non_matched_order = {"seller": [], "buyer": [], "bid": [], "quantity": []}

    def make_bid_price(self, upper_bound, lower_bound, action):
        raise NotImplementedError()

    def get_orderbook(
        self,
        state: Dict[str, np.array],
        action_dict: Dict[str, Union[int, float]],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        seller = {"id": [], "quantity": [], "bid_price": [], "reserved": []}
        buyer = {"id": [], "quantity": [], "bid_price": [], "reserved": []}

        for cid, action in action_dict.items():
            consumption = state[str(cid)][0]
            generation = state[str(cid)][1]
            quantity = generation - consumption
            lower_bound = state[str(cid)][2]
            upper_bound = state[str(cid)][3]
            has_product = state[str(cid)][4]

            bid_price = self.make_bid_price(
                lower_bound=lower_bound, upper_bound=upper_bound, action=action
            )

            if has_product > 0:
                # Seller
                seller["id"].append(cid)
                seller["quantity"].append(has_product * quantity)
                seller["bid_price"].append(bid_price)
                seller["reserved"].append(lower_bound)
            else:
                # Buyer
                buyer["id"].append(cid)
                buyer["quantity"].append(has_product * quantity)
                buyer["bid_price"].append(bid_price)
                buyer["reserved"].append(upper_bound)

        seller_book = (
            pd.DataFrame(seller)
            .sort_values(by=["bid_price"], ascending=False)
            .reset_index(drop=True)
        )

        buyer_book = (
            pd.DataFrame(buyer)
            .sort_values(by=["bid_price"], ascending=True)
            .reset_index(drop=True)
        )

        return seller_book, buyer_book

    def transaction(self, sellers, buyers):
        raise NotImplementedError()


class UniformAuction(AuctionStrategy):

    def make_bid_price(
        self, action: Union[int, float], upper_bound: float, lower_bound: float
    ) -> float:
        if action > 1:
            return action

        bid_price = lower_bound * action + upper_bound * (1 - action)

        return bid_price

    def get_clearing_price(
        self,
        buyers: pd.DataFrame,
        supply_limit: float,
        fee_rate: float,
    ) -> Dict[str, float]:
        accumulated_demand = 0
        clearing_price = 0
        for row in buyers.itertuples(index=False):
            accumulated_demand += row.quantity
            if accumulated_demand > supply_limit:
                clearing_price = row.bid_price
                break

        return {
            "clearing_price": clearing_price * (1 - fee_rate),
            "operator_benefit": clearing_price * fee_rate,
        }

    def match(
        self, sellers: pd.DataFrame, buyers: pd.DataFrame, fee_rate: float
    ) -> pd.DataFrame:
        transactions = []
        seller_supply = sellers.set_index("bid_price")["quantity"].to_dict()
        supply_limit = sellers["quantity"].sum()

        market_dict = self.get_clearing_price(buyers, supply_limit, fee_rate)

        for buyer in buyers.itertuples(index=False):
            buyer_demand = buyer.quantity
            for seller in sellers.itertuples(index=False):
                seller_price = seller.bid_price

                if (
                    buyer.reserved >= market_dict["clearing_price"]
                    and seller.reserved <= market_dict["clearing_price"]
                    and buyer_demand > 0
                    and seller_supply[seller_price] > 0
                ):
                    sold_quantity = min(buyer_demand, seller_supply[seller_price])
                    transactions.append(
                        {
                            "buyer_id": buyer.id,
                            "buyer_price": buyer.bid_price,
                            "buyer_bound": buyer.reserved,
                            "buyer_benefit": buyer.reserved
                            - market_dict["clearing_price"],
                            "seller_id": seller.id,
                            "seller_price": seller_price,
                            "seller_bound": seller.reserved,
                            "seller_benefit": market_dict["clearing_price"]
                            - seller.reserved,
                            "quantity": sold_quantity,
                        }
                    )

                    # update demand and supply
                    buyer_demand -= sold_quantity
                    seller_supply[seller_price] -= sold_quantity
                    # # set clearning price as the last traded price
                    # last_traded_price = seller_price

        # add clearning price
        for transaction in transactions:
            transaction["clearing_price"] = market_dict["clearing_price"]
            transaction["operator_benefit"] = market_dict["operator_benefit"]
            transaction["fee_rate"] = fee_rate

        transaction_df = pd.DataFrame(transactions)

        return transaction_df

    def transaction(
        self, state: Dict[str, np.array], action: Tuple[Dict[str, int], float]
    ) -> Union[Dict[str, list], None]:
        participants_action = action[0]
        fee_rate = action[1]

        sellers, buyers = self.get_orderbook(state, participants_action)

        # UniformAuction
        if sellers.empty:
            print("No participants completed the transaction.")
            return None

        transactions = self.match(sellers=sellers, buyers=buyers, fee_rate=fee_rate)

        return transactions


class DoubleAuction(AuctionStrategy):
    def make_bid_price(self, upper_bound, lower_bound, action):
        pass

    def transaction(self, bids):
        pass


class AuctionContext(object):
    def __init__(self, auction_strategy: AuctionStrategy):
        self._strategy = auction_strategy

    @property
    def strategy(self) -> None:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: AuctionStrategy) -> None:
        print(f"Change strategy from {self._strategy} to {strategy}")
        self._strategy = strategy

    def run_auction(
        self, state: Dict[str, np.array], action: Tuple[Dict[str, int], float]
    ) -> Optional[Dict[str, list]]:

        return self._strategy.transaction(state, action)
