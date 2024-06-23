from typing import Union
from collections import defaultdict

import pandas as pd


class Reward(object):
    def __init__(self):
        self.reward_dict = defaultdict(int)
    
    def get_reward(self, transactions:pd.DataFrame):
        raise NotImplementedError()
    

class BenefitReward(Reward):
    def get_reward(self, transactions: Union[pd.DataFrame, None]):
        if transactions is None:
            return 0
        
        for transaction in transactions.itertuples(index=False):
            self.reward_dict[transaction.buyer_id] += transaction.buyer_benefit
            self.reward_dict[transaction.seller_id] += transaction.seller_benefit
            self.reward_dict["operator"] += transaction.operator_benefit
            
        return sum(self.reward_dict.values())
