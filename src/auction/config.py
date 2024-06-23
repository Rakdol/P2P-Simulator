import os
from dataclasses import dataclass

import pandas as pd

CUSTOMER_PATH: str = os.getcwd() + "\P2P-simulator\data\customers_env.csv"
SMP_PATH: str = os.getcwd() + "\P2P-simulator\data\smp_env.csv"
PRICE_PATH: str = os.getcwd() + "\P2P-simulator\data\prices.csv"
BLOCK_PATH: str = os.getcwd() + "\P2P-simulator\data\\block.csv"


@dataclass
class DataPath(object):
    customer_path: str
    smp_path: str
    price_path: str
    block_path: str


@dataclass
class DataSet(object):
    customer: pd.DataFrame
    smp: pd.DataFrame
    price: pd.DataFrame
    block: pd.DataFrame
