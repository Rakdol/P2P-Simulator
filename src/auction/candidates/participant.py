import random

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict

import numpy as np


class Participant(object):
    def __init__(self, id: int, action_space: np.array, has_product: bool = None):
        self.id = str(id)
        self.action_space = action_space
        self.has_product = has_product

    def __repr__(self) -> str:
        return f"Participant's id is {self.id}"

    def bid_action(self, maker):
        raise NotImplementedError()


class ZeroParticipant(Participant):
    def bid_action(self, maker):
        return maker


class LinearParticipant(Participant):
    def bid_action(self, maker):
        if self.has_product:
            return maker.ask[self.id].value

        if not self.has_product:
            return maker.bid[self.id].value


class StackelBergParticipant(Participant):
    def bid_action(self, maker):
        if self.has_product:
            return maker.ask[self.id].value

        if not self.has_product:
            return maker.bid[self.id].value


class SingleReinforceParticipant(Participant):
    def bid_action(self, maker):
        return super().bid_action(maker)
