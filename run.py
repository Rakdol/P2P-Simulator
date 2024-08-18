# --- built in modules ---
import os
from typing import List

# --- Third party modules ---
import numpy as np
import pandas as pd

# --- Custom modules ---
from src.auction.config import *
from src.auction.candidates.market_operator import (
    MarketOperator,
    LinearOperator,
    StackelBergOperator,
    ZeroOperator,
)
from src.auction.candidates.participant import (
    Participant,
    StackelBergParticipant,
    LinearParticipant,
    ZeroParticipant,
)
from src.auction.candidates.optimizer import (
    ZeroOptimizer,
    LinearOptimizer,
    StackelBergOptimizer,
    Optimizer,
)
# from src.network.dist_operator import DistOperator

from src.auction.components.model import AuctionModel, DataContainer
from src.auction.components.view import AuctionView
from src.auction.components.controller import AuctionController
from src.auction.components.strategy import (
    AuctionContext,
    AuctionStrategy,
    UniformAuction,
)
from src.auction.components.scenario import BlackBox, AuctionScenario
from src.auction.components.reward import Reward, BenefitReward
from src.auction.env import AuctionEnv


def make_env(
    participant: Participant,
    operator: MarketOperator,
    # dist_operator: DistOperator,
    optimizer: Optimizer,
    scenario: AuctionScenario,
    reward: Reward,
    strategy: AuctionStrategy,
):

    # ---- Data Container ----
    data_container = DataContainer(
        customer_path=CUSTOMER_PATH,
        smp_path=SMP_PATH,
        price_path=PRICE_PATH,
        block_path=BLOCK_PATH,
    )

    # ----- Auction Components -----
    scenario = scenario()
    model = AuctionModel(
        data_container=data_container,
        scenario=scenario,
    )

    view = AuctionView()
    reward = reward()
    # dms = dist_operator()
    contoller = AuctionController(model, view, reward)

    # ---- Auction Participants ----
    participants = []
    for i in data_container.get_customer_ids():
        participants.append(participant(id=i, action_space=scenario.get_action_space()))

    operator = operator()
    optimizer = optimizer(operator=operator, participants=participants)
    strategy = AuctionContext(strategy())

    # ---- Auction Env ------
    env = AuctionEnv(controller=contoller, strategy=strategy)

    return env, optimizer


if __name__ == "__main__":

    env, optimizer = make_env(
        # participant=LinearParticipant,
        # operator=LinearOperator,
        # optimizer=LinearOptimizer,
        # dist_operator=DistOperator,
        participant=ZeroParticipant,
        operator=ZeroOperator,
        optimizer=ZeroOptimizer,
        # participant=StackelBergParticipant,
        # operator=StackelBergOperator,
        # optimizer=StackelBergOptimizer,
        scenario=BlackBox,
        strategy=UniformAuction,
        reward=BenefitReward,
    )
    from dqn import DQNAgent, convert_action
    
    obs_dim = len(env.customer_list) * env.observation_space[0]
    action_dim = len(env.customer_list) + 1
    agent = DQNAgent(state_size=obs_dim, action_size=action_dim, customer_list=env.customer_list)
    
    
    def convert_state(state):
        return np.array(list(state.values())).reshape(1, -1)
    
    MAX_EPISODE = 5
    batch_size = 64
    scores = []
    rewards = []
    IS_RENDER = True

    for epi in range(0, MAX_EPISODE):
        score = 0
        state = env.reset()
        done = False
        while not done:
            # actions = optimizer.get_actions(state=state)
            actions = agent.act(convert_state(state))
            
            next_state, reward, done, _ = env.step(action=actions)
            agent.remember(convert_state(state), convert_action(actions), reward, convert_state(next_state), done)
            score += reward
            if IS_RENDER:
                env.render()

            state = next_state
            
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        scores.append(score)
        avg_score = np.mean(scores)

        print(f"Episode {epi+1} / {MAX_EPISODE}", end="\n")
        print(f"Episode Score {score:.2f}", end="\n")
        print(f"Average_score {avg_score:.2f}", end="\n")
