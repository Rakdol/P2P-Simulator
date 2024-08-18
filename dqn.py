from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
import numpy as np

def convert_action(actions):
    x = list(actions[0].values())
    x.append(actions[1])
    return np.array(x)


class DQNAgent:
    def __init__(self, state_size, action_size, customer_list):
        self.state_size = state_size
        self.action_size = action_size
        self.customer_list = customer_list
        self.memory = deque(maxlen=2000)
        self.gamma = 0.98    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='sigmoid'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        action_dict = {}
        if np.random.rand() <= self.epsilon:
            act_values = np.random.rand(self.action_size)
            for i, cid in enumerate(self.customer_list):
                action_dict[cid] = act_values[i]
            
            return action_dict, act_values[-1]
        
        act_values = self.model.predict(state, verbose=False)
        
        for i, cid in enumerate(self.customer_list):
            action_dict[cid] = act_values[0][i]
        
        return action_dict, act_values[0][-1]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state, verbose=False)[0]))
            target_f = self.model.predict(state, verbose=False)
            target_f[0][np.argmax(action)] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay