import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from torchvision import transforms

from ReplayBuffer import ReplayBuffer
from DQNCNN import DQNCNN


STACK_SIZE = 1

class DQNAgent:
    def __init__(self, DQNCNN:DQNCNN, action_size:int):
        self.action_size = action_size
        self.memory = ReplayBuffer(buffer_size=10000) 
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.update_frequency = 4

        # Create two networks: one for the current Q-function and one for the target Q-function
        self.q_network = DQNCNN
        self.target_network = DQNCNN
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)

        # Copy weights from the current network to the target network
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state, explore=True):
        # During exploration, select a random action
        if explore and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # During exploitation, select the action with the highest Q-value
        state = torch.FloatTensor(state).unsqueeze(0)  # Convert to tensor and add batch dimension
        q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to numpy arrays before converting to tensors
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Get current Q values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get target Q values
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
