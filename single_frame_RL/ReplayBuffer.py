import random
import numpy as np
from collections import deque

# Replay buffer to store past experiences
class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.memory = deque(maxlen=buffer_size)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)