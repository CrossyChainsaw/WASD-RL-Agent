import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from torchvision import transforms

# Define the CNN Q-network
class DQNCNN(nn.Module):
    def __init__(self, action_size, stack_size):
        super(DQNCNN, self).__init__()
        self.conv1 = nn.Conv2d(stack_size, 64, kernel_size=3, stride=2)  # Using default stride and padding
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces size further

        # Adjust this size based on your calculations
        self.fc1 = nn.Linear(64 * 20 * 20, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)  # Apply max pooling to reduce size further
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
