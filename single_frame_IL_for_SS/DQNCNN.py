import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from torchvision import transforms

class DQNCNN(nn.Module):
    def __init__(self, action_size, input_channels=3, input_size=(48, 48)):
        super(DQNCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dynamically determine the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, *input_size)
            dummy_output = self.pool(torch.relu(self.conv1(dummy_input)))
            self.flattened_size = dummy_output.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),  # Smaller input
        transforms.ToTensor(),        # Output shape: [3, 32, 32]
    ])
    return transform(frame)
