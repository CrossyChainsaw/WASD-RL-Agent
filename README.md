# DQN Agent with Convolutional Neural Network for Game Control

This project implements a **Deep Q-Network (DQN)** using a **Convolutional Neural Network (CNN)** to interact with and control a Windows Forms-based game. The DQN agent learns from visual input, and the CNN processes game frames to make decisions in real-time. The agent aims to optimize its behavior using reinforcement learning techniques by interacting with the game environment.

## Table of Contents
- [Project Structure](#project-structure)
- [Agent Design](#agent-design)
- [Environment](#environment)
- [Training Results](#training-results)
- [Testing Results](#testing-results)

## Project Structure
- `DQNCNN.py`: Defines the Convolutional Neural Network used as the Q-network.
- `DQNAgent.py`: The agent class that interacts with the environment, chooses actions, and learns from experiences.
- `ReplayBuffer.py`: Manages the replay memory to store past experiences.
- `WASDEnv.py`: Custom Gym environment for interacting with the Windows Forms game.
- `rl.ipynb`: Script that initializes the environment, agent, and handles the training process.

## Agent Design
### Neural Network (CNN)
The CNN architecture consists of the following layers:

- Conv2d layer with 64 filters, kernel size 3, and stride 2
- MaxPooling layer with a kernel size of 2
- Fully connected (Dense) layers:
  - 64 * 20 * 20 -> 512 neurons
  - 512 -> action_size (4 actions: W, A, S, D)
The CNN extracts features from the game frames and predicts the Q-values for each possible action.

### Deep Q-Learning
The agent follows the Deep Q-Learning algorithm, which optimizes its policy by:

- Selecting actions using an ε-greedy strategy (exploration vs exploitation)
- Storing experiences in a replay buffer and training using random mini-batches
- Using a target network to stabilize learning

## Environment
The custom environment is based on Gymnasium and interacts with a Windows Forms game by capturing screenshots and simulating keypresses (W, A, S, D). It processes the game screen using grayscale conversion and resizing to feed the CNN with an 84x84 image.

### Key Components:
- Action Space: 4 possible actions (W, A, S, D)
- Observation Space: Grayscale screenshots of the full screen, resized to 84x84 pixels
- Rewards: The agent receives a reward of 1 for a successful action and 0 for an unsuccessful action.

## Training Results
Training was conducted over a series of 130 episodes. Below is a plot that shows the total reward per episode during training:

## Testing Results
The agent was tested over two episodes, each achieving a total reward of 20,000:
