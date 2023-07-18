# -*- coding: UTF-8 -*-

import collections
import random
from typing import Any, Dict, List, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from jingwei import *
from playground.jingwei import ActionType, DistributionType, LossType, ObservationType


class Experiences(object):
    def __init__(self) -> None:
        pass

    def add(self):
        pass


class DequeReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int) -> None:
        self.buffer: List[Experience] = collections.deque(maxlen=capacity)
    
    def add(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)
    
    def size(self) -> int:
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: ObservationType) -> ActionType:
        x = F.relu(self.fc1(state))
        return self.fc2(x)


class BaseActor(Actor):
    def __init__(self, 
                 model: Any,
                 action_dim: int,
                 epsilon: float) -> None:
        self.model = model
        self.action_dim = action_dim
        self.epsilon = epsilon

    def take_action(self, observation: List[ObservationType]) -> List[ActionType]:
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            action = self.model(observation)
        return action
    
    def update(self, loss: LossType) -> None:
        return super().update(loss)
    
    def get_action_distribution(self, observation: List[ObservationType]) -> DistributionType:
        return super().get_action_distribution(observation)