# -*- coding: UTF-8 -*-

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

StateType = Type[np.ndarray]
ActionType = Type[Union[int, np.ndarray, Any]]
LossType = Type[Any]
ModelType = Type[Union[nn.Module, np.ndarray]]


@dataclass
class Transition(object):
    state: StateType
    action: ActionType
    reward: float
    done: bool
    next_state: StateType


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: StateType) -> ActionType:
        x = F.relu(self.fc1(state))
        return F.softmax(self.fc2(x), dim=1)


class BaseActor(ABC):
    @abstractmethod
    def take_action(self, state: StateType) -> ActionType:
        pass

    @abstractmethod
    def update_fn(self, loss: LossType) -> None:
        pass


class Actor(BaseActor):
    def __init__(self, model: ModelType, device: str, optimizer: optim.Optimizer):
        self.model = model
        self.device = device
        self.optimzer = optimizer

    def take_action(self, state: StateType) -> ActionType:
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.model(state)
        action_dist = torch.distributions.Categorical(probs=probs)
        action = action_dist.sample()
        return action.item()

    def update_fn(self, loss: LossType) -> None:
        self.optimzer.zero_grad()
        loss.backward()
        self.optimzer.step()


class BaseCritic(ABC):
    @abstractmethod
    def estimate(self, state: StateType) -> float:
        pass

    @abstractmethod
    def update_fn(self, loss: LossType) -> None:
        pass


class BaseAlgorithm(ABC):
    @abstractmethod
    def take_action(self, state: StateType) -> ActionType:
        pass

    @abstractmethod
    def estimate_return(self, transitions: List[Transition]) -> float:
        pass

    @abstractmethod
    def update_step(self, transitions: List[Transition]) -> None:
        pass


class Reinforce(BaseAlgorithm):
    def __init__(
        self, actor: BaseActor, learning_rate: float, gamma: float, device: str
    ) -> None:
        super(Reinforce, self).__init__()
        self.actor = actor
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = device

    def take_action(self, state: StateType) -> ActionType:
        return self.actor.take_action(state)

    def estimate_return(self, transitions: List[Transition]) -> float:
        pass

    def update_step(self, transitions: List[Transition]) -> None:
        pass
