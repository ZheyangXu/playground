# -*- coding: UTF-8 -*-

import collections
import random
from dataclasses import dataclass
from typing import  Any, Dic, List, Tuple, Type, Union
from abc import ABCMeta, abstractmethod

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ObservationType = Type[np.ndarray]
ObservationsType = Type[List[ObservationType]]
ValueType = Type[float]
AdvantagesType = Type[float]
ActionType = Type[int]
ActionsType = Type[List[ActionType]]
RewardType = Type[float]
Environment = Type[Any]
LossType = Type[Union[float, torch.Tensor]]
DistributionType = Type[Any]


@dataclass
class Experience(object):
    observation: ObservationType
    action: ActionType
    reward: RewardType
    next_observation: ObservationType
    done: bool


class Experiences(ABCMeta):
    """
    Experiences
    """
    @abstractmethod
    def add(self, *args) -> bool:
        pass
        
    @abstractmethod
    def batch_data(self, *args) -> Any:
        pass


class ReplayBuffer(ABCMeta):
    """
    Implent ReplayBuffer
    """

    @abstractmethod
    def add(self, transition: Experience) -> None:
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> Experience:
        pass

    @abstractmethod
    def size(self) -> int:
        pass
    

class Actor(ABCMeta):
    """
    The Actor
    """

    @abstractmethod
    def take_action(self, observation: List[ObservationType]) -> List[ActionType]:
        pass
    
    @abstractmethod
    def update(self, loss: LossType) -> None:
        pass
    
    @abstractmethod
    def get_action_distribution(self, observation: List[ObservationType]) -> DistributionType:
        pass


class Critic(ABCMeta):
    """
    The critic
    """
    @abstractmethod
    def critic(self, observation: ObservationType) -> ValueType:
        pass

    @abstractmethod
    def update(self, loss: LossType) -> None:
        pass


class Policy(ABCMeta):
    """
    The Policy
    """
    
    @abstractmethod
    def learn(self) -> None:
        pass

    @abstractmethod
    def advanatge_fn(self) -> AdvantagesType:
        pass

    @abstractmethod
    def loss_fn(self) -> LossType:
        pass

    @abstractmethod
    def take_action(self, observation: List[ObservationType]) -> List[ActionType]:
        pass


class Trainer(ABCMeta):
    """
    Trainer
    """
    @abstractmethod
    def train(self) -> None:
        pass


@dataclass
class Params(object):
    learning_rate: float = 1e-3
    epochs: int = 10
    discount_factor: float = 1

