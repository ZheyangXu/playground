# -*- coding: UTF-8 -*-

from dataclasses import dataclass
from typing import Any, Dict, List, Type, Union, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

Observation = Type[np.ndarray]
ActionProbs = Type[np.ndarray]
Action = Type[int]
Reward = Type[float]
Environment = Type[Any]


@dataclass
class Trajectory(object):
    state: Observation
    action: Action
    reward: Reward
    next_state: Observation
    done: bool


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: Observation) -> ActionProbs:
        x = F.relu(self.fc1(state))
        return F.softmax(self.fc2(x), dim=1)


class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, states: Observation) -> Tuple[torch.Tensor, torch.Tensor]:
        states = F.relu(self.fc1(states))
        mu = 2.0 * torch.tanh(self.fc_mu(states))
        std = F.softplus(self.fc_std(states))
        return mu, std


def as_tensor(x, dtype=torch.float, device="cpu") -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


class Actor(object):
    def __init__(
        self,
        model: PolicyNet,
        learning_rate: float = 0.001,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = device

    def take_action(self, state: Observation) -> Action:
        state = as_tensor([state], torch.float, self.device)
        probs = self.model(state)
        action_dist = torch.distributions.Categorical(probs=probs)
        action = action_dist.sample()
        return action.item()

    def update_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        self.optimizer.step()

    def get_log_probs(self, states: Observation, actions: Action) -> torch.Tensor:
        return torch.log(self.model(states).gather(1, actions)).detach()


class GaussianContinuousActor(object):
    def __init__(
        self,
        model: PolicyNetContinuous,
        learning_rate: float = 0.001,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = device

    def take_action(self, state: Observation) -> Action:
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, sigma = self.model(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return [action.item()]

    def update_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        self.optimizer.step()

    def get_log_probs(self, states: Observation, actions: Action) -> torch.Tensor:
        mu, std = self.model(states)
        action_dist = torch.distributions.Normal(mu.detach(), std.detach())
        return action_dist.log_prob(actions)


@dataclass
class Params(object):
    pass


@dataclass
class REINFORCEParams(Params):
    learning_rate: float = 1e-3
    num_episodes: int = 500
    gamma: float = 0.98
    epsilon: float = 0.01
    target_update: int = 10
    buffer_size: int = 10000
    minimal_size: int = 500
    device: Union[str, torch.device] = "mps"
    epochs: int = 10
    seed: int = 0
    env_name: str = "CartPole-v0"
    hidden_dim: int = 128
    batch_size: int = 64


class Reinforce(object):
    def __init__(self, actor: Actor, gamma: float, device: float = "cpu") -> None:
        self.actor = actor
        self.gamma = gamma
        self.device = device

    def take_action(self, state: Observation) -> Action:
        return self.actor.take_action(state)

    def update(self, transitions: List[Trajectory]) -> None:
        rewards = [trajectory.reward for trajectory in transitions]
        states = [trajectory.state for trajectory in transitions]
        actions = [trajectory.action for trajectory in transitions]

        G = 0
        loss = 0
        for i in reversed(range(len(rewards))):
            reward = rewards[i]
            state = as_tensor([states[i]], dtype=torch.float, device=self.device)
            action = as_tensor(actions[i], dtype=torch.int64, device=self.device).view(
                -1, 1
            )
            log_prob = self.actor.get_log_probs(state, action)
            G = self.gamma * G + reward
            loss += -log_prob * G
        self.actor.update_step(loss)
        

class OnPolicyTrainer(object):
    def __init__(
        self, policy: Reinforce, env: gym.Env, params: REINFORCEParams
    ) -> None:
        self.policy = policy
        self.env = env
        self.params = params

    def learn(self) -> None:
        return_list = []
        for i in range(self.params.epochs):
            with tqdm(
                total=int(self.params.num_episodes / 10), desc="Iteration %d" % i
            ) as pbar:
                for i_episode in range(int(self.params.num_episodes / 10)):
                    episode_return = 0
                    state, _ = self.env.reset()
                    done = False
                    trajectories = []
                    while not done:
                        action = self.policy.take_action(state)
                        next_state, reward, terminated, truncated, _ = self.env.step(
                            action
                        )
                        done = terminated or truncated
                        trajectories.append(
                            Trajectory(state, action, reward, next_state, done)
                        )
                        state = next_state
                        episode_return += reward
                    return_list.append(episode_return)
                    self.policy.update(trajectories)
                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix(
                            {
                                "episode": (
                                    self.params.num_episodes / 10 * i + i_episode + 1
                                ),
                                "return": np.mean(return_list[-10:]),
                            }
                        )
                    pbar.update(1)


def main():
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    params = REINFORCEParams()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = PolicyNet(state_dim, params.hidden_dim, action_dim)
    actor = Actor(model, params.learning_rate, params.device)
    policy = Reinforce(actor, params.gamma, params.device)
    trainer = OnPolicyTrainer(policy, env, params)
    trainer.learn()
    
    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    params = REINFORCEParams()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = PolicyNetContinuous(state_dim, params.hidden_dim, action_dim)
    actor = GaussianContinuousActor(model, params.learning_rate, params.device)
    policy = Reinforce(actor, params.gamma, params.device)
    trainer = OnPolicyTrainer(policy, env, params)
    trainer.learn()

if __name__ == "__main__":
    main()
