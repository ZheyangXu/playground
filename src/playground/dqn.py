# -*- coding: UTF-8 -*-

import collections
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

Observation = Type[np.ndarray]
Action = Type[int]
Reward = Type[float]
Environment = Type[gym.Env]


@dataclass
class Trajectory(object):
    state: Observation
    action: Action
    reward: Reward
    next_state: Observation
    done: bool


class ReplayBuffer(object):
    def __init__(self, capacity: int) -> None:
        self.buffer: List[Trajectory] = collections.deque(maxlen=capacity)

    def add(self, trajectory: Trajectory) -> int:
        self.buffer.append(trajectory)
        return self.size()

    def size(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int) -> List[Trajectory]:
        transitions = random.sample(self.buffer, batch_size)
        return transitions


class QNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: Observation) -> Action:
        x = F.relu(self.fc1(state))
        return self.fc2(x)


class VAnet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(VAnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_a = nn.Linear(hidden_dim, action_dim)
        self.fc_v = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> float:
        a = self.fc_a(F.relu(self.fc1(state)))
        v = self.fc_v(F.relu(self.fc1(state)))
        return v + a - a.mean(1).view(-1, 1)


def as_tensor(x, dtype=torch.float, device="cpu") -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


class Actor(object):
    def __init__(
        self,
        model: nn.Module,
        action_dim: int,
        episilon: float,
        learning_rate: float,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.action_dim = action_dim
        self.episilon = episilon
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def __call__(self, states: Observation, *args: Any, **kwds: Any) -> float:
        return self.model(states)

    def values(self, states: Observation) -> float:
        return self.model(states)

    def take_action(self, state: Observation) -> Action:
        if np.random.random() < self.episilon:
            action = np.random.randint(self.action_dim)
        else:
            state = as_tensor([state])
            action = self.model(state).argmax().item()
        return action

    def update_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        self.optimizer.step()

    def max_q_values(self, state: Observation) -> float:
        state = as_tensor([state])
        return self.model(state).max().item()


class DQN(object):
    def __init__(
        self, actor: Actor, gamma: float, target_update: int = 2, dqn_type="VanillaDQN"
    ) -> None:
        self.actor = actor
        self.target_actor = actor
        self.gamma = gamma
        self.dqn_type = dqn_type
        self.target_update = target_update
        self.count = 0

    def take_action(self, state: Observation) -> Action:
        return self.actor.take_action(state)

    def update(self, transitions: List[Trajectory]) -> None:
        states = as_tensor([trajectory.state for trajectory in transitions])
        actions = as_tensor([trajectory.action for trajectory in transitions]).view(
            -1, 1
        )
        rewards = as_tensor([trajectory.reward for trajectory in transitions]).view(
            -1, 1
        )
        next_states = as_tensor([trajectory.next_state for trajectory in transitions])
        dones = as_tensor([trajectory.done for trajectory in transitions]).view(-1, 1)

        q_values = self.actor.model(states).gather(
            1, as_tensor(actions, dtype=torch.int64)
        )
        max_next_q_values = self.target_actor.model(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.actor.update_step(loss)

        if self.count % self.target_update == 0:
            self.target_actor.model.load_state_dict(self.actor.model.state_dict())
        self.count += 1


@dataclass
class Params(object):
    pass


@dataclass
class DQNParams(Params):
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


class OffPolicyTrainer(object):
    def __init__(
        self, policy: DQN, env: Environment, params: Params, replay_buffer: ReplayBuffer
    ) -> None:
        self.policy = policy
        self.env = env
        self.params = params
        self.replay_buffer = replay_buffer

    def learn(self) -> None:
        return_list = []
        for i in range(self.params.epochs):
            with tqdm(
                total=int(self.params.num_episodes / 10), desc="Interation %d" % i
            ) as pbar:
                for i_episode in range(int(self.params.num_episodes / 10)):
                    episode_return = 0
                    state, _ = self.env.reset()
                    done = False
                    while not done:
                        action = self.policy.take_action(state)
                        next_state, reward, terminated, truncated, _ = self.env.step(
                            action
                        )
                        done = terminated or truncated
                        trajectory = Trajectory(state, action, reward, next_state, done)
                        self.replay_buffer.add(trajectory)
                        state = next_state
                        episode_return += reward
                        if self.replay_buffer.size() > self.params.minimal_size:
                            trajectories = self.replay_buffer.sample(
                                self.params.batch_size
                            )
                            self.policy.update(trajectories)

                    return_list.append(episode_return)
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
    params = DQNParams()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = QNet(state_dim, params.hidden_dim, action_dim)
    actor = Actor(
        model, action_dim, params.epsilon, params.learning_rate, params.device
    )
    policy = DQN(actor, params.gamma)
    replay_buffer = ReplayBuffer(params.buffer_size)
    trainner = OffPolicyTrainer(policy, env, params, replay_buffer)
    trainner.learn()


if __name__ == "__main__":
    main()
