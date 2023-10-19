# -*- coding: UTF-8 -*-

import collections
import copy
import random
from dataclasses import dataclass
from typing import Any, List, Tuple, Type, Union

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
Experience = Type[Tuple[Observation, Action, Reward, Observation, bool]]
Environment = gym.Env
Value = Type[float]


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


def as_tensor(x, dtype=torch.float, device="cpu") -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


class PolicyNet(nn.Module):
    def __init__(
        self, state_dim: int, hidden_dim: int, action_dim: int, action_bound: float
    ) -> None:
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, states: Observation) -> Action:
        x = F.relu(self.fc1(states))
        return torch.tanh(self.fc2(x)) * self.action_bound


class QValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, state: Observation, action: Action) -> Value:
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class Actor(object):
    def __init__(
        self,
        model: nn.Module,
        action_dim: int,
        sigma: float,
        learning_rate: float,
        tau: float,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.action_dim = action_dim
        self.sigma = sigma
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.tau = tau

    def __call__(self, states: Observation, *args: Any, **kwds: Any) -> float:
        return self.model(states)

    def values(self, states: Observation) -> float:
        return self.model(states)

    def take_action(self, state: Observation) -> Action:
        state = as_tensor([state], dtype=torch.float, device=self.device)
        action = self.model(state).item()
        return action + self.sigma * np.random.randn(self.action_dim)

    def update_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        self.optimizer.step()

    def max_q_values(self, state: Observation) -> float:
        state = as_tensor([state])
        return self.model(state).max().item()

    def load_state_dict(self, actor: "Actor") -> None:
        self.model.load_state_dict(actor.state_dict())

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def soft_update(self, net: nn.Module) -> None:
        for net_parameter, target_net_parameter in zip(
            net.parameters(), self.model.parameters()
        ):
            target_net_parameter.data.copy_(
                target_net_parameter.data * (1.0 - self.tau)
                + net_parameter.data * self.tau
            )


class Critic(object):
    def __init__(
        self,
        model: QValueNet,
        tau: float,
        learning_rate: float = 0.001,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = device
        self.tau = tau

    def estimate_return(self, state: Observation) -> Value:
        return self.model(state)

    def estimate_q_values(self, state: Observation, action: Action) -> Value:
        return self.model(state, action)

    def update_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        self.optimizer.step()

    def soft_update(self, net: QValueNet) -> None:
        for net_parameter, target_net_parameter in zip(
            net.parameters(), self.model.parameters()
        ):
            target_net_parameter.data.copy_(
                target_net_parameter.data * (1.0 - self.tau)
                + net_parameter.data * self.tau
            )


class DDPG(object):
    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        action_dim: int,
        sigma: float,
        tau: float,
        gamma: float,
        device: str = "cpu",
    ) -> None:
        self.actor = actor
        self.target_actor = copy.copy(self.actor)
        self.critic = critic
        self.target_critic = copy.copy(self.critic)
        self.action_dim = action_dim
        self.gamma = gamma
        self.sigma = sigma
        self.tau = tau
        self.device = device

    def take_action(self, state: Observation) -> Action:
        return self.actor.take_action(state)

    def update(self, trajectories: List[Trajectory]) -> None:
        states = as_tensor(
            [trajectory.state for trajectory in trajectories],
            dtype=torch.float,
            device=self.device,
        )
        actions = as_tensor(
            [trajectory.action for trajectory in trajectories], torch.int64, self.device
        ).view(-1, 1)
        rewards = as_tensor([trajectory.reward for trajectory in trajectories]).view(
            -1, 1
        )
        next_states = as_tensor(
            [trajectory.next_state for trajectory in trajectories],
            dtype=torch.float,
            device=self.device,
        )
        dones = as_tensor(
            [trajectory.done for trajectory in trajectories], torch.float, self.device
        ).view(-1, 1)

        next_q_values = self.target_critic.estimate_q_values(
            next_states, self.target_actor.values(next_states)
        )

        q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        critic_loss = torch.mean(
            F.mse_loss(self.critic.estimate_q_values(states, actions), q_targets)
        )
        self.critic.update_step(critic_loss)

        actor_loss = -torch.mean(
            self.critic.estimate_q_values(states, self.actor.values(states))
        )
        self.actor.update_step(actor_loss)

        self.target_actor.soft_update(self.actor.model)
        self.target_critic.soft_update(self.critic.model)


@dataclass
class Params(object):
    pass


@dataclass
class DDPGParams(Params):
    learning_rate: float = 1e-3
    num_episodes: int = 500
    gamma: float = 0.98
    epsilon: float = 0.01
    target_update: int = 10
    buffer_size: int = 10000
    minimal_size: int = 500
    device: Union[str, torch.device] = "cpu"
    epochs: int = 10
    seed: int = 0
    env_name: str = "CartPole-v0"
    hidden_dim: int = 128
    batch_size: int = 64
    sigma: float = 0.01
    tau: float = 0.005


class OffPolicyTrainer(object):
    def __init__(
        self,
        policy: DDPG,
        env: Environment,
        params: Params,
        replay_buffer: ReplayBuffer,
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
    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    params = DDPGParams()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    actor_net = PolicyNet(state_dim, params.hidden_dim, action_dim, action_bound)
    actor = Actor(
        actor_net,
        action_dim,
        params.sigma,
        params.learning_rate,
        params.tau,
        params.device,
    )
    critic_net = QValueNet(state_dim, params.hidden_dim, action_dim)
    critic = Critic(critic_net, params.tau, params.learning_rate, params.device)
    policy = DDPG(
        actor, critic, action_dim, params.sigma, params.tau, params.gamma, params.device
    )
    replay_buffer = ReplayBuffer(params.buffer_size)
    trainer = OffPolicyTrainer(policy, env, params, replay_buffer)
    trainer.learn()


if __name__ == "__main__":
    main()
