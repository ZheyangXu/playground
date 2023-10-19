# -*- coding: UTF-8 -*-

import collections
import copy
import random
from dataclasses import dataclass
from typing import Any, List, Tuple, Type, Union

import gymnasium as gym
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


class PolicyNetContinuous(nn.Module):
    def __init__(
        self, state_dim: int, hidden_dim: int, action_dim: int, action_bound: float
    ) -> None:
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, states: Observation) -> Action:
        x = F.relu(self.fc1(states))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = torch.distributions.Normal(mu, std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        log_prob -= torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob


class QValueNetContinuous(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(QValueNetContinuous, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, states: Observation, actions: Action) -> Value:
        x = F.relu(self.fc1(torch.cat([states, actions], dim=1)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def as_tensor(x, dtype=torch.float, device="cpu") -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


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

    def take_action(self, states: Observation) -> Action:
        states = torch.tensor([states], dtype=torch.float).to(self.device)
        action = self.actor(states)[0]
        return [action.item()]

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
        model: QValueNetContinuous,
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

    def soft_update(self, net: QValueNetContinuous) -> None:
        for net_parameter, target_net_parameter in zip(
            net.parameters(), self.model.parameters()
        ):
            target_net_parameter.data.copy_(
                target_net_parameter.data * (1.0 - self.tau)
                + net_parameter.data * self.tau
            )


class SAC(object):
    def __init__(
        self,
        actor: Actor,
        critic1: Critic,
        critic2: Critic,
        alpha_learning_rate: float,
        target_entropy: float,
        tau: float,
        gamma: float,
        device: str = "cpu",
    ) -> None:
        self.actor = actor
        self.critic_1 = critic1
        self.critic_2 = critic2
        self.target_critic_1 = copy.copy(critic1)
        self.target_critic_2 = copy.copy(critic2)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_learning_rate)

        self.target_entropy = target_entropy
        self.tau = tau
        self.gamma = gamma
        self.device = device

    def take_action(self, states: Observation) -> Action:
        states = torch.tensor([states], dtype=torch.float).to(self.device)
        action = self.actor(states)[0]
        return [action.item()]

    def calc_target(
        self, rewards: Reward, next_states: Observation, dones: bool
    ) -> Value:
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1.estimate_q_values(next_states, next_actions)
        q2_value = self.target_critic_2.estimate_q_values(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

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
        # 和之前章节一样,对倒立摆环境的奖励进行重塑以便训练
        rewards = (rewards + 8.0) / 8.0

        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(
                self.critic_1.estimate_q_values(states, actions), td_target.detach()
            )
        )
        critic_2_loss = torch.mean(
            F.mse_loss(
                self.critic_2.estimate_q_values(states, actions), td_target.detach()
            )
        )
        self.critic_1.update_step(critic_1_loss)
        self.critic_2.update_step(critic_2_loss)

        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1.estimate_q_values(states, new_actions)
        q2_value = self.critic_2.estimate_q_values(states, new_actions)
        actor_loss = torch.mean(
            -self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value)
        )
        self.actor.update_step(actor_loss)

        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp()
        )
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.target_critic_1.soft_update(self.critic_1.model)
        self.target_critic_2.soft_update(self.critic_2.model)


@dataclass
class Params(object):
    pass


@dataclass
class SACParams(Params):
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
    alpha_learning_rate: float = 3e-3


class OffPolicyTrainer(object):
    def __init__(
        self,
        policy: SACX,
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
    params = SACParams()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    target_entropy = -env.action_space.shape[0]
    actor_net = PolicyNetContinuous(
        state_dim, params.hidden_dim, action_dim, action_bound
    )
    actor = Actor(
        actor_net,
        action_dim,
        params.sigma,
        params.learning_rate,
        params.tau,
        params.device,
    )
    critic_net1 = QValueNetContinuous(state_dim, params.hidden_dim, action_dim)
    critic1 = Critic(critic_net1, params.tau, params.learning_rate, params.device)
    critic_net2 = QValueNetContinuous(state_dim, params.hidden_dim, action_dim)
    critic2 = Critic(critic_net2, params.tau, params.learning_rate, params.device)
    policy = SAC(
        actor,
        critic1,
        critic2,
        params.alpha_learning_rate,
        target_entropy,
        params.tau,
        params.gamma,
        params.device,
    )
    replay_buffer = ReplayBuffer(params.buffer_size)
    trainer = OffPolicyTrainer(policy, env, params, replay_buffer)
    trainer.learn()


if __name__ == "__main__":
    main()
