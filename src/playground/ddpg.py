# -*- coding: UTF-8 -*-

import collections
import random
from typing import Any, Tuple, Type

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


class DDPG(object):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        action_bound: float,
        sigma: float,
        actor_learning_rate: float,
        critic_learning_rate: float,
        tau: float,
        gamma: float,
        device: str = "cpu",
    ) -> None:
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(
            device
        )
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(
            state_dim, hidden_dim, action_dim, action_bound
        ).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_learning_rate
        )

        self.gamma = gamma
        self.sigma = sigma
        self.tau = tau
        self.action_dim = action_dim
        self.device = device

    def take_action(self, state: Observation) -> Action:
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).item()
        return action + self.sigma * np.random.randn(self.action_dim)

    def soft_update(self, net: nn.Module, target_net: nn.Module) -> None:
        for net_parameter, target_net_parameter in zip(
            net.parameters(), target_net.parameters()
        ):
            target_net_parameter.data.copy_(
                target_net_parameter.data * (1.0 - self.tau)
                + net_parameter.data * self.tau
            )

    def update(self, transition_dict: Any) -> None:
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(
            self.device
        )
        actions = (
            torch.tensor(transition_dict["actions"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)


def train_off_policy_agent(
    env, agent, num_episodes, replay_buffer, minimal_size, batch_size
):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            "states": b_s,
                            "actions": b_a,
                            "next_states": b_ns,
                            "rewards": b_r,
                            "dones": b_d,
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "episode": "%d" % (num_episodes / 10 * i + i_episode + 1),
                            "return": "%.3f" % np.mean(return_list[-10:]),
                        }
                    )
                pbar.update(1)
    return return_list


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def main():
    actor_lr = 3e-4
    critic_lr = 3e-3
    num_episodes = 200
    hidden_dim = 64
    gamma = 0.98
    tau = 0.005
    buffer_size = 10000
    minimal_size = 1000
    batch_size = 64
    sigma = 0.01
    device = "cpu"

    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    agent = DDPG(
        state_dim,
        hidden_dim,
        action_dim,
        action_bound,
        sigma,
        actor_lr,
        critic_lr,
        tau,
        gamma,
        device,
    )
    train_off_policy_agent(
        env, agent, num_episodes, replay_buffer, minimal_size, batch_size
    )


if __name__ == "__main__":
    main()
