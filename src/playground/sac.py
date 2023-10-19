# -*- coding: UTF-8 -*-

import collections
import random
from typing import Any, Dict, List, Tuple, Type

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


class SAC(object):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        action_bound: float,
        actor_learning_rate: float,
        critic_learning_rate: float,
        alpha_learning_rate: float,
        target_entropy: float,
        tau: float,
        gamma: float,
        device: str = "cpu",
    ) -> None:
        self.actor = PolicyNetContinuous(
            state_dim, hidden_dim, action_dim, action_bound
        ).to(device)
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(
            device
        )
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(
            device
        )
        self.target_critic_1 = QValueNetContinuous(
            state_dim, hidden_dim, action_dim
        ).to(device)
        self.target_critic_2 = QValueNetContinuous(
            state_dim, hidden_dim, action_dim
        ).to(device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate
        )
        self.critic_1_optimizer = optim.Adam(
            self.critic_1.parameters(), lr=critic_learning_rate
        )
        self.critic_2_optimizer = optim.Adam(
            self.critic_2.parameters(), lr=critic_learning_rate
        )

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
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net: nn.Module, target_net: nn.Module) -> None:
        for parameter, target_net_parameter in zip(
            net.parameters(), target_net.parameters()
        ):
            target_net_parameter.data.copy_(
                target_net_parameter.data * (1 - self.tau) + parameter.data * self.tau
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
        # 和之前章节一样,对倒立摆环境的奖励进行重塑以便训练
        rewards = (rewards + 8.0) / 8.0

        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach())
        )
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach())
        )
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(
            -self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value)
        )
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp()
        )
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


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
    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    actor_learning_rate = 3e-4
    critic_learning_rate = 3e-3
    alpha_learning_rate = 3e-4
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.99
    tau = 0.005
    buffer_size = 100000
    minimal_size = 1000
    batch_size = 64
    target_entropy = -env.action_space.shape[0]
    device = "cpu"
    replay_buffer = ReplayBuffer(buffer_size)
    agent = SAC(
        state_dim,
        hidden_dim,
        action_dim,
        action_bound,
        actor_learning_rate,
        critic_learning_rate,
        alpha_learning_rate,
        target_entropy,
        tau,
        gamma,
        device,
    )

    train_off_policy_agent(
        env, agent, num_episodes, replay_buffer, minimal_size, batch_size
    )


if __name__ == "__main__":
    main()
