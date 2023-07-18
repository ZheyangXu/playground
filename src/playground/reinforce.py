# -*- coding: UTF-8 -*-
from dataclasses import dataclass
from typing import Any, Dict, Type, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

Observation = Type[np.ndarray]
ActionProbs = Type[np.ndarray]
Action = Type[int]
Environment = Type[Any]


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: Observation) -> ActionProbs:
        x = F.relu(self.fc1(state))
        return F.softmax(self.fc2(x), dim=1)


class REINFORCEPolicy(object):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        learning_rate: float,
        gamma: float,
        device: str,
    ) -> None:
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )
        self.gamma = gamma
        self.device = device

    def take_action(self, state: Observation) -> Action:
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs=probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict: Dict[str, Any]) -> None:
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor(action_list[i]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G
            loss.backward()
        self.optimizer.step()


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
    device: Union[str, torch.device] = "cuda"
    epochs: int = 10
    seed: int = 0
    env_name: str = "CartPole-v0"
    hidden_dim: int = 128
    batch_size: int = 64


class OnPolicyTrainer(object):
    def __init__(
        self, policy: REINFORCEPolicy, env: Environment, params: Params
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
                    transition_dict = {
                        "states": [],
                        "actions": [],
                        "next_states": [],
                        "rewards": [],
                        "dones": [],
                    }
                    state, _ = self.env.reset()
                    done = False
                    while not done:
                        action = self.policy.take_action(state)
                        next_state, reward, terminated, truncated, _ = self.env.step(
                            action
                        )
                        done = terminated or truncated
                        transition_dict["states"].append(state)
                        transition_dict["actions"].append(action)
                        transition_dict["next_states"].append(next_state)
                        transition_dict["rewards"].append(reward)
                        transition_dict["dones"].append(done)
                        state = next_state
                        episode_return += reward
                    return_list.append(episode_return)
                    self.policy.update(transition_dict)
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
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    params = REINFORCEParams()
    agent = REINFORCEPolicy(
        env.observation_space.shape[0],
        params.hidden_dim,
        env.action_space.n,
        params.learning_rate,
        params.gamma,
        params.device
    )
    trainer = OnPolicyTrainer(agent, env, params)
    trainer.learn()


if __name__ == '__main__':
    main()
