# -*- coding: UTF-8 -*-

from dataclasses import dataclass
from typing import Type, Tuple, Any, Dict, Union

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

Observation = Type[np.ndarray]
Action = Type[int]
Reward = Type[float]
Experience = Type[Tuple[Observation, Action, Reward, Observation, bool]]
Environment = Type[Any]
Value = Type[float]

class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state: Observation) -> Action:
        x = F.relu(self.fc1(state))
        return F.softmax(self.fc2(x), dim=1)
    

class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state: Observation) -> Value:
        x = F.relu(self.fc1(state))
        return self.fc2(x)
    

class ActorCriticPolicy(object):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, learning_rate: float, gamma: float, device: str) -> None:
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.device = device

    def _as_tensor(self, x, dtype=torch.float) -> torch.Tensor:
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def take_action(self, state: Observation) -> Action:
        state = self._as_tensor([state])
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def update(self, transition_dict: Dict[str, Any]) -> None:
        states = self._as_tensor(transition_dict['states'])
        actions = self._as_tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1)
        rewards = self._as_tensor(transition_dict['rewards']).view(-1, 1)
        next_states = self._as_tensor(transition_dict['next_states'])
        dones = self._as_tensor(transition_dict['dones']).view(-1, 1)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()


@dataclass
class Params(object):
    pass


@dataclass
class A2CParams(Params):
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
        self, policy: ActorCriticPolicy, env: Environment, params: Params
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
    params = A2CParams()
    agent = ActorCriticPolicy(
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
