# -*- coding: UTF-8 -*-

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type, Union

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


def as_tensor(x, dtype=torch.float, device="cpu") -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


class Actor(object):
    def __init__(
        self, model: PolicyNet, learning_rate: float = 0.001, device: str = "cpu"
    ) -> None:
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = device

    def take_action(self, state: Observation) -> Action:
        state = as_tensor([state], dtype=torch.float, device=self.device)
        probs = self.model(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()


class Critic(object):
    def __init__(
        self, model: ValueNet, learning_rate: float = 0.001, device: str = "cpu"
    ) -> None:
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = device

    def estimate_return(self, state: Observation) -> Value:
        return self.model(state)


def compute_advantage(gamma: float, lmbda: float, td_delta: torch.Tensor) -> float:
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class PPO(object):
    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        gamma: float = 0.9,
        eps: float = 0.02,
        epochs: int = 500,
        lmbda: float = 0.2,
        device: str = "cpu",
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

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

        td_target = rewards + self.gamma * self.critic.estimate_return(next_states) * (
            1 - dones
        )
        td_delta = td_target - self.critic.estimate_return(next_states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(
            self.device
        )
        old_log_probs = torch.log(self.actor.model(states).gather(1, actions)).detach()
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor.model(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic.estimate_return(states), td_target.detach())
            )
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor.optimizer.step()
            self.critic.optimizer.step()


@dataclass
class Params(object):
    pass


@dataclass
class PPOParams(Params):
    learning_rate: float = 1e-3
    num_episodes: int = 500
    gamma: float = 0.98
    lmbda: float = 0.95
    eps: float = 0.2
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


class OnPolicyTrainer(object):
    def __init__(self, policy: PPO, env: Environment, params: PPOParams) -> None:
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
                    trajectories = []
                    state, _ = self.env.reset()
                    done = False
                    while not done:
                        action = self.policy.actor.take_action(state)
                        next_state, reward, terminated, truncated, _ = self.env.step(
                            action
                        )
                        done = terminated or truncated
                        trajectories.append(
                            Trajectory(
                                state=state,
                                action=action,
                                reward=reward,
                                next_state=next_state,
                                done=done,
                            )
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
    params = PPOParams()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    actor_net = PolicyNet(state_dim, params.hidden_dim, action_dim)
    actor = Actor(actor_net, params.learning_rate, params.device)
    critic_net = ValueNet(state_dim, params.hidden_dim)
    critic = Critic(critic_net, params.learning_rate, params.device)
    policy = PPO(
        actor,
        critic,
        params.gamma,
        params.eps,
        params.epochs,
        params.lmbda,
        params.device,
    )
    trainer = OnPolicyTrainer(policy, env, params)
    trainer.learn()


if __name__ == "__main__":
    main()
