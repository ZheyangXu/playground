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
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from tqdm import tqdm

Observation = Type[np.ndarray]
Action = Type[int]
Reward = Type[float]
Experience = Type[Tuple[Observation, Action, Reward, Observation, bool]]
Environment = Type[Any]


class ReplayBuffer(object):
    def __init__(self, capacity: int) -> None:
        self.buffer: List[Experience] = collections.deque(maxlen=capacity)

    def add(
        self,
        state: Observation,
        action: Action,
        reward: Reward,
        next_state: Observation,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Experience]:
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self) -> int:
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: Observation) -> Action:
        x = F.relu(self.fc1(state))
        return self.fc2(x)


class CNNQNet(nn.Module):
    def __init__(self, input_shape: Tuple[int], action_dim: int) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.in_channels = input_shape[0]
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, state: Observation) -> Action:
        x = self.net(state)
        x = self.fc(x)
        return x


class DQNPolicy(object):
    def __init__(
        self,
        observation_space_shape: int,
        action_dim: int,
        learning_rate: float,
        gamma: float,
        epsilon: float,
        target_update: int,
        device: str,
    ) -> None:
        self.action_dim = action_dim
        self.q_net = CNNQNet(observation_space_shape, action_dim).to(device)
        self.target_q_net = CNNQNet(observation_space_shape, action_dim).to(device)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device

    def _as_tensor(self, x, dtype=torch.float) -> torch.Tensor:
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def take_action(self, state: Observation) -> Action:
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = self._as_tensor([state])
            q_value = self.q_net(state)
            action = q_value.argmax().item()
        return action

    def update(self, transition_dic: Dict[str, Any]) -> None:
        states = self._as_tensor(transition_dic["states"])
        actions = self._as_tensor(transition_dic["actions"], torch.int64).view(-1, 1)
        rewards = self._as_tensor(transition_dic["rewards"]).view(-1, 1)
        next_states = self._as_tensor(transition_dic["next_states"])
        dones = self._as_tensor(transition_dic["dones"]).view(-1, 1)
        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
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
        self,
        policy: DQNPolicy,
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
                    episode_retrun = 0
                    state, _ = self.env.reset()
                    done = False
                    while not done:
                        action = self.policy.take_action(state)
                        next_state, reward, terminated, truncated, _ = self.env.step(
                            action
                        )
                        done = terminated or truncated
                        self.replay_buffer.add(state, action, reward, next_state, done)
                        state = next_state
                        episode_retrun += reward
                        if self.replay_buffer.size() > self.params.minimal_size:
                            (
                                batch_states,
                                batch_actions,
                                batch_rewards,
                                batch_next_states,
                                batch_dones,
                            ) = self.replay_buffer.sample(self.params.batch_size)
                            transition_dict = {
                                "states": batch_states,
                                "actions": batch_actions,
                                "rewards": batch_rewards,
                                "next_states": batch_next_states,
                                "dones": batch_dones,
                            }
                            self.policy.update(transition_dict)
                    return_list.append(episode_retrun)
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


class Experiment(object):
    def __init__(self, params: Params) -> None:
        pass


class SwapAxesToPyTorchWrapper(gym.ObservationWrapper):
    """
    Image shape to num_channels x weight x height and convert to Tensor.
    """

    def __init__(self, env):
        super(SwapAxesToPyTorchWrapper, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)


def main():
    env_name = "PongNoFrameskip-v4"
    env = SwapAxesToPyTorchWrapper(
        AtariPreprocessing(
            gym.make(env_name, render_mode="rgb_array"), grayscale_newaxis=True
        )
    )
    params = DQNParams()
    agent = DQNPolicy(
        env.observation_space.shape,
        env.action_space.n,
        params.learning_rate,
        params.gamma,
        params.epsilon,
        params.target_update,
        params.device,
    )
    replay_buffer = ReplayBuffer(params.buffer_size)
    trainer = OffPolicyTrainer(agent, env, params, replay_buffer)
    trainer.learn()


if __name__ == "__main__":
    main()
