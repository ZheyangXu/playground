# -*- coding: UTF-8 -*-

import copy
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


def hessian_matrix_vector_product(
    states: Observation,
    actor: Actor,
    old_action_dists: torch.distributions.Categorical,
    vector: torch.Tensor,
) -> torch.Tensor:
    new_action_dists = torch.distributions.Categorical(actor.model(states))
    kl = torch.mean(
        torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists)
    )
    kl_grad = torch.autograd.grad(kl, actor.model.parameters(), create_graph=True)
    kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
    kl_grad_vector_product = torch.dot(kl_grad_vector, vector)

    grad2 = torch.autograd.grad(kl_grad_vector_product, actor.model.parameters())
    return torch.cat([grad.view(-1) for grad in grad2])


def conjugate_gradient(
    grad: torch.Tensor,
    actor: Actor,
    states: Observation,
    old_action_dist: torch.distributions.Categorical,
    iteration_num: int = 10,
    thre: float = 1e-10,
) -> torch.Tensor:
    x = torch.zeros_like(grad)
    r = grad.clone()
    p = grad.clone()
    rdotr = torch.dot(r, r)
    for _ in range(iteration_num):
        hessian_matrix = hessian_matrix_vector_product(
            states, actor, old_action_dist, p
        )
        alpha = rdotr / torch.dot(p, hessian_matrix)
        x += alpha * p
        r -= alpha * p
        new_rdotr = torch.dot(r, r)
        if new_rdotr < thre:
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
    return x


def compute_surrogate_obj(
    states: Observation,
    actor: Actor,
    actions: Action,
    advantage: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> torch.Tensor:
    log_probs = torch.log(actor.model(states).gather(1, actions))
    ratio = torch.exp(log_probs - old_log_probs)
    return torch.mean(ratio * advantage)


def line_search(
    states: Observation,
    actor: Actor,
    actions: Action,
    advantage: float,
    old_log_probs: torch.Tensor,
    old_action_dists: torch.distributions.Categorical,
    max_vec: torch.Tensor,
    kl_constraint: float = 0.0005,
    alpha: float = 0.5,
    max_iterate_num: int = 15,
) -> torch.Tensor:
    old_parameters = nn.utils.convert_parameters.parameters_to_vector(
        actor.model.parameters()
    )
    old_obj = compute_surrogate_obj(states, actor, actions, advantage, old_log_probs)
    for i in range(max_iterate_num):
        coef = alpha**i
        new_parameters = old_parameters + coef * max_vec
        new_actor = copy.deepcopy(actor)
        nn.utils.convert_parameters.vector_to_parameters(
            new_parameters, new_actor.model.parameters()
        )
        new_action_dists = torch.distributions.Categorical(new_actor.model(states))
        kl_div = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists)
        )
        new_obj = compute_surrogate_obj(
            states, new_actor, actions, advantage, old_log_probs
        )
        return (
            new_parameters
            if new_obj > old_obj and kl_div < kl_constraint
            else old_parameters
        )


class TRPO(object):
    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        gamma: float = 0.9,
        kl_constraint: float = 0.02,
        epochs: int = 500,
        lmbda: float = 0.2,
        device: str = "cpu",
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.kl_constraint = kl_constraint
        self.device = device

    def policy_learn(
        self,
        states: Observation,
        actions: Action,
        old_action_dists: torch.distributions.Categorical,
        old_log_probs: torch.Tensor,
        advantage: torch.Tensor,
    ) -> None:
        surrogate_obj = compute_surrogate_obj(
            states, self.actor, actions, advantage, old_log_probs
        )
        grads = torch.autograd.grad(surrogate_obj, self.actor.model.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads])
        descent_direction = conjugate_gradient(
            obj_grad, self.actor, states, old_action_dists
        )
        hessian_matrix = hessian_matrix_vector_product(
            states, self.actor, old_action_dists, descent_direction
        )
        max_coef = torch.sqrt(
            2
            * self.kl_constraint
            / (torch.dot(descent_direction, hessian_matrix) + 1e-8)
        )
        new_parameters = line_search(
            states,
            self.actor,
            actions,
            advantage,
            old_log_probs,
            old_action_dists,
            descent_direction * max_coef,
        )
        nn.utils.convert_parameters.vector_to_parameters(
            new_parameters, self.actor.model.parameters()
        )

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
        old_action_dists = torch.distributions.Categorical(
            self.actor.model(states).detach()
        )
        critic_loss = torch.mean(
            F.mse_loss(self.critic.estimate_return(states), td_target.detach())
        )
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        self.policy_learn(states, actions, old_action_dists, old_log_probs, advantage)


@dataclass
class Params(object):
    pass


@dataclass
class TRPOParams(Params):
    learning_rate: float = 1e-3
    num_episodes: int = 500
    gamma: float = 0.98
    lmbda: float = 0.95
    eps: float = 0.2
    epsilon: float = 0.01
    kl_constraint: float = 0.005
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
    def __init__(self, policy: TRPO, env: Environment, params: TRPOParams) -> None:
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
    params = TRPOParams()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    actor_net = PolicyNet(state_dim, params.hidden_dim, action_dim)
    actor = Actor(actor_net, params.learning_rate, params.device)
    critic_net = ValueNet(state_dim, params.hidden_dim)
    critic = Critic(critic_net, params.learning_rate, params.device)
    policy = TRPO(
        actor,
        critic,
        params.gamma,
        params.kl_constraint,
        params.epochs,
        params.lmbda,
        params.device,
    )
    trainer = OnPolicyTrainer(policy, env, params)
    trainer.learn()


if __name__ == "__main__":
    main()
