from typing import List, Tuple
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from config import (
    GPU, DEVICE_INDEX,
    DISCOUNT_RATE,
    CRITIC_STATE_DIM,
    ACTOR_STATE_DIM,
    ACTION_DIM,
    ACTOR_HIDDEN_DIM,
    CRITIC_HIDDEN_DIM,
    BATCH_SIZE,
    Q_LEARNING_RATE,
    POLICY_LEARNING_RATE,
    TARGET_UPDATE_DELAY,
    SOFT_UPDATE_TAU
)

if GPU:
    cuda = "cuda:" + str(DEVICE_INDEX)
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    print("cuda available:", torch.cuda.is_available())
else:
    device = torch.device("cpu")

torch.autograd.set_detect_anomaly(True)


class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer: List[dict] = []
        self.position = 0


    def __len__(self):
        return len(self.buffer)


    def push(
            self,
            state_actor: np.ndarray,
            state_critic: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state_actor: np.ndarray,
            next_state_critic: np.ndarray,
            done: bool,
            time_duration: float
        ):
        """
        Stores a transition in the replay buffer.

        Args:
            state_actor (np.ndarray): Actor-specific state at time t.
            state_critic (np.ndarray): Critic-specific state at time t.
            action (np.ndarray): Action taken at time t.
            reward (float): Scalar reward received after taking the action.
            next_state_actor (np.ndarray): Actor-specific state at time t+1.
            next_state_critic (np.ndarray): Critic-specific state at time t+1.
            done (bool): Whether the episode has ended after this transition.
            time_duration (float): The actual time duration elapsed during this transition.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        index = [
            'state_actor', 'state_critic', 'action', 'reward',
            'next_actor', 'next_critic', 'done', 'time_duration'
        ]
        values = [
            state_actor, state_critic, action, reward,
            next_state_actor, next_state_critic, done, time_duration
        ]
        dic = dict(zip(index, values))
        self.buffer[self.position] = dic
        self.position = int((self.position + 1) % self.capacity)


    def sample(self, batch_size: int):
        """
        Samples a batch of transitions from the replay buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple of lists for actor states, critic states, actions, rewards,
            next actor states, next critic states, done flags, and time durations.
        """
        batch = random.sample(self.buffer, batch_size)

        states_actor = [item['state_actor'] for item in batch]
        states_critic = [item['state_critic'] for item in batch]
        actions = [item['action'] for item in batch]
        rewards = [item['reward'] for item in batch]
        next_states_actor = [item['next_actor'] for item in batch]
        next_states_critic = [item['next_critic'] for item in batch]
        done_flags = [item['done'] for item in batch]
        time_durations = [item['time_duration'] for item in batch]

        return (
            states_actor, states_critic, actions, rewards,
            next_states_actor, next_states_critic, done_flags, time_durations
        )


class ActorNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, init_w=3e-3):
        super().__init__()
        self.normal_distribution = Normal(0, 1)

        self.linear1 = nn.Linear(input_dim, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.linear3 = nn.Linear(hidden_dim[1], output_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)


    def forward(self, state):
        x = torch.tanh(self.linear1(state))
        x = torch.tanh(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x


    def select_action(self, state: np.ndarray, noise_width: float = 0.0) -> np.ndarray:
        """
        Selects actions for the given states without any gradient flow.
        Adds optional Gaussian noise (for exploration).

        Args:
            state (np.ndarray): States for which to select actions. Shape: (num_instances, state_dim)
            noise_std (float, optional): Standard deviation of Gaussian noise to add to actions.

        Returns:
            np.ndarray: Selected actions. Shape: (num_instances, action_dim)
        """
        state = torch.FloatTensor(state).to(device)

        with torch.no_grad():
            action_tensor = self(state)

        lower_bound = action_tensor - noise_width
        upper_bound = action_tensor + noise_width

        upper_shift = (upper_bound - 1.0).clamp(min=0)
        lower_shift = (lower_bound - 0.0).clamp(max=0).abs()

        final_lower = lower_bound - upper_shift + lower_shift
        final_upper = upper_bound - upper_shift + lower_shift

        random_sample = torch.rand_like(action_tensor)
        scaled_sample = random_sample * (final_upper - final_lower)

        action = final_lower + scaled_sample

        return action.cpu().numpy().astype(np.float32)


class QNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, init_w=3e-3):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.linear3 = nn.Linear(hidden_dim[1], 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)


    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the estimated Q-value for a given state-action pair.

        Args:
            state (torch.Tensor): State tensor. Shape: (num_instances, state_dim)
            action (torch.Tensor): Action tensor. Shape: (num_instances, action_dim)

        Returns:
            torch.Tensor: Computed rewards. Shape: (num_instances, 1)
        """
        x = torch.cat([state, action], 1)
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = self.linear3(x)
        return x


class MADDPG:
    """
    Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm.

    Args:
        replay_buffer (ReplayBuffer): Experience replay buffer.
        gamma (float): Discount factor for future rewards.
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        hidden_dim (List[int]): List of two integers specifying the sizes of the two hidden layers.
        q_lr (float): Learning rate for the Q-network (critic).
        policy_lr (float): Learning rate for the policy network (actor).
        target_update_delay (int): Number of steps between target network updates.
    """
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = BATCH_SIZE,
        gamma: float = DISCOUNT_RATE,
        critic_state_dim: int = CRITIC_STATE_DIM,
        actor_state_dim: int = ACTOR_STATE_DIM,
        action_dim: int = ACTION_DIM,
        actor_hidden_dim: List[int] = ACTOR_HIDDEN_DIM,
        critic_hidden_dim: List[int] = CRITIC_HIDDEN_DIM,
        q_lr: float = Q_LEARNING_RATE,
        policy_lr: float = POLICY_LEARNING_RATE,
        target_update_delay: int = TARGET_UPDATE_DELAY
    ):
        self.replay_buffer: ReplayBuffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma

        self.policy_net = ActorNetwork(actor_state_dim, action_dim, actor_hidden_dim).to(device)
        self.target_policy_net = ActorNetwork(actor_state_dim, action_dim, actor_hidden_dim).to(device)
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.q_net = QNetwork(critic_state_dim + action_dim, critic_hidden_dim).to(device)
        self.target_q_net = QNetwork(critic_state_dim + action_dim, critic_hidden_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=q_lr)
        self.q_criterion = nn.MSELoss()

        self.target_update_delay = target_update_delay
        self.update_cnt = 0


    def update(self, soft_tau: float = SOFT_UPDATE_TAU) -> Tuple[float, float]:
        """
        Performs a single update step for the actor and critic networks using a batch of experiences.

        Args:
            batch_size (int): Number of transitions to sample from the replay buffer.
            soft_tau (float): Soft update coefficient for target networks.

        Returns:
            Tuple[float, float]: Average Q-network loss and policy-network loss for this update step.
        """
        self.update_cnt += 1
        batch = self.replay_buffer.sample(self.batch_size)
        states_actor, states_critic, actions, rewards, next_states_actor, next_states_critic, done_flags, time_durations = batch

        predicted_q_values = []
        target_q_values = []

        for i in range(self.batch_size):

            current_state_critic = torch.FloatTensor(states_critic[i]).to(device)
            current_action = torch.FloatTensor(actions[i]).to(device)
            reward = torch.FloatTensor([rewards[i]]).to(device)

            current_q_value = torch.mean(self.q_net(current_state_critic, current_action), 0)

            target_q_value = reward
            if not done_flags[i]:

                if next_states_actor[i].size > 0:
                    next_state_actor = torch.FloatTensor(next_states_actor[i]).to(device)
                    with torch.no_grad():
                        next_action = self.target_policy_net(next_state_actor)
                else:
                    next_action = np.zeros((1, ACTION_DIM), dtype=np.float32)
                    next_action = torch.FloatTensor(next_action).to(device)
                
                next_state_critic = torch.FloatTensor(next_states_critic[i]).to(device)
                with torch.no_grad():
                    next_q_value = torch.mean(self.target_q_net(next_state_critic, next_action), 0)

                target_q_value += self.gamma ** time_durations[i] * next_q_value

            predicted_q_values.append(current_q_value)
            target_q_values.append(target_q_value)

        predicted_q_values = torch.stack(predicted_q_values).to(device)
        target_q_values = torch.stack(target_q_values).to(device).detach()

        # train q network
        self.q_optimizer.zero_grad()
        q_loss = self.q_criterion(predicted_q_values, target_q_values)
        q_loss.backward()
        self.q_optimizer.step()

        predicted_q_values = []

        for i in range(self.batch_size):
            if states_actor[i].size > 0:
                current_state_actor = torch.FloatTensor(states_actor[i]).to(device)
                current_state_critic = torch.FloatTensor(states_critic[i]).to(device)
                predicted_action = self.policy_net(current_state_actor)
                predicted_q = torch.mean(self.q_net(current_state_critic, predicted_action), 0)
                predicted_q_values.append(predicted_q)

        predicted_q_values = torch.stack(predicted_q_values).to(device)

        # train policy network
        self.policy_optimizer.zero_grad()
        policy_loss = -torch.mean(predicted_q_values)
        policy_loss.backward()
        self.policy_optimizer.step()

        # update the target_qnet
        if self.update_cnt % self.target_update_delay == 0:

            self.target_q_net = self.target_soft_update(
                self.q_net, self.target_q_net, soft_tau
            )

            self.target_policy_net = self.target_soft_update(
                self.policy_net, self.target_policy_net, soft_tau
            )

        return q_loss.detach().cpu().numpy(), policy_loss.detach().cpu().numpy()


    def target_soft_update(self, net: nn.Module, target_net: nn.Module, tau: float) -> nn.Module:
        """
        Performs a soft update of the target network parameters.

        Args:
            net (nn.Module): Source network (policy or Q-network).
            target_net (nn.Module): Target network to be updated.
            tau (float): Soft update coefficient (0 < tau < 1).

        Returns:
            nn.Module: Updated target network.
        """
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
        return target_net


    def save_model(self, path: str):
        """
        Saves the parameters of the Q-network, target Q-network, policy network,
        and target policy network.

        Args:
            path (str): Directory path where model weights will be saved.
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.q_net.state_dict(), os.path.join(path, 'q_net.pth'))
        torch.save(self.target_q_net.state_dict(), os.path.join(path, 'target_q_net.pth'))
        torch.save(self.policy_net.state_dict(), os.path.join(path, 'policy_net.pth'))
        torch.save(self.target_policy_net.state_dict(), os.path.join(path, 'target_policy_net.pth'))


    def load_model(self, path: str):
        """
        Loads the parameters of the Q-network, target Q-network, policy network,
        and target policy network.

        Args:
            path (str): Directory path where model weights are saved.
        """
        self.q_net.load_state_dict(torch.load(os.path.join(path, 'q_net.pth')))
        self.target_q_net.load_state_dict(torch.load(os.path.join(path, 'target_q_net.pth')))
        self.policy_net.load_state_dict(torch.load(os.path.join(path, 'policy_net.pth')))
        self.target_policy_net.load_state_dict(torch.load(os.path.join(path, 'target_policy_net.pth')))