from typing import List
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from config import *


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


    def push(self, state, action, reward, next_state, done):

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        index = ['state', 'action', 'reward', 'next', 'done']
        dic = dict(zip(index, [state, action, reward, next_state, done]))
        self.buffer[self.position] = dic
        self.position = int((self.position + 1) % self.capacity)


    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = [], [], [], [], []

        for i in range(batch_size):
            state.append(batch[i]['state'])
            action.append(batch[i]['action'])
            reward.append(batch[i]['reward'])
            next_state.append(batch[i]['next'])
            done.append(batch[i]['done'])

        return state, action, reward, next_state, done


    def __len__(self):
        return len(self.buffer)


class ActorNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, init_w=3e-3):

        super().__init__()
        self.action_dim = output_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.linear3 = nn.Linear(hidden_dim[1], output_dim)

        # weight initialization
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)


    def forward(self, state):
        x = torch.tanh(self.linear1(state))
        x = torch.tanh(self.linear2(x))
        x = torch.sigmoid(self.linear3(x)).clone()
        return x


    def select_action(self, state, noise=0, noise_scale=0.5):
        '''
        select action for sampling, no gradients flow, noisy action, return .cpu
        '''
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self(state)

        noise = noise_scale * Normal(0, noise).sample(action.shape).to(device)
        action = torch.clamp(action + noise, 0, 1)

        return action.detach().cpu().numpy()[0]


    @staticmethod
    def sample_action():
        normal = Normal(0.5, 1).sample((1,))
        random_action = torch.clamp(normal, 0.001, 1)
        return random_action.cpu().numpy()


    def evaluate_action(self, state, noise_scale=0.0):
        '''
        evaluate action within GPU graph, for gradients flowing through it, noise_scale controllable
        '''
        action = self(state)
        noise = noise_scale * Normal(0, 1).sample(action.shape).to(device)
        action = action + noise
        return action


class QNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, init_w=3e-3):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.linear3 = nn.Linear(hidden_dim[1], 1)

        # weight initialization
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)


    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = self.linear3(x)
        return x


class DDPG:

    def __init__(self, replay_buffer, state_dim, action_dim, hidden_dim):

        q_lr = 1e-4
        policy_lr = 1e-5

        self.replay_buffer: ReplayBuffer = replay_buffer

        self.q_net = QNetwork(state_dim + action_dim, hidden_dim).to(device)
        self.target_qnet = QNetwork(state_dim + action_dim, hidden_dim).to(device)
        self.policy_net = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)

        print(self.q_net)
        print(self.policy_net)

        self.target_qnet.load_state_dict(self.q_net.state_dict())

        self.q_criterion = nn.MSELoss()
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.update_cnt = 0


    def target_soft_update(self, net, target_net, tau):

        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

        return target_net


    def update(self, batch_size, gamma=0.99, soft_tau=1e-2, target_update_delay=3):

        self.update_cnt += 1
        states, actions, rewards, next_states, done_flags = self.replay_buffer.sample(batch_size)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        done_flags = torch.FloatTensor(done_flags).unsqueeze(1).to(device)

        predicted_q_values = []
        target_q_values = []

        for i in range(batch_size):
            current_state = torch.FloatTensor(states[i]).to(device)
            current_action = torch.FloatTensor(actions[i]).to(device)
            predicted_q = torch.mean(self.q_net(current_state, current_action), 0)

            next_predicted_q = 0
            if len(next_states[i]) > 0 and len(next_states[i][0]) > 0:
                next_state = torch.FloatTensor(next_states[i]).to(device)
                next_action = self.target_policy_net.evaluate_action(next_state)
                next_predicted_q = torch.mean(self.target_qnet(next_state, next_action), 0)

            target_q = rewards[i] + (1 - done_flags[i]) * gamma * next_predicted_q

            target_q_values.append(target_q)
            predicted_q_values.append(predicted_q)

        predicted_q_values = torch.stack(predicted_q_values).to(device)
        target_q_values = torch.stack(target_q_values).to(device).detach()

        # train q network
        self.q_optimizer.zero_grad()
        q_loss = self.q_criterion(predicted_q_values, target_q_values)
        q_loss.backward()
        self.q_optimizer.step()

        predicted_q_values = []

        for i in range(batch_size):
            current_state = torch.FloatTensor(states[i]).to(device)
            predicted_action = self.policy_net.evaluate_action(current_state)
            predicted_q = torch.mean(self.q_net(current_state, predicted_action), 0)
            predicted_q_values.append(predicted_q)

        predicted_q_values = torch.stack(predicted_q_values).to(device)

        # train policy network
        self.policy_optimizer.zero_grad()
        policy_loss = -torch.mean(predicted_q_values)
        policy_loss.backward()
        self.policy_optimizer.step()

        # update the target_qnet
        if self.update_cnt % target_update_delay == 0:
            self.target_qnet = self.target_soft_update(self.q_net, self.target_qnet, soft_tau)
            self.target_policy_net = self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

        return q_loss.detach().cpu().numpy(), policy_loss.detach().cpu().numpy()


    def save_model(self, path):
        torch.save(self.q_net.state_dict(), path + '_q')
        torch.save(self.target_qnet.state_dict(), path + '_target_q')
        torch.save(self.policy_net.state_dict(), path + '_policy')

    def load_model(self, path):
        self.q_net.load_state_dict(torch.load(path + '_q'))
        self.target_qnet.load_state_dict(torch.load(path + '_target_q'))
        self.policy_net.load_state_dict(torch.load(path + '_policy'))
        self.q_net.eval()
        self.target_qnet.eval()
        self.policy_net.eval()

    def plot(self, rewards, edf_rewards, lsf_rewards):
        plt.close("all")
        plt.figure(figsize=(20, 5))
        plt.plot(range(len(rewards)), rewards, color='green', label='DDPG')
        # plt.plot(range(len(edf_rewards)), edf_rewards, color='red', label='EDF')
        # plt.plot(range(len(lsf_rewards)), lsf_rewards, color='blue', label='LSF')
        plt.xlabel('Episode')
        plt.ylabel('Scheduling Success Ratio ')
        plt.legend()
        plt.ylim(0, 100)
        plt.savefig('plot/ddpg VS edf delay.png')
        # plt.show()
        plt.clf()