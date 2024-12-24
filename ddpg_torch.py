#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/9


import math
import random
from config import *
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
from collections import namedtuple

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display

import argparse

if GPU:
    device = torch.device("cuda:" + str(DEVICE_INDEX) if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
else:
    device = torch.device("cpu")
torch.autograd.set_detect_anomaly(True)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        index = ['state', 'action', 'reward', 'next', 'done']
        dic = dict(zip(index, [state, action, reward, next_state, done]))
        self.buffer[self.position] = dic
        # self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state = []
        action = []
        reward = []
        next_state = []
        done = []
        for i in range(batch_size):
            state.append(batch[i]['state'])
            action.append(batch[i]['action'])
            reward.append(batch[i]['reward'])
            next_state.append(batch[i]['next'])
            done.append(batch[i]['done'])
        # state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element

        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, init_w=3e-3):
        super(ActorNetwork, self).__init__()
        self.action_dim = output_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.linear3 = nn.Linear(hidden_dim[1], output_dim)  # output dim = dim of action

        # weights initialization
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        activation = torch.tanh
        x = activation(self.linear1(state))
        x = activation(self.linear2(x))
        x = torch.sigmoid(self.linear3(x)).clone()  # for simplicity, no restriction on action rang
        return x

    def select_action(self, state, noise=0, noise_scale=0.5):
        '''
        select action for sampling, no gradients flow, noisy action, return .cpu
        '''
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # state dim: (N, dim of state)
        normal = Normal(0, noise)
        action = self.forward(state)
        noise = noise_scale * normal.sample(action.shape).to(device)
        action += noise
        # action = torch.from_numpy(np.clip(action.detach().numpy(), 0, 1)[0])
        torch.clamp(action, 0, 1)
        return action.detach().cpu().numpy()[0]

    @staticmethod
    def sample_action(action_range=1.):
        normal = Normal(0.5, 1)
        random_action = torch.clamp(normal.sample((1,)), 0.001, 1)
        return random_action.cpu().numpy()

    def evaluate_action(self, state, noise_scale=0.0):
        '''
        evaluate action within GPU graph, for gradients flowing through it, noise_scale controllable
        '''
        normal = Normal(0, 1)
        action = self.forward(state)
        # action = torch.tanh(action)
        noise = noise_scale * normal.sample(action.shape).to(device)
        action += noise
        return action


class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, init_w=3e-3):
        super(QNetwork, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.linear3 = nn.Linear(hidden_dim[1], 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)  # the dim 0 is number of samples
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = self.linear3(x)
        return x


class DDPG:
    def __init__(self, replay_buffer, state_dim, action_dim, hidden_dim):
        self.replay_buffer = replay_buffer
        self.q_net = QNetwork(state_dim + action_dim, hidden_dim).to(device)
        self.target_qnet = QNetwork(state_dim + action_dim, hidden_dim).to(device)
        self.policy_net = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)

        print('Q network: ', self.q_net)
        print('Policy network: ', self.policy_net)

        for target_param, param in zip(self.target_qnet.parameters(), self.q_net.parameters()):
            target_param.data.copy_(param.data)
        self.q_criterion = nn.MSELoss()
        q_lr = 1e-4
        policy_lr = 1e-5
        self.update_cnt = 0

        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

    def target_soft_update(self, net, target_net, soft_tau):
        # Soft update the target net
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return target_net

    def update(self, batch_size, reward_scale=10.0, gamma=0.99, soft_tau=1e-2, policy_up_itr=10, target_update_delay=3,
               warmup=True):
        self.update_cnt += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)
        # print(reward.shape,state)
        predict_q = []
        predict_new_q = []
        target_q = []
        for i in range(batch_size):
            state1 = torch.FloatTensor(state[i]).to(device)
            next_state1 = torch.FloatTensor(next_state[i]).to(device)
            action1 = torch.FloatTensor(action[i]).to(device)
            pq = torch.mean(self.q_net(state1, action1), 0)
            nna = self.target_policy_net.evaluate_action(next_state1)
            na = self.policy_net.evaluate_action(state1)
            predict_q.append(pq)
            pnq = torch.mean(self.q_net(state1, na), 0)
            predict_new_q.append(pnq)
            tq = reward[i] + (1 - done[i]) * gamma * torch.mean(self.target_qnet(next_state1, nna), 0)
            target_q.append(tq)
        # state = torch.FloatTensor(state).to(device)
        # next_state = torch.FloatTensor(next_state).to(device)
        # action = torch.FloatTensor(action).to(device)
        # reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        # done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        # predict_q = self.q_net(state, action)  # for q
        # new_next_action = self.target_policy_net.evaluate_action(next_state)  # for q
        # new_action = self.policy_net.evaluate_action(state)  # for policy
        # predict_new_q = self.q_net(state, new_action)  # for policy
        # target_q = reward + (1 - done) * gamma * self.target_qnet(next_state, new_next_action)  # for q
        # reward = reward_scale * (reward - reward.mean(dim=0)) /reward.std(dim=0) # normalize with batch mean and std
        predict_q = torch.stack(predict_q).to(device)
        target_q = torch.stack(target_q).to(device)
        predict_new_q = torch.stack(predict_new_q).to(device)

        # train qnet
        q_loss = self.q_criterion(predict_q, target_q.detach())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # train policy_net
        policy_loss = -torch.mean(predict_new_q)
        self.policy_optimizer.zero_grad()
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
        plt.plot(range(len(edf_rewards)), edf_rewards, color='red', label='EDF')
        plt.plot(range(len(lsf_rewards)), lsf_rewards, color='blue', label='LSF')
        plt.xlabel('Episode')
        plt.ylabel('Scheduling Success Ratio ')
        plt.legend()
        plt.ylim(0, 100)
        plt.savefig('plot/ddpg VS edf delay.png')
        # plt.show()
        plt.clf()