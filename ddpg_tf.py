#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/3/2

"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import math
from config import *

LR_A = 0.0005  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 100
BATCH_SIZE = 16


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, model, retrain=True):
        # self.memory = np.zeros((MEMORY_CAPACITY, 100, 13) , dtype=np.float32)
        self.memory = []
        self.pointer = 0
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0
        self.model = model
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.O = tf.placeholder(tf.float32, [None, s_dim], 'o')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.L = tf.placeholder(tf.float32, [None], "L")
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.QA = tf.placeholder(tf.float32, [None, 1], 'qa')
        self.QC = tf.placeholder(tf.float32, [None, 1], 'qc')
        self.Q_ = tf.placeholder(tf.float32, [None, 1], "q_")
        self.retrain = retrain

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            self.q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            self.q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        # in the feed_dic for the td_error, the self.a should change to actions in memory
        target_q = self.R + GAMMA * self.q_
        print(target_q)
        td_error = tf.losses.mean_squared_error(labels=self.Q_, predictions=self.QC)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
        a_loss = - tf.reduce_mean(self.QA)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=4)
        ckpt = tf.train.get_checkpoint_state('./' + model)
        if ckpt and ckpt.model_checkpoint_path and not self.retrain:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def choose_action(self, s):
        return self.sess.run(self.a, {self.O: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = []
        length = []
        for i in indices:
            bt.append(self.memory[i])
            length.append(len(self.memory[i][0]))
        # bt = self.memory[indices]
        bs = bt[0][0]
        ba = bt[0][1]
        br = bt[0][2]
        bs_ = bt[0][3]

        mean_qa = np.array([])
        mean_qc = np.array([])
        mean_q_ = np.array([])

        r = np.array([])
        for i in range(len(bt)):
            with tf.variable_scope('Actor'):
                print(bt[i][0])
                a = self._build_a(np.array(bt[i][0], dtype=np.float32), scope='eval', trainable='True')
                a_ = self._build_a(np.array(bt[i][3], dtype=np.float32), scope='target', trainable='False')
            with tf.variable_scope('Critic'):
                qa = self._build_c(np.array(bt[i][0], dtype=np.float32), a, scope='eval', trainable='True')
                qc = self._build_c(np.array(bt[i][0], dtype=np.float32), np.array(bt[i][1], dtype=np.float32), scope='eval', trainable='True')
                q_ = self._build_c(np.array(bt[i][3], dtype=np.float32), a_, scope='target', trainable='False')
            print(a)
            mean_qa = np.append(mean_qa, tf.reduce_mean(qa))
            mean_qc = np.append(mean_qc, tf.reduce_mean(qc))
            mean_q_ = np.append(mean_q_, tf.reduce_mean(q_))
            r = np.append(r, bt[i][2])

        print(mean_qa)
        mean_qa = tf.stack(mean_qa.tolist())
        mean_qc = tf.stack(mean_qc.tolist())
        mean_q_ = tf.stack(mean_q_.tolist())

        # bs = bt[:, :self.s_dim]
        # ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        # br = bt[:, -self.s_dim - 1: -self.s_dim]
        # bs_ = bt[:, -self.s_dim:]
        # print(length)
        index = 0
        # bs = np.array(bs, dtype=np.float32)
        # ba = np.array(ba, dtype=np.float32)
        # br = np.array(br, dtype=np.float32)
        # bs_ = np.array(bs_, dtype=np.float32)

        # with tf.variable_scope('Actor'):
        #     a = self._build_a(bs, scope='eval', trainable='True')
        #     a_ = self._build_a(bs_, scope='target', trainable='False')
        # with tf.variable_scope('Critic'):
        #     qa = self._build_c(bs, a, scope='eval', trainable='True')
        #     qc = self._build_c(bs, ba, scope='eval', trainable='True')
        #     q_ = self._build_c(bs_, a_, scope='target', trainable='False')

        # self.sess.run(self.atrain, {self.q: mean_qa[:, np.newaxis]})
        # self.sess.run(self.ctrain, {self.q: mean_qc[:, np.newaxis], self.Q_: mean_q_[:, np.newaxis], self.R: br})
        target_q = r + GAMMA * mean_q_
        print(tf.convert_to_tensor(target_q))
        print(target_q)
        print(mean_qc)
        td_error = tf.losses.mean_squared_error(labels=self.sess.run(target_q), predictions=mean_qc)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
        a_loss = - tf.reduce_mean(mean_qa)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
        print(self.sess.run(td_error))

        variable_names = [v.name for v in tf.variable]
        values = self.sess.run(variable_names)
        for k, v in zip(variable_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)
            #print(v)

        self.sess.run(self.ctrain)
        self.sess.run(self.atrain)


    def save(self, episode):
        self.saver.save(self.sess, self.model + '/' + self.model, global_step=episode)

    def store_transition(self, s, a, r, s_):
        #list = ['state', 'action', 'reward', 'next']
        #dic = dict(zip(list, [s, a, r, s_]))
        # transition = np.hstack((s, a, r, s_))
        transition = [s, a, r, s_]
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        if self.pointer < MEMORY_CAPACITY:
            self.memory.append(transition)
        else:
            self.memory[index] = transition
        self.pointer += 1

    def mean_q(self, q, length):
        index = 0
        mean = np.array([])
        for i in length:
            mean = np.append(mean, np.mean(q[index: index + i], axis=0))
            index += i
        return mean

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0)

            hidden = tf.layers.dense(s, 40, activation=tf.nn.tanh, kernel_initializer=init_w,
                                     bias_initializer=init_b, name='l1', trainable=trainable, reuse=tf.AUTO_REUSE)
            # a = tf.layers.dense(net, self.a_dim, name='a', trainable=trainable)
            hidden = tf.layers.dense(hidden, 20, activation=tf.nn.tanh, kernel_initializer=init_w,
                                     bias_initializer=init_b, name='a1', trainable=trainable, reuse=tf.AUTO_REUSE)
            out = tf.layers.dense(hidden, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                  bias_initializer=init_b, name='a2', trainable=trainable, reuse=tf.AUTO_REUSE)
            return tf.multiply(out, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            n_l1 = 40
            n_l2 = 20
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0)
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable, initializer=init_w)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable, initializer=init_w)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable, initializer=init_b)
            #  net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            hidden = tf.nn.tanh(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            hidden = tf.layers.dense(hidden, n_l2, activation=tf.nn.tanh, kernel_initializer=init_w,
                                     bias_initializer=init_b, trainable=trainable, name='net2', reuse=tf.AUTO_REUSE)
            out = tf.layers.dense(hidden, 1, trainable=trainable, kernel_initializer=init_w,
                                  bias_initializer=init_b, name='net3', reuse=tf.AUTO_REUSE)  # Q(s,a)
            return out

    def get_targetq(self, s, a):
        q = []
        for i in range(len(s)):
            q.append(tf.reduce_mean(self._build_c(s[i], a[i], 'target', 'False')))
        return q


def lcm(a):
    lc = 1
    for i in a:
        lc = int(i * lc / math.gcd(i, lc))
    return lc
