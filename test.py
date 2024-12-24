#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/9
# from ddpg_torch import *
from task import *
import copy
# import ddpg_tf
# import tensorflow as tf
# import ddpg_torch
import math
# import torch
import os
import time
import datetime
from env import *
from timeit import timeit
from timeit import repeat


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))


def relu(x):
    return np.maximum(x, 0)


def leakyrelu(x):
    m = np.maximum(x, 0)
    n = np.minimum(x, 0)
    n = 0.01 * n
    return m + n

def overhead():
    def nn():
        o1 = leakyrelu(np.add(np.matmul(temp, w1), b1))
        o2 = leakyrelu(np.add(np.matmul(o1, w2), b2))
        o3 = sigmoid(np.add(np.matmul(o2, w3), b3))
    w1 = 2 * np.random.randn(8, 16)
    b1 = 2 * np.random.randn(1, 16)
    w2 = 2 * np.random.randn(16, 8)
    b2 = 2 * np.random.randn(1, 8)
    w3 = 2 * np.random.randn(8, 1)
    b3 = 2 * np.random.randn(1, 1)

    for size in (20, 50, 100, 200, 300, 500):
        te = []
        b1 = np.tile(b1[0], (size, 1))
        b2 = np.tile(b2[0], (size, 1))
        b3 = np.tile(b3[0], (size, 1))
        for j in range(10000):
            temp = 2 * np.random.randn(size, 8)
            t = repeat('nn()', 'from __main__ import nn', number=1, repeat=100)
            te.append(min(t))
        print(str(size) + ':', '%.5f' % (min(te) * 1000), '%.5f' % (np.mean(te) * 1000), '%.5f' % (max(te) * 1000))
