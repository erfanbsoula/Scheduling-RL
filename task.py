#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/9

import numpy as np
from config import *

np.random.seed(199686)


class Task:
    def __init__(self):
        self.interval = np.random.poisson(LAMBDA, FREQUENCY)
        self.arrive_time = []
        for i in range(FREQUENCY):
            interval_from_beginning = 0
            for j in range(i + 1):
                interval_from_beginning += self.interval[j]
            self.arrive_time.append(interval_from_beginning)
        self.deadline = np.random.randint(MIN_DEADLINE, MAX_DEADLINE, 1)[0]
        self.execute_time = round(np.random.exponential((MAX_DEADLINE + MIN_DEADLINE) / 2 * GRANULARITY))
        self.execute_time = np.clip(self.execute_time, MIN_EXECUTE_TIME, self.deadline)
        self.count = 0

    def create_instance(self):
        self.count += 1
        return Instance(self)


class Instance:
    def __init__(self, task):
        self.execute_time = task.execute_time
        self.deadline = task.deadline
        self.laxity_time = self.deadline - self.execute_time
        self.over = False

    def step(self, execute):
        if execute:
            self.execute_time -= 1
        self.deadline -= 1
        self.laxity_time = self.deadline - self.execute_time
        if self.execute_time == 0:
            return "finish"
        if self.deadline == 0:
        # if self.deadline - self.execute_time < 0:
            return "miss"
        return

