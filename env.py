#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/9

from task import *
import copy as cp


class Env(object):
    def __init__(self):
        self.time = 0
        self.task_set = []
        self.instance = []
        self.no_processor = 0
        self.no_task = 0
        self.mean_deadline = 0
        self.mean_execute = 0
        self.mean_laxity = 0
        self.min_deadline = 0
        self.min_execute = 0
        self.min_laxity = 0
        self.max_deadline = 0
        self.max_execute = 0
        self.max_laxity = 0
        self.saved = 0
        self.count = 0

    def reset(self):
        self.time = 0
        del self.task_set
        self.no_processor = np.random.randint(NO_PROCESSOR - 4, NO_PROCESSOR + 4)
        self.no_task = self.no_processor * TASK_PER_PROCESSOR
        self.task_set = []
        self.instance = []
        for i in range(self.no_task):
            task = Task()
            self.task_set.append(task)
        self.saved = 0
        self.arrive()
        self.update()

    def step(self, actions):
        global_reward = 0
        info = np.zeros(2)
        next_state = []
        # 对优先级排序，选前m个执行
        executable = np.argsort(actions)[-self.no_processor:]
        #if len(actions) > self.no_processor:
        #    self.instance[executable[0]].execute_time += 1
        for i in range(len(self.instance)):
            instance = self.instance[i]
            result = instance.step(True if i in executable and actions[i] > 0.1 else False)
            if result == "miss":
                global_reward -= 1
                instance.over = True
                info[1] += 1
            elif result == "finish":
                info[0] += 1
                global_reward += 1
                instance.over = True
        self.time += 1
        self.arrive()
        self.update()
        for i in range(len(self.instance)):
            next_state.append(self.observation(self.instance[i]))
        self.del_instance()
        return global_reward, self.done(), next_state, info

    def update(self):
        if len(self.instance) == 0:
            self.min_deadline = 0
            self.min_execute = 0
            self.min_laxity = 0
            self.mean_deadline = 0
            self.mean_execute = 0
            self.mean_laxity = 0
            self.max_deadline = 0
            self.max_execute = 0
            self.max_laxity = 0
            return
        instance_deadline = []
        instance_execute = []
        instance_laxity = []
        for i in self.instance:
            instance_deadline.append(i.deadline)
            instance_execute.append(i.execute_time)
            instance_laxity.append(i.laxity_time)
        self.min_deadline = min(instance_deadline)
        self.min_execute = min(instance_execute)
        self.min_laxity = min(instance_laxity)
        self.max_deadline = max(instance_deadline)
        self.max_execute = max(instance_execute)
        self.max_laxity = max(instance_laxity)
        self.mean_deadline = np.mean(instance_deadline)
        self.mean_execute = np.mean(instance_execute)
        self.mean_laxity = np.mean(instance_laxity)

    def arrive(self):
        for task in self.task_set:
            if task.count < FREQUENCY and self.time == task.arrive_time[task.count]:
                self.instance.append(task.create_instance())
        self.update()

    def done(self):
        if len(self.instance) > 0:
            return 0
        for t in self.task_set:
            if t.count < FREQUENCY:
                return 0
        return 1

    def observation(self, instance):
        return np.array([instance.execute_time - self.mean_execute, instance.deadline - self.mean_deadline,
                         instance.execute_time - self.max_execute, instance.deadline - self.max_deadline,
                         instance.execute_time - self.min_execute, instance.deadline - self.min_deadline,
                         instance.laxity_time - self.mean_laxity, instance.laxity_time - self.min_laxity,
                         instance.laxity_time - self.max_laxity])

    def save(self):
        self.saved = cp.deepcopy(self.task_set)

    def load(self):
        self.time = 0
        self.instance = []
        del self.task_set
        self.task_set = cp.deepcopy(self.saved)
        del self.saved
        self.update()

    def del_instance(self):
        for i in self.instance[::-1]:
            if i.over:
                self.instance.remove(i)
                del i

    def utilization(self):
        c = []
        t = []
        for task in self.task_set:
            c.append(task.execute_time)
            t.append(np.mean(task.interval))
        return (np.mean(c) / np.mean(t)) * (self.no_task / self.no_processor)
