from typing import List
import copy as cp
import numpy as np
from config import *

np.random.seed(199686)


class Task:

    def __init__(self):
        # !! Attention !!
        # the paper states exponentially distributed arrivals
        # but here they have used poisson distribution!
        self.interval = np.random.poisson(LAMBDA, FREQUENCY)
        self.arrive_time = np.cumsum(self.interval)
        self.deadline = np.random.randint(MIN_DEADLINE, MAX_DEADLINE)
        self.execute_time = round(
            np.random.exponential((MAX_DEADLINE + MIN_DEADLINE) / 2 * GRANULARITY)
        )
        self.execute_time = np.clip(self.execute_time, MIN_EXECUTE_TIME, self.deadline)
        self.count = 0


    def create_instance(self) -> "Instance":
        self.count += 1
        return Instance(self)


class Instance:

    def __init__(self, task: Task):
        self.execute_time = task.execute_time
        self.deadline = task.deadline
        self.laxity_time = self.deadline - self.execute_time
        self.over = False


    def step(self, execute: bool):
        self.deadline -= 1
        self.laxity_time = self.deadline - self.execute_time

        if execute:
            self.execute_time -= 1

        if self.execute_time == 0:
            return "finished"

        if self.deadline == 0:
            return "missed"

        return "ready"


class Env(object):

    def __init__(self):
        self.time = 0
        self.no_processor = 0
        self.no_task = 0
        self.task_set: List[Task] = []
        self.instance: List[Instance] = []
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
        self.saved = 0
        del self.task_set
        self.instance = []
        self.no_processor = np.random.randint(NO_PROCESSOR - 4, NO_PROCESSOR + 4)
        self.no_task = self.no_processor * TASK_PER_PROCESSOR
        self.task_set = [Task() for i in range(self.no_task)]
        self.arrive()


    def step(self, actions):

        global_reward = 0
        info = np.zeros(2)
        next_state = []

        executable = np.argsort(actions)[-self.no_processor:]

        for i in range(len(self.instance)):
            instance = self.instance[i]
            result = instance.step(True if i in executable and actions[i] > 0.1 else False)

            if result == "missed":
                instance.over = True
                global_reward -= 1
                info[1] += 1

            elif result == "finished":
                info[0] += 1
                global_reward += 1
                instance.over = True

        self.time += 1
        self.del_instance()
        self.arrive()
        for i in range(len(self.instance)):
            next_state.append(self.observation(self.instance[i]))

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

        self.min_deadline = np.min(instance_deadline)
        self.min_execute = np.min(instance_execute)
        self.min_laxity = np.min(instance_laxity)
        self.max_deadline = np.max(instance_deadline)
        self.max_execute = np.max(instance_execute)
        self.max_laxity = np.max(instance_laxity)
        self.mean_deadline = np.mean(instance_deadline)
        self.mean_execute = np.mean(instance_execute)
        self.mean_laxity = np.mean(instance_laxity)


    # later on merge this function to update
    def arrive(self):
        for task in self.task_set:
            if task.count < FREQUENCY and self.time == task.arrive_time[task.count]:
                self.instance.append(task.create_instance())

        self.update()


    def done(self):
        if len(self.instance) > 0:
            return False

        for t in self.task_set:
            if t.count < FREQUENCY:
                return False

        return True


    def observation(self, instance):
        return np.array([
            instance.execute_time - self.mean_execute,
            instance.execute_time - self.max_execute,
            instance.execute_time - self.min_execute,
            instance.deadline - self.mean_deadline,
            instance.deadline - self.max_deadline,
            instance.deadline - self.min_deadline,
            instance.laxity_time - self.mean_laxity,
            instance.laxity_time - self.min_laxity,
            instance.laxity_time - self.max_laxity
        ])


    # save and load need attention and maybe they can be removed
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
