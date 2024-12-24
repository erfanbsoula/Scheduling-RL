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
