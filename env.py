from typing import List
import numpy as np
from config import *

np.random.seed(199686)


class Task:


    def __init__(self, load):
        self.deadline = np.random.randint(MIN_DEADLINE, MAX_DEADLINE+1)

        self.granularity = np.random.uniform(MIN_GRANULARITY, MAX_GRANULARITY)
        self.execution_rate = self.deadline * self.granularity
        self.execution_times = np.random.exponential(
            self.execution_rate, INSTANCES_PER_TASK
        )
        self.execution_times = self.execution_times.round().astype(int)
        self.execution_times = np.clip(self.execution_times, 1, self.deadline)

        # self.arrival_rate = np.random.randint(12, 26)
        # (min_deadline + max_deadline) / 2 * granularity / arrival_rate * 5 = load
        # arrival_rate = (min_deadline + max_deadline) / 2 * granularity / load * 5
        self.expected_execution_time = (MIN_DEADLINE + MAX_DEADLINE) / 2 * self.granularity
        self.arrival_rate = self.expected_execution_time / load * TASK_PER_PROCESSOR
        self.arrival_intervals = np.random.exponential(
            self.arrival_rate, INSTANCES_PER_TASK
        )
        self.arrival_intervals = self.arrival_intervals.round().astype(int)
        self.arrival_times = np.cumsum(self.arrival_intervals)

        self.instance_count = 0


    def create_instance(self) -> "Instance":
        execution_time = self.execution_times[self.instance_count]
        instance = Instance(self.deadline, execution_time)
        self.instance_count += 1
        return instance


class Instance:

    def __init__(self, deadline, execution_time):
        self.execution_time = execution_time
        self.deadline = deadline
        self.laxity = self.deadline - self.execution_time
        self.over = False


    def step(self, execute: bool):
        self.deadline -= 1
        if execute:
            self.execution_time -= 1

        self.laxity = self.deadline - self.execution_time

        if self.execution_time == 0 or self.deadline == 0:
            self.over = True


    def is_finished(self):
        return self.execution_time == 0


    def is_missed(self):
        return self.execution_time != 0 and self.deadline == 0


class Environment(object):

    def __init__(self):

        self.time = 0
        self.processor_count = 0
        self.task_count = 0
        self.task_set: List[Task] = []
        self.active_instances: List[Instance] = []

        self.min_execution_time = 0
        self.mean_execution_time = 0
        self.max_execution_time = 0

        self.min_deadline = 0
        self.mean_deadline = 0
        self.max_deadline = 0

        self.min_laxity = 0
        self.mean_laxity = 0
        self.max_laxity = 0


    def reset(self):

        self.time = 0
        self.processor_count = PROCESSOR_COUNT
        self.task_count = self.processor_count * TASK_PER_PROCESSOR
        load = np.random.uniform(MIN_LOAD, MAX_LOAD)
        self.task_set = [Task(load) for i in range(self.task_count)]
        self.active_instances = []

        self.arrive_instances()


    def arrive_instances(self):
        for task in self.task_set:
            if task.instance_count < INSTANCES_PER_TASK and self.time == task.arrival_times[task.instance_count]:
                self.active_instances.append(task.create_instance())

        self.update_env_stats()


    def update_env_stats(self):

        if len(self.active_instances) == 0:
            self.min_execution_time = 0
            self.mean_execution_time = 0
            self.max_execution_time = 0
            self.min_deadline = 0
            self.mean_deadline = 0
            self.max_deadline = 0
            self.min_laxity = 0
            self.mean_laxity = 0
            self.max_laxity = 0
            return

        instance_deadlines = []
        instance_execution_times = []
        instance_laxities = []
        for i in self.active_instances:
            instance_execution_times.append(i.execution_time)
            instance_deadlines.append(i.deadline)
            instance_laxities.append(i.laxity)

        self.min_execution_time = np.min(instance_execution_times)
        self.mean_execution_time = np.mean(instance_execution_times)
        self.max_execution_time = np.max(instance_execution_times)

        self.min_deadline = np.min(instance_deadlines)
        self.mean_deadline = np.mean(instance_deadlines)
        self.max_deadline = np.max(instance_deadlines)

        self.min_laxity = np.min(instance_laxities)
        self.mean_laxity = np.mean(instance_laxities)
        self.max_laxity = np.max(instance_laxities)


    def step(self, actions: np.ndarray[np.float64]):

        completed_count = 0
        missed_count = 0

        executable = np.argsort(actions)[-self.processor_count:]

        for i in range(len(self.active_instances)):

            instance = self.active_instances[i]
            instance.step(True if i in executable and actions[i] > 0.1 else False)

            if instance.is_missed():
                missed_count += 1

            elif instance.is_finished():
                completed_count += 1

        global_reward = completed_count - missed_count

        self.time += 1
        self.delete_inactive_instances()
        self.arrive_instances()
        next_state = self.get_state()

        return global_reward, next_state, self.done(), completed_count, missed_count


    def delete_inactive_instances(self):
        for instance in self.active_instances[:]:
            if instance.over:
                self.active_instances.remove(instance)
                del instance


    def get_state(self):

        state = []
        for instance in self.active_instances:
            state.append(self.observation(instance))

        return np.array(state, dtype=np.float32)


    def observation(self, instance):
        return [
            instance.execution_time - self.min_execution_time,
            instance.execution_time - self.mean_execution_time,
            instance.execution_time - self.max_execution_time,
            instance.deadline - self.min_deadline,
            instance.deadline - self.mean_deadline,
            instance.deadline - self.max_deadline,
            instance.laxity - self.min_laxity,
            instance.laxity - self.mean_laxity,
            instance.laxity - self.max_laxity
        ]


    def done(self):

        if len(self.active_instances) > 0:
            return False

        for task in self.task_set:
            if task.instance_count < INSTANCES_PER_TASK:
                return False

        return True


    def calc_utilization(self):

        c, t = [], []
        for task in self.task_set:
            c.append(np.mean(task.execution_times))
            t.append(np.mean(task.arrival_intervals))

        return (np.mean(c) / np.mean(t)) * TASK_PER_PROCESSOR
