from typing import List
import numpy as np
from config import *

np.random.seed(199686)


class Task:

    def __init__(self):
        # !! Attention !!
        # the paper states exponentially distributed arrivals
        # but here they have used poisson distribution!
        self.interval = np.random.poisson(LAMBDA, INSTANCES_PER_TASK)
        self.arrive_time = np.cumsum(self.interval)
        self.deadline = np.random.randint(MIN_DEADLINE, MAX_DEADLINE)
        self.execute_time = round(
            np.random.exponential((MAX_DEADLINE + MIN_DEADLINE) / 2 * GRANULARITY)
        )
        self.execute_time = np.clip(self.execute_time, MIN_EXECUTE_TIME, self.deadline)
        self.instance_count = 0


    def create_instance(self) -> "Instance":
        self.instance_count += 1
        return Instance(self)


class Instance:

    def __init__(self, task: Task):
        self.execute_time = task.execute_time
        self.deadline = task.deadline
        self.laxity_time = self.deadline - self.execute_time
        self.over = False


    def step(self, execute: bool):
        self.deadline -= 1
        if execute:
            self.execute_time -= 1

        self.laxity_time = self.deadline - self.execute_time

        if self.execute_time == 0 or self.deadline == 0:
            self.over = True


    def is_finished(self):
        return self.execute_time == 0


    def is_missed(self):
        return self.execute_time != 0 and self.deadline == 0


class Environment(object):

    def __init__(self):

        self.time = 0
        self.processor_count = 0
        self.task_count = 0
        self.task_set: List[Task] = []
        self.active_instances: List[Instance] = []

        self.mean_deadline = 0
        self.mean_execute = 0
        self.mean_laxity = 0
        self.min_deadline = 0
        self.min_execute = 0
        self.min_laxity = 0
        self.max_deadline = 0
        self.max_execute = 0
        self.max_laxity = 0



    def reset(self):

        self.time = 0
        self.processor_count = np.random.randint(PROCESSOR_COUNT - 4, PROCESSOR_COUNT + 4)
        self.task_count = self.processor_count * TASK_PER_PROCESSOR
        self.task_set = [Task() for i in range(self.task_count)]
        self.active_instances = []
        
        self.arrive_instances()
    

    def arrive_instances(self):
        for task in self.task_set:
            if task.instance_count < INSTANCES_PER_TASK and self.time == task.arrive_time[task.instance_count]:
                self.active_instances.append(task.create_instance())

        self.update_env_stats()


    def update_env_stats(self):

        if len(self.active_instances) == 0:
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
        for i in self.active_instances:
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
            instance.execute_time - self.mean_execute,
            instance.execute_time - self.max_execute,
            instance.execute_time - self.min_execute,
            instance.deadline - self.mean_deadline,
            instance.deadline - self.max_deadline,
            instance.deadline - self.min_deadline,
            instance.laxity_time - self.mean_laxity,
            instance.laxity_time - self.min_laxity,
            instance.laxity_time - self.max_laxity
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
            c.append(task.execute_time)
            t.append(np.mean(task.interval))

        return (np.mean(c) / np.mean(t)) * (self.task_count / self.processor_count)
