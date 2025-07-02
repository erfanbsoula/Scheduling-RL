from typing import List
import numpy as np
from config import (
    PROCESSOR_COUNT,
    TASK_PER_PROCESSOR,
    INSTANCES_PER_TASK,
    MIN_LOAD,
    MAX_LOAD,
    STATIC_POWER_COEFF,
    DYNAMIC_POWER_COEFF,
    ENERGY_PENALTY_COEFF,
    MIN_TASK_AVG_LOG_EXEC_TIME, 
    MAX_TASK_AVG_LOG_EXEC_TIME, 
    MIN_TASK_SIGMA_LOG_EXEC_TIME,
    MAX_TASK_SIGMA_LOG_EXEC_TIME,
    MIN_DEADLINE_FACTOR,
    MAX_DEADLINE_FACTOR
)

np.random.seed(199686)


class Task:

    def __init__(self, task_target_utilization: float):

        task_mean_log_exec_time = np.random.uniform(
            MIN_TASK_AVG_LOG_EXEC_TIME, MAX_TASK_AVG_LOG_EXEC_TIME
        )

        task_sigma_log_exec_time = np.random.uniform(
            MIN_TASK_SIGMA_LOG_EXEC_TIME, MAX_TASK_SIGMA_LOG_EXEC_TIME
        )

        # m = (MAX_TASK_SIGMA_LOG_EXEC_TIME - MIN_TASK_SIGMA_LOG_EXEC_TIME) / \
        #     (MAX_TASK_AVG_LOG_EXEC_TIME - MIN_TASK_AVG_LOG_EXEC_TIME)
        
        # task_sigma_log_exec_time = \
        #     m * (task_mean_log_exec_time - MIN_TASK_AVG_LOG_EXEC_TIME) + \
        #     MIN_TASK_SIGMA_LOG_EXEC_TIME

        self.work_units_per_instance = np.random.lognormal(
            task_mean_log_exec_time, task_sigma_log_exec_time, INSTANCES_PER_TASK*10
        )
        self.work_units_per_instance = np.maximum(self.work_units_per_instance, 1.0)
        self.mean_work_units = np.mean(self.work_units_per_instance)

        deadline_factor = np.random.uniform(MIN_DEADLINE_FACTOR, MAX_DEADLINE_FACTOR)
        self.deadline = np.ceil(self.mean_work_units * deadline_factor)
        self.work_units_per_instance = np.random.choice(
            self.work_units_per_instance[self.work_units_per_instance < self.deadline], INSTANCES_PER_TASK
        )
        self.mean_work_units = np.mean(self.work_units_per_instance)

        self.mean_arrival_interval = self.mean_work_units / task_target_utilization
        self.arrival_intervals = np.random.exponential(
            self.mean_arrival_interval, INSTANCES_PER_TASK
        )
        self.arrival_intervals = self.arrival_intervals.round().astype(int)
        self.arrival_intervals = np.maximum(self.arrival_intervals, 1)
        self.arrival_times = np.cumsum(self.arrival_intervals)

        self.instance_count = 0


    def create_instance(self) -> "Instance":
        work_units = self.work_units_per_instance[self.instance_count]
        instance = Instance(self.deadline, work_units)
        self.instance_count += 1
        return instance


class Instance:

    def __init__(self, deadline, total_work_units):
        self.remaining_work_units = total_work_units
        self.deadline = deadline
        self.over = False

    def step(self, execute: bool, frequency_scale: float = 1.0):
        self.deadline -= 1
        if execute:
            work_done_this_step = 1.0 * frequency_scale
            self.remaining_work_units -= work_done_this_step

        if self.remaining_work_units <= 0 or self.deadline <= 0:
            self.over = True

    @property
    def laxity(self):
        return self.deadline - self.remaining_work_units

    def is_finished(self):
        return self.remaining_work_units <= 0

    def is_missed(self):
        return self.remaining_work_units > 0 and self.deadline <= 0


class Environment(object):

    def __init__(self):

        self.time = 0
        self.processor_count = PROCESSOR_COUNT
        self.task_count = 0
        self.task_set: List[Task] = []
        self.active_instances: List[Instance] = []
        self.total_energy_consumed = 0.0

        self.state_dim = 9
        self.stats = {
            "remaining_work_units": {"min": 0, "mean": 0, "max": 0},
            "deadline": {"min": 0, "mean": 0, "max": 0},
            "laxity": {"min": 0, "mean": 0, "max": 0},
        }


    def reset(self):

        self.time = 0
        self.processor_count = PROCESSOR_COUNT
        self.task_count = self.processor_count * TASK_PER_PROCESSOR
        target_system_utilization_per_processor = np.random.uniform(MIN_LOAD, MAX_LOAD)
        avg_target_util_per_task = target_system_utilization_per_processor / TASK_PER_PROCESSOR

        self.task_set = []
        for _ in range(self.task_count):
            variation_factor = np.random.uniform(0.5, 1.5)
            task_specific_target_util = avg_target_util_per_task * variation_factor
            task_specific_target_util = np.maximum(task_specific_target_util, 0.01)
            self.task_set.append(Task(task_specific_target_util))

        self.active_instances = []
        self.total_energy_consumed = 0.0

        self.arrive_instances()
        self.update_env_stats()


    def arrive_instances(self):
        for task in self.task_set:
            if (
                task.instance_count < INSTANCES_PER_TASK and
                self.time >= task.arrival_times[task.instance_count]
            ):
                self.active_instances.append(task.create_instance())


    def update_env_stats(self):

        if len(self.active_instances) == 0:
            for key in self.stats:
                for stat in self.stats[key]:
                    self.stats[key][stat] = 0
            return

        instance_deadlines = []
        instance_remaining_execution_times = []
        instance_laxities = []
        for i in self.active_instances:
            instance_remaining_execution_times.append(i.remaining_work_units)
            instance_deadlines.append(i.deadline)
            instance_laxities.append(i.laxity)

        self.stats["remaining_work_units"]["min"] = np.min(instance_remaining_execution_times)
        self.stats["remaining_work_units"]["mean"] = np.mean(instance_remaining_execution_times)
        self.stats["remaining_work_units"]["max"] = np.max(instance_remaining_execution_times)

        self.stats["deadline"]["min"] = np.min(instance_deadlines)
        self.stats["deadline"]["mean"] = np.mean(instance_deadlines)
        self.stats["deadline"]["max"] = np.max(instance_deadlines)

        self.stats["laxity"]["min"] = np.min(instance_laxities)
        self.stats["laxity"]["mean"] = np.mean(instance_laxities)
        self.stats["laxity"]["max"] = np.max(instance_laxities)


    def step(self, scheduling_priorities: np.ndarray, frequency_scales: np.ndarray):
        """
        Executes one environment step.

        Args:
            scheduling_priorities (np.ndarray): Array of size num_active_instances,
                where each element contains the scheduling priority for an instance.
            frequency_scales (np.ndarray): Array of size num_active_instances,
                where each element contains the frequency scale for execution of an instance.

        Returns:
            transition (tuple):
                global_reward (int): Total reward received this step.
                next_state (np.ndarray): The next state after the step (shape: [num_active_instances, state_dim]).
                done (bool): Whether the episode is finished (no more new instances).
                completed_count (int): Number of instances completed this step.
                missed_count (int): Number of instances missed this step.
        """
        completed_count = 0
        missed_count = 0
        current_step_energy = 0.0

        num_active_instances = len(self.active_instances)

        if num_active_instances == 0:
            self.time += 1
            self.arrive_instances()
            self.update_env_stats()
            next_state = self.get_state()
            return 0, next_state, self.done(), 0, 0

        # Sort descending by priority
        indices_by_execution_priority = np.argsort(scheduling_priorities)[::-1]
        instances_to_execute = []

        for i in range(min(self.processor_count, num_active_instances)):
            instance_idx = indices_by_execution_priority[i]
            instances_to_execute.append(self.active_instances[instance_idx])

        for idx, instance in enumerate(self.active_instances):
            execute_flag = False
            frequency_scale = 1.0

            if instance in instances_to_execute:
                execute_flag = True
                frequency_scale = frequency_scales[idx]

                static_power = STATIC_POWER_COEFF * frequency_scale
                dynamic_power = DYNAMIC_POWER_COEFF * (frequency_scale ** 3)
                power_consumed_by_task = static_power + dynamic_power
                current_step_energy += power_consumed_by_task

            instance.step(execute_flag, frequency_scale)

            if instance.is_missed():
                missed_count += 1

            elif instance.is_finished():
                completed_count +=1


        self.total_energy_consumed += current_step_energy
        energy_penalty = ENERGY_PENALTY_COEFF * current_step_energy
        global_reward = completed_count - missed_count - energy_penalty

        self.time += 1
        self.delete_inactive_instances()
        self.arrive_instances()
        self.update_env_stats()
        next_state = self.get_state()

        return global_reward, next_state, self.done(), completed_count, missed_count


    def delete_inactive_instances(self):
        new_active_instances = []
        for instance in self.active_instances:
            if not instance.over:
                new_active_instances.append(instance)

        self.active_instances = new_active_instances


    def get_state(self):
        state = []
        for instance in self.active_instances:
            state.append(self.observation(instance))

        state = np.array(state, dtype=np.float32)
        state = state.reshape(-1, self.state_dim)
        return state


    def observation(self, instance: Instance):
        obs = [
            instance.remaining_work_units - self.stats["remaining_work_units"]["min"],
            instance.remaining_work_units - self.stats["remaining_work_units"]["mean"],
            instance.remaining_work_units - self.stats["remaining_work_units"]["max"],
            instance.deadline - self.stats["deadline"]["min"],
            instance.deadline - self.stats["deadline"]["mean"],
            instance.deadline - self.stats["deadline"]["max"],
            instance.laxity - self.stats["laxity"]["min"],
            instance.laxity - self.stats["laxity"]["mean"],
            instance.laxity - self.stats["laxity"]["max"]
        ]
        return obs


    def done(self):
        if len(self.active_instances) > 0:
            return False

        for task in self.task_set:
            if task.instance_count < INSTANCES_PER_TASK:
                return False
        return True


    def calc_utilization(self):
        utils = []
        for task in self.task_set:
            w = np.mean(task.work_units_per_instance)
            t = np.mean(task.arrival_intervals)
            utils.append(w / t)

        return np.mean(utils) * TASK_PER_PROCESSOR