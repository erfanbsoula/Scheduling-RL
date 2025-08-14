from typing import List
import numpy as np
from config import (
    PROCESSOR_COUNT,
    TASK_PER_PROCESSOR,
    INSTANCES_PER_TASK,
    MIN_LOAD, MAX_LOAD,
    MIN_PERIOD, MAX_PERIOD,
    STATIC_POWER_COEFF,
    DYNAMIC_POWER_COEFF,
    ENERGY_PENALTY_COEFF,
)
from task_gen import StaffordRandFixedSum, gen_periods

np.random.seed(199686)


class Task:

    def __init__(self, task_target_utilization: float, task_period: int):

        self.deadline = task_period
        self.work_units = max(1, int(task_period * task_target_utilization))
        self.mean_arrival_interval = task_period

        self.arrival_intervals = np.random.exponential(
            self.mean_arrival_interval, INSTANCES_PER_TASK
        )
        self.arrival_intervals = self.arrival_intervals.round().astype(int)
        self.arrival_intervals = np.maximum(self.arrival_intervals, 1)
        self.arrival_times = np.cumsum(self.arrival_intervals)

        self.instance_count = 0


    def create_instance(self) -> "Instance":
        instance = Instance(self.deadline, self.work_units)
        self.instance_count += 1
        return instance


class Instance:

    def __init__(self, deadline, total_work_units):
        self.initial_work_units = total_work_units
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

        self.state_dim = 16
        self.stats = {
            "system_load": 0,
            "normalized_instance_count": 0,
            "arrived_instance_ratio": 0,
            "remaining_work_units": {"min": 0, "mean": 0, "max": 0},
            "deadline": {"min": 0, "mean": 0, "max": 0},
            "laxity": {"min": 0, "mean": 0, "max": 0}
        }


    def reset(self):

        self.time = 0
        self.processor_count = PROCESSOR_COUNT
        self.task_count = self.processor_count * TASK_PER_PROCESSOR
        target_system_utilization_per_processor = np.random.uniform(MIN_LOAD, MAX_LOAD)
        target_system_utilization = target_system_utilization_per_processor * self.processor_count

        utilizations = StaffordRandFixedSum(self.task_count, target_system_utilization, 1).flatten()
        periods = gen_periods(self.task_count, 1, MIN_PERIOD, MAX_PERIOD, 1.0, "logunif").flatten()
        periods = periods.round().astype(int) # periods are integers for now

        self.task_set = []
        for utilization, period in zip(utilizations, periods):
            self.task_set.append(Task(utilization, period))

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
            for key, value in self.stats.items():
                if isinstance(value, dict):
                    for stat in value:
                        self.stats[key][stat] = 0
                else:
                    self.stats[key] = 0
            return

        total_load = sum(i.remaining_work_units / i.deadline for i in self.active_instances)
        self.stats["system_load"] = total_load / self.processor_count
        self.stats["normalized_instance_count"] = len(self.active_instances) / self.task_count

        instance_deadlines = [i.deadline for i in self.active_instances]
        instance_remaining_execution_times = [i.remaining_work_units for i in self.active_instances]
        instance_laxities = [i.laxity for i in self.active_instances]

        self.stats["remaining_work_units"]["min"] = np.min(instance_remaining_execution_times)
        self.stats["remaining_work_units"]["mean"] = np.mean(instance_remaining_execution_times)
        self.stats["remaining_work_units"]["max"] = np.max(instance_remaining_execution_times)

        self.stats["deadline"]["min"] = np.min(instance_deadlines)
        self.stats["deadline"]["mean"] = np.mean(instance_deadlines)
        self.stats["deadline"]["max"] = np.max(instance_deadlines)

        self.stats["laxity"]["min"] = np.min(instance_laxities)
        self.stats["laxity"]["mean"] = np.mean(instance_laxities)
        self.stats["laxity"]["max"] = np.max(instance_laxities)

        total_instances = self.task_count * INSTANCES_PER_TASK
        arrived_instances = sum(task.instance_count for task in self.task_set)
        self.stats["arrived_instance_ratio"] = arrived_instances / total_instances


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
        normalized_energy_penalty = energy_penalty / (self.stats["system_load"] + 1e-6)
        global_reward = completed_count - missed_count - normalized_energy_penalty

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

        if len(self.active_instances) == 0:
            return np.array([])

        global_state = [
            self.stats["system_load"],
            self.stats["normalized_instance_count"],
            self.stats["arrived_instance_ratio"],
            self.stats["remaining_work_units"]["min"] / MAX_PERIOD,
            self.stats["remaining_work_units"]["mean"] / MAX_PERIOD,
            self.stats["remaining_work_units"]["max"] / MAX_PERIOD,
            self.stats["deadline"]["min"] / MAX_PERIOD,
            self.stats["deadline"]["mean"] / MAX_PERIOD,
            self.stats["deadline"]["max"] / MAX_PERIOD,
            self.stats["laxity"]["min"] / MAX_PERIOD,
            self.stats["laxity"]["mean"] / MAX_PERIOD,
            self.stats["laxity"]["max"] / MAX_PERIOD,
        ]
        global_state = np.array(global_state, dtype=np.float32)

        mean_remaining_work_units = self.stats["remaining_work_units"]["mean"] + 1e-6
        mean_deadline = self.stats["deadline"]["mean"] + 1e-6
        mean_laxity = self.stats["laxity"]["mean"] + 1e-6

        local_observations = []
        for instance in self.active_instances:
            local_obs = [
                (instance.deadline - self.stats["deadline"]["min"]) / mean_deadline,
                (instance.laxity - self.stats["laxity"]["min"]) / mean_laxity,
                instance.remaining_work_units / mean_remaining_work_units,
                instance.remaining_work_units / instance.initial_work_units,
            ]
            local_observations.append(local_obs)

        local_observations = np.array(local_observations, dtype=np.float32)
        global_observations = np.tile(global_state, (len(self.active_instances), 1))
        state = np.hstack((local_observations, global_observations))

        return state


    def done(self):
        if len(self.active_instances) > 0:
            return False

        for task in self.task_set:
            if task.instance_count < INSTANCES_PER_TASK:
                return False
        return True


    def calc_mean_utilization(self):
        utils = []
        for task in self.task_set:
            t = np.mean(task.arrival_intervals)
            utils.append(task.work_units / t)

        return np.mean(utils) * TASK_PER_PROCESSOR