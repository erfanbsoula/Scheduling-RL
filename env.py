from typing import List, Any
from enum import Enum
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
import heapq


class EventType(Enum):

    ARRIVAL = 1
    DEADLINE = 2


class Event:
    """Represents an event in the simulation's priority queue."""

    def __init__(self, timestamp: float, event_type: EventType, data: Any = None):
        """
        Args:
            timestamp (float): The time at which the event occurs.
            event_type (EventType): The type of the event (e.g., EventType.ARRIVAL).
            data (Any, optional): Additional data associated with the event.
        """
        self.timestamp = timestamp
        self.event_type = event_type
        self.data = data


    def __lt__(self, other: "Event") -> bool:
        return self.timestamp < other.timestamp


class EventQueue:
    """Represents a priority queue for managing simulation events."""

    def __init__(self):
        self.queue: List[Event] = []

    def push_event(self, event: Event):
        heapq.heappush(self.queue, event)

    def peek_next_timestamp(self) -> float:
        while (
            len(self.queue) != 0 and
            self.queue[0].data.status in [InstanceStatus.COMPLETED, InstanceStatus.MISSED]
        ):
            heapq.heappop(self.queue)

        return self.queue[0].timestamp if self.queue else float('inf')


    def pop_events(self) -> List[Event]:
        """
        Pops all events that occur at the earliest timestamp.

        Returns:
            List[Event]: A list of events with the same earliest timestamp.
        """
        if len(self.queue) == 0:
            raise Exception("Trying to pop events from an empty event queue")

        earliest_timestamp = self.peek_next_timestamp()
        if earliest_timestamp == float('inf'):
            return []

        events_to_process = []
        while self.queue and self.queue[0].timestamp == earliest_timestamp:
            if self.queue[0].data.status not in [InstanceStatus.COMPLETED, InstanceStatus.MISSED]:
                 events_to_process.append(heapq.heappop(self.queue))
            else:
                heapq.heappop(self.queue)

        return events_to_process


    def reset(self):
        self.queue.clear()

    def is_empty(self) -> bool:
        self.peek_next_timestamp()
        return len(self.queue) == 0


class InstanceStatus(Enum):

    PENDING = 0
    ACTIVE = 1
    COMPLETED = 2
    MISSED = 3


class Instance:

    def __init__(self, arrival_time: float, deadline: float, total_work_units: float):
        self.arrival_time = arrival_time
        self.deadline = deadline
        self.initial_work_units = total_work_units
        self.remaining_work_units = total_work_units
        self.status = InstanceStatus.PENDING


    def execute(self, duration: float, frequency_scale: float = 1.0):
        if self.status != InstanceStatus.ACTIVE:
            raise Exception("Cannot execute an instance that is not active")

        work_done = duration * frequency_scale
        self.remaining_work_units -= work_done


    def relative_deadline(self, current_time: float) -> float:
        if self.status != InstanceStatus.ACTIVE:
            raise Exception("Trying to get relative deadline of a non-active instance")

        return self.deadline - current_time


    def laxity(self, current_time: float) -> float:
        if self.status != InstanceStatus.ACTIVE:
            raise Exception("Trying to get laxity of a non-active instance")

        return self.deadline - current_time - self.remaining_work_units


    def update_status(self, current_time: float):

        if self.status == InstanceStatus.PENDING:
            if self.arrival_time <= current_time:
                self.status = InstanceStatus.ACTIVE

        elif self.status == InstanceStatus.ACTIVE:
            if self.remaining_work_units <= 1e-6:
                self.status = InstanceStatus.COMPLETED
            elif self.deadline <= current_time:
                self.status = InstanceStatus.MISSED


class Task:

    def __init__(self, task_target_utilization: float, task_period: float):

        self.relative_deadline = task_period
        self.work_units = task_period * task_target_utilization
        self.mean_arrival_interval = task_period
        self.arrival_intervals = np.random.exponential(
            self.mean_arrival_interval, INSTANCES_PER_TASK
        )
        self.arrival_intervals = np.clip(self.arrival_intervals, 0.1, None)
        self.arrival_times = np.cumsum(self.arrival_intervals)

        self.instances: List[Instance] = [
            Instance(arrival_time, arrival_time + self.relative_deadline, self.work_units)
            for arrival_time in self.arrival_times
        ]


class Environment(object):

    def __init__(self):

        self.time = 0.0
        self.processor_count = PROCESSOR_COUNT
        self.task_count = 0
        self.task_set: List[Task] = []
        self.event_queue = EventQueue()
        self.instance_arrival_count = 0
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


    def reset(self, per_core_utilization: float = None):

        self.time = 0.0
        self.task_count = self.processor_count * TASK_PER_PROCESSOR

        if per_core_utilization is None:
            per_core_utilization = np.random.uniform(MIN_LOAD, MAX_LOAD)

        target_util = per_core_utilization * self.processor_count
        utilizations = StaffordRandFixedSum(self.task_count, target_util, 1).flatten()
        periods = gen_periods(self.task_count, 1, MIN_PERIOD, MAX_PERIOD, 0.1, "logunif").flatten()

        self.task_set.clear()
        for task_util, task_period in zip(utilizations, periods):
            self.task_set.append(Task(task_util, task_period))

        self.event_queue.reset()
        for task in self.task_set:
            for instance in task.instances:
                self.event_queue.push_event(Event(instance.arrival_time, EventType.ARRIVAL, instance))
                self.event_queue.push_event(Event(instance.deadline, EventType.DEADLINE, instance))

        self.instance_arrival_count = 0
        self.active_instances = []
        self.total_energy_consumed = 0.0

        self.time = self.event_queue.peek_next_timestamp()
        self.process_events_at_current_time()
        self.update_env_stats()


    def process_events_at_current_time(self):

        events = self.event_queue.pop_events()
        for event in events:
            instance = event.data
            instance.update_status(self.time)

            if event.event_type == EventType.ARRIVAL:
                self.active_instances.append(instance)
                self.instance_arrival_count += 1


    def step(self, scheduling_priorities: np.ndarray, frequency_scales: np.ndarray):
        """
        Executes one event-driven step in the simulation.
        Time advances to the next significant event.

        Args:
            scheduling_priorities (np.ndarray): Array of size num_active_instances,
                where each element contains the scheduling priority for an instance.
            frequency_scales (np.ndarray): Array of size num_active_instances,
                where each element contains the frequency scale for execution of an instance.

        Returns:
            transition (tuple):
                global_reward (int): Total reward received this step.
                next_state (np.ndarray): The next state (shape: [num_active_instances, state_dim]).
                done (bool): Whether the episode is finished (no more new instances).
                completed_count (int): Number of instances completed this step.
                missed_count (int): Number of instances missed this step.
        """
        if not self.active_instances:
            self.time = self.event_queue.peek_next_timestamp()
            self.process_events_at_current_time()
            self.update_env_stats()
            return 0.0, self.get_state(), self.done(), 0, 0

        indices_by_priority = np.argsort(scheduling_priorities)[::-1]
        exec_indices = indices_by_priority[:min(self.processor_count, len(self.active_instances))]

        next_event_in_queue = self.event_queue.peek_next_timestamp()

        earliest_completion_time = float('inf')
        for i in exec_indices:
            instance = self.active_instances[i]
            freq = frequency_scales[i]
            time_to_complete = instance.remaining_work_units / freq
            earliest_completion_time = min(earliest_completion_time, self.time + time_to_complete)

        next_timestamp = min(next_event_in_queue, earliest_completion_time)
        duration = next_timestamp - self.time

        step_energy = 0.0
        for i in exec_indices:
            instance = self.active_instances[i]
            freq = frequency_scales[i]
            instance.execute(duration, freq)
            
            static_power = STATIC_POWER_COEFF * freq
            dynamic_power = DYNAMIC_POWER_COEFF * (freq ** 3)
            step_energy += (static_power + dynamic_power) * duration

        self.total_energy_consumed += step_energy

        # Advance simulation time
        self.time = next_timestamp

        completed_count, missed_count = 0, 0
        for instance in self.active_instances:
            instance.update_status(self.time)
            if instance.status == InstanceStatus.COMPLETED:
                completed_count += 1
            elif instance.status == InstanceStatus.MISSED:
                missed_count += 1

        if (
            self.time == next_event_in_queue and
            next_event_in_queue == self.event_queue.peek_next_timestamp()
        ):
             self.process_events_at_current_time()

        self.active_instances = [
            instance for instance in self.active_instances
            if instance.status == InstanceStatus.ACTIVE
        ]

        # Calculate reward
        energy_penalty = ENERGY_PENALTY_COEFF * step_energy
        norm_energy_penalty = energy_penalty# / (self.stats.get("system_load", 0) + 1e-6)
        global_reward = completed_count - missed_count - norm_energy_penalty

        self.update_env_stats()
        return global_reward, self.get_state(), self.done(), completed_count, missed_count


    def update_env_stats(self):

        if len(self.active_instances) == 0:
            for key, value in self.stats.items():
                if isinstance(value, dict):
                    for stat in value:
                        self.stats[key][stat] = 0
                else:
                    self.stats[key] = 0
            return

        total_load = sum(
            i.remaining_work_units / i.relative_deadline(self.time) for i in self.active_instances
        )
        self.stats["system_load"] = total_load / self.processor_count
        self.stats["normalized_instance_count"] = len(self.active_instances) / self.task_count

        instance_deadlines = [i.relative_deadline(self.time) for i in self.active_instances]
        instance_remaining_execution_times = [i.remaining_work_units for i in self.active_instances]
        instance_laxities = [i.laxity(self.time) for i in self.active_instances]

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
        self.stats["arrived_instance_ratio"] = self.instance_arrival_count / total_instances


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
                (instance.relative_deadline(self.time) - self.stats["deadline"]["min"]) / mean_deadline,
                (instance.laxity(self.time) - self.stats["laxity"]["min"]) / mean_laxity,
                instance.remaining_work_units / mean_remaining_work_units,
                instance.remaining_work_units / instance.initial_work_units,
            ]
            local_observations.append(local_obs)

        local_observations = np.array(local_observations, dtype=np.float32)
        global_observations = np.tile(global_state, (len(self.active_instances), 1))
        state = np.hstack((local_observations, global_observations))

        return state


    def done(self) -> bool:
        return self.event_queue.is_empty()


    def calc_mean_utilization(self) -> float:
        utils = [task.work_units / np.mean(task.arrival_intervals) for task in self.task_set]
        return np.sum(utils) / self.processor_count