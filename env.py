from typing import List, Any
from enum import Enum
import numpy as np
from config import (
    CRITIC_STATE_DIM,
    ACTOR_STATE_DIM,
    PROCESSOR_COUNT,
    TASK_PER_PROCESSOR,
    MIN_LOAD, MAX_LOAD,
    MIN_PERIOD, MAX_PERIOD,
    INSTANCE_COMPLETION_REWARD,
    INSTANCE_MISS_PENALTY,
    STATIC_POWER_COEFF,
    DYNAMIC_POWER_COEFF,
    ENERGY_PENALTY_COEFF,
    MAX_EPISODE_TIME
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

    def __init__(self, arrival_time: float, deadline: float, total_work_units: float,
                 task_index: int):
        self.arrival_time = arrival_time
        self.deadline = deadline
        self.initial_work_units = total_work_units
        self.remaining_work_units = total_work_units
        self.status = InstanceStatus.PENDING
        self.task_index = task_index


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

    def __init__(self, index: int, task_target_utilization: float, task_period: float):

        self.index = index
        self.relative_deadline = task_period
        self.work_units = task_period * task_target_utilization
        self.mean_arrival_interval = task_period


    def create_instance(self, time: float) -> Instance:

        arrival_interval = np.random.exponential(self.mean_arrival_interval)
        arrival_interval = max(arrival_interval, 0.1)
        arrival_time = time + arrival_interval
        deadline = arrival_time + self.relative_deadline
        return Instance(arrival_time, deadline, self.work_units, self.index)


class Environment(object):

    def __init__(self):

        self.time = 0.0
        self.processor_count = PROCESSOR_COUNT
        self.task_count = PROCESSOR_COUNT * TASK_PER_PROCESSOR
        self.task_set: List[Task] = []
        self.event_queue = EventQueue()
        self.instance_arrival_count = 0
        self.active_instances: List[Instance] = []
        self.total_energy_consumed = 0.0

        self.stats = {
            "simulation_progress": 0,
            "normalized_instance_count": 0,
            "system_load": 0,
            "remaining_work_units": {"min": 0, "mean": 0, "max": 0},
            "deadline": {"min": 0, "mean": 0, "max": 0},
            "laxity": {"min": 0, "mean": 0, "max": 0}
        }


    def reset(self, per_core_utilization: float = None):

        self.time = 0.0
        self.instance_arrival_count = 0
        self.active_instances = []
        self.total_energy_consumed = 0.0

        if per_core_utilization is None:
            per_core_utilization = np.random.uniform(MIN_LOAD, MAX_LOAD)

        target_util = per_core_utilization * self.processor_count
        utilizations = StaffordRandFixedSum(self.task_count, target_util, 1).flatten()
        periods = gen_periods(self.task_count, 1, MIN_PERIOD, MAX_PERIOD, 0.1, "logunif").flatten()

        self.task_set = [
            Task(idx, utilizations[idx], periods[idx]) for idx in range(self.task_count)
        ]

        self.event_queue.reset()
        for task in self.task_set:
            instance = task.create_instance(self.time)
            self.push_instance_to_event_queue(instance)

        self.time = self.event_queue.peek_next_timestamp()
        self.process_events_at_current_time()
        self.update_env_stats()
    

    def push_instance_to_event_queue(self, instance: Instance):
        self.event_queue.push_event(Event(instance.arrival_time, EventType.ARRIVAL, instance))
        self.event_queue.push_event(Event(instance.deadline, EventType.DEADLINE, instance))


    def process_events_at_current_time(self):

        events = self.event_queue.pop_events()
        for event in events:
            instance = event.data
            instance.update_status(self.time)

            if event.event_type == EventType.ARRIVAL:
                self.active_instances.append(instance)
                self.instance_arrival_count += 1

                parent_task = self.task_set[instance.task_index]
                next_instance = parent_task.create_instance(self.time)
                self.push_instance_to_event_queue(next_instance)


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
                time_duration (float): The actual time duration elapsed in this step.
        """
        if not self.active_instances:
            previous_time = self.time
            self.time = self.event_queue.peek_next_timestamp()
            self.process_events_at_current_time()
            self.update_env_stats()
            time_duration = self.time - previous_time
            idle_energy = (STATIC_POWER_COEFF * 0.25 * self.processor_count) * time_duration
            self.total_energy_consumed += idle_energy
            energy_penalty = ENERGY_PENALTY_COEFF * idle_energy
            global_reward = -energy_penalty
            return global_reward, self.get_state(), self.done(), 0, 0, time_duration

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

        num_active_processors = len(exec_indices)
        num_idle_processors = self.processor_count - num_active_processors
        idle_energy = (num_idle_processors * STATIC_POWER_COEFF) * duration

        active_energy = 0.0
        for i in exec_indices:
            instance = self.active_instances[i]
            freq = frequency_scales[i]
            instance.execute(duration, freq)
            static_power = STATIC_POWER_COEFF * freq
            dynamic_power = DYNAMIC_POWER_COEFF * (freq ** 3)
            active_energy += (static_power + dynamic_power) * duration

        step_energy = idle_energy + active_energy
        self.total_energy_consumed += step_energy

        # Advance simulation time
        self.time = next_timestamp

        completed_count, missed_count = 0, 0
        efficiency_reward = 0
        for instance in self.active_instances:
            instance.update_status(self.time)
            if instance.status == InstanceStatus.COMPLETED:
                completed_count += 1
                final_laxity = max(0, instance.deadline - self.time)
                initial_time = self.task_set[instance.task_index].relative_deadline
                if final_laxity / initial_time < 0.1:
                    efficiency_reward += 1
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
        global_reward = INSTANCE_COMPLETION_REWARD * efficiency_reward
        global_reward -= INSTANCE_MISS_PENALTY * missed_count
        global_reward -= ENERGY_PENALTY_COEFF * step_energy

        self.update_env_stats()
        return global_reward, self.get_state(), self.done(), completed_count, missed_count, duration


    def update_env_stats(self):

        self.stats["simulation_progress"] = self.time / MAX_EPISODE_TIME

        if len(self.active_instances) == 0:
            for key, value in self.stats.items():
                if isinstance(value, dict):
                    for stat in value:
                        self.stats[key][stat] = 0

            self.stats["normalized_instance_count"] = 0
            self.stats["system_load"] = 0
            return

        total_load = sum(i.remaining_work_units for i in self.active_instances)
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


    def get_state(self):

        global_state_critic = [
            self.stats["simulation_progress"],
            self.stats["normalized_instance_count"],
            self.stats["system_load"] / MAX_PERIOD,
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
        global_state_critic = np.array(global_state_critic, dtype=np.float32)

        if len(self.active_instances) == 0:
            local_observation = np.array([0., 0., 0.], dtype=np.float32)
            state_critic = np.concatenate((local_observation, global_state_critic))
            state_critic = state_critic.reshape(1, CRITIC_STATE_DIM)
            state_actor = np.zeros((0, ACTOR_STATE_DIM), dtype=np.float32)
            return state_actor, state_critic

        global_state_actor = [
            self.stats["normalized_instance_count"],
            self.stats["system_load"] / MAX_PERIOD,
            self.stats["remaining_work_units"]["min"] / MAX_PERIOD,
            self.stats["remaining_work_units"]["mean"] / MAX_PERIOD,
            self.stats["remaining_work_units"]["max"] / MAX_PERIOD,
            self.stats["deadline"]["min"] / MAX_PERIOD,
            self.stats["deadline"]["mean"] / MAX_PERIOD,
            self.stats["deadline"]["max"] / MAX_PERIOD,
        ]
        global_state_actor = np.array(global_state_actor, dtype=np.float32)

        # mean_remaining_work_units = self.stats["remaining_work_units"]["mean"] + 1e-6
        # mean_deadline = self.stats["deadline"]["mean"] + 1e-6
        # mean_laxity = self.stats["laxity"]["mean"] + 1e-6

        local_observations = []
        for instance in self.active_instances:
            local_obs = [
                instance.relative_deadline(self.time) / MAX_PERIOD,
                # instance.laxity(self.time) / MAX_PERIOD,
                instance.remaining_work_units / MAX_PERIOD,
                instance.remaining_work_units / instance.initial_work_units,
            ]
            local_observations.append(local_obs)

        local_observations = np.array(local_observations, dtype=np.float32)

        global_observations_actor = np.tile(global_state_actor, (len(self.active_instances), 1))
        global_observations_critic = np.tile(global_state_critic, (len(self.active_instances), 1))

        state_actor = np.hstack((local_observations, global_observations_actor))
        state_critic = np.hstack((local_observations, global_observations_critic))

        return state_actor, state_critic


    def done(self) -> bool:
        return self.time >= MAX_EPISODE_TIME or self.event_queue.is_empty()

    # def calc_mean_utilization(self) -> float:
    #     utils = [task.work_units / np.mean(task.arrival_intervals) for task in self.task_set]
    #     return np.sum(utils) / self.processor_count