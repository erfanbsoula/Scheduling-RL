import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import *
from env import Environment, Task
from ddpg_torch import ReplayBuffer, MADDPG
from task_gen import StaffordRandFixedSum, gen_periods

np.set_printoptions(precision=4, suppress=True)

# Set fixed seeds for reproducibility
SEED = 199686
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(SAVE_PATH, exist_ok=True)

environment = Environment()
replay_buffer = ReplayBuffer(BUFFER_SIZE)

algorithm = MADDPG(replay_buffer)

start_noise_scale = float(os.getenv('GRID_START_NOISE_SCALE', 0.1))
end_noise_scale = 1e-4
noise_decay = (end_noise_scale / start_noise_scale) ** (1 / MAX_EPISODES)
noise_scale = start_noise_scale

rewards_log = []
success_ratio_log = []
energy_consumption_log = []
frequency_scale_log = []

# Generate a fixed task set only once
def generate_fixed_task_set(per_core_utilization=None):
    """
    Generate a fixed task set that will be used throughout training.
    Only arrival times will be regenerated for each episode.
    """
    if per_core_utilization is None:
        per_core_utilization = np.random.uniform(MIN_LOAD, MAX_LOAD)

    processor_count = PROCESSOR_COUNT
    task_count = processor_count * TASK_PER_PROCESSOR

    # Generate fixed utilization and periods for the task set
    target_util = per_core_utilization * processor_count
    utilizations = StaffordRandFixedSum(task_count, target_util, 1).flatten()
    periods = gen_periods(task_count, 1, MIN_PERIOD, MAX_PERIOD, 0.1, "logunif").flatten()

    print(f"Generated fixed task set with {task_count} tasks")
    print(f"System utilization: {target_util:.4f} ({per_core_utilization:.4f} per core)")
    print(f"Task utilizations: {utilizations}")
    print(f"Task periods: {periods}")

    # Save task set parameters for reproducibility
    np.savez(
        os.path.join(SAVE_PATH, 'task_set_params.npz'),
        utilizations=utilizations,
        periods=periods,
        system_utilization=target_util,
        per_core_utilization=per_core_utilization
    )

    return utilizations, periods

frequency_scale_log_tmp = []
def log_frequency_scales(algorithm: MADDPG, state: np.ndarray):

    with torch.no_grad():
        state = torch.FloatTensor(state)
        x = torch.tanh(algorithm.policy_net.linear1(state))
        x = torch.tanh(algorithm.policy_net.linear2(x))
        x = algorithm.policy_net.linear3(x)
        x = torch.mean(x[:, 1])
        frequency_scale_log_tmp.append(x.item())

# Generate the fixed task set parameters
fixed_utilizations, fixed_periods = generate_fixed_task_set(0.7)

class TaskSetEnvironment(Environment):
    """
    Extended Environment class that uses a fixed task set but
    regenerates arrival times for each episode.
    """
    def reset(self, per_core_utilization=None):
        """
        Override the reset method to use the fixed task set
        but regenerate arrival times.
        """
        self.time = 0.0
        self.instance_arrival_count = 0
        self.active_instances = []
        self.total_energy_consumed = 0.0

        self.task_set = [
            Task(idx, fixed_utilizations[idx], fixed_periods[idx])
            for idx in range(self.task_count)
        ]

        self.event_queue.reset()
        for task in self.task_set:
            instance = task.create_instance(self.time)
            self.push_instance_to_event_queue(instance)

        self.time = self.event_queue.peek_next_timestamp()
        self.process_events_at_current_time()
        self.update_env_stats()


# Use our custom environment class
environment = TaskSetEnvironment()

# Main training loop
for i_episode in range(1, MAX_EPISODES+1):
    environment.reset()  # Reset with the fixed task set but new arrival times
    next_state = environment.get_state()
    noise_scale *= noise_decay

    print(f"--- Episode {i_episode} ---")

    q_loss_list = []
    policy_loss_list = []
    episode_reward_sum = 0
    total_completed_in_episode = 0
    total_missed_in_episode = 0
    frequency_scale_log_tmp = []

    for step in range(MAX_STEPS):
        current_state_actor, current_state_critic = next_state
        num_active_instances = len(environment.active_instances)

        if num_active_instances > 0:
            action = algorithm.policy_net.select_action(current_state_actor, noise_std=noise_scale)
            log_frequency_scales(algorithm, current_state_actor)
        else:
            action = np.zeros((1, ACTION_DIM), dtype=np.float32)

        scheduling_priorities = action[:, 0]
        frequency_scales = action[:, 1]

        if frequency_scales.size > 0:
            num_levels = len(DVFS_LEVELS)
            level_indices = np.floor(action[:, 1] * num_levels).astype(int)
            level_indices = np.clip(level_indices, 0, num_levels - 1)
            frequency_scales = np.array([DVFS_LEVELS[i] for i in level_indices]).astype(np.float32)

        transition = environment.step(scheduling_priorities, frequency_scales)
        global_reward, next_state, is_done, num_completed, num_missed, time_duration = transition
        next_state_actor, next_state_critic = next_state

        replay_buffer.push(current_state_actor, current_state_critic, action, global_reward,
                           next_state_actor, next_state_critic, is_done, time_duration)

        episode_reward_sum += global_reward
        total_completed_in_episode += num_completed
        total_missed_in_episode += num_missed

        if len(replay_buffer) > BATCH_SIZE and (step + 1) % UPDATE_INTERVAL == 0:
            for _ in range(UPDATE_REPEAT_COUNT):
                q_loss, policy_loss = algorithm.update(BATCH_SIZE, SOFT_UPDATE_TAU)
                q_loss_list.append(q_loss)
                policy_loss_list.append(policy_loss)

        if is_done:
            break

    # End of episode
    rewards_log.append(episode_reward_sum)

    total_tasks_in_episode = total_completed_in_episode + total_missed_in_episode
    success_ratio = total_completed_in_episode / total_tasks_in_episode * 100 if total_tasks_in_episode > 0 else 0
    success_ratio_log.append(success_ratio)
    energy_consumption_log.append(environment.total_energy_consumed)
    frequency_scale_log.append(np.mean(frequency_scale_log_tmp) if frequency_scale_log_tmp else 0)

    avg_q_loss = np.mean(q_loss_list) if q_loss_list else 0
    avg_policy_loss = np.mean(policy_loss_list) if policy_loss_list else 0

    print(f"Episode Reward: {episode_reward_sum}")
    print(f"Episode Success Ratio: {success_ratio:.2f}%")
    print(f"Total Completed: {total_completed_in_episode}/{total_tasks_in_episode}")
    print(f"Total Missed: {total_missed_in_episode}/{total_tasks_in_episode}")
    print(f"Total Energy Consumed: {environment.total_energy_consumed:.2f}")
    print(f"Avg Q_Loss: {avg_q_loss:.4f}, Avg Policy_Loss: {avg_policy_loss:.4f}")
    print(f"Noise Scale: {noise_scale:.4f}")
    print(f"Replay Buffer Size: {len(replay_buffer)}")
    print(f"Total Steps: {step + 1}")

    if i_episode % CHECKPOINT_INTERVAL == 0 or i_episode == MAX_EPISODES:
        # Save reward plot
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(rewards_log)
        plt.title("Episode Reward Trend")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        # Save success ratio plot
        plt.subplot(3, 1, 2)
        plt.plot(success_ratio_log)
        plt.title("Success Ratio Trend")
        plt.xlabel("Episode")
        plt.ylabel("Success Ratio (%)")

        # Save energy consumption plot
        plt.subplot(3, 1, 3)
        plt.plot(energy_consumption_log)
        plt.title("Energy Consumption Trend")
        plt.xlabel("Episode")
        plt.ylabel("Energy")

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_PATH, "training_metrics.png"), dpi=300, bbox_inches='tight')
        plt.close()

        plt.plot(frequency_scale_log)
        plt.title("Frequency Scale Trend")
        plt.xlabel("Episode")
        plt.ylabel("Frequency Scale")
        plt.savefig(os.path.join(SAVE_PATH, "frequency_scales.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Save model checkpoints
        model_path = os.path.join(SAVE_PATH, f"ep_{i_episode}")
        algorithm.save_model(model_path)
        print(f"Models saved to {model_path}")

        # Save training metrics
        np.savez(
            os.path.join(SAVE_PATH, "training_metrics.npz"),
            rewards=np.array(rewards_log),
            success_ratios=np.array(success_ratio_log),
            energy_consumption=np.array(energy_consumption_log)
        )

print("Training finished.")
