import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from environment import Environment
from ddpg_torch import MADDPG
from config import (
    SAVE_PATH,
    RANDOM_SEED,
    CURRENT_LOAD,
    DVFS_LEVELS
)


def convert_to_discrete_levels(frequency_scales: np.ndarray) -> np.ndarray:
    """
    Converts continuous frequency scales to discrete DVFS levels.

    Args:
        frequency_scales (np.ndarray): Array of continuous frequency scales in [0, 1].

    Returns:
        np.ndarray: Array of discrete DVFS levels.
    """
    num_levels = len(DVFS_LEVELS)
    level_indices = np.floor(frequency_scales * num_levels).astype(int)
    level_indices = np.clip(level_indices, 0, num_levels - 1)
    discrete_levels = np.array([DVFS_LEVELS[i] for i in level_indices]).astype(np.float32)
    return discrete_levels


def gedf_scheduler(active_instances: list):
    """
    Schedules tasks based on Global Earliest Deadline First (GEDF).
    Always uses maximum frequency for scheduled tasks.

    Args:
        active_instances (list): A list of active Instance objects.

    Returns:
        tuple: (scheduling_priorities, frequency_scales)
            scheduling_priorities (np.ndarray): Array of scheduling priorities for each instance.
            frequency_scales (np.ndarray): Array of frequency levels for each instance.
    """
    num_active_instances = len(active_instances)
    scheduling_priorities = np.zeros(num_active_instances, dtype=float)
    frequency_scales = np.full(num_active_instances, DVFS_LEVELS[-1], dtype=np.float32)

    if num_active_instances == 0:
        return scheduling_priorities, frequency_scales

    for i in range(num_active_instances):
        scheduling_priorities[i] = 1 / active_instances[i].deadline

    return scheduling_priorities, frequency_scales


def es_dvfs_scheduler(active_instances: list):
    """
    Schedules tasks based on the ES-DVFS algorithm.
    Calculates speed based on workload and intensity, and schedules by EDF.

    Args:
        active_instances (list): A list of active Instance objects.

    Returns:
        tuple: (scheduling_priorities, frequency_scales)
            scheduling_priorities (np.ndarray): Array of scheduling priorities for each instance.
            frequency_scales (np.ndarray): Array of frequency levels for each instance.
    """
    num_active_instances = len(active_instances)
    if num_active_instances == 0:
        return np.array([]), np.array([])

    total_remaining_work = sum(inst.remaining_work_units for inst in active_instances)
    max_deadline = max(inst.deadline for inst in active_instances)
    h_k = total_remaining_work / max_deadline if max_deadline > 0 else float('inf')

    sorted_instances = sorted(active_instances, key=lambda inst: inst.deadline)
    cumulative_work = 0
    max_intensity = 0
    for instance in sorted_instances:
        cumulative_work += instance.remaining_work_units
        if instance.deadline > 0:
            intensity = cumulative_work / instance.deadline
            if intensity > max_intensity:
                max_intensity = intensity

    I_j = max_intensity

    speed = max(h_k, I_j)
    frequency_scales = np.full(num_active_instances, speed, dtype=np.float32)
    frequency_scales = convert_to_discrete_levels(frequency_scales)

    scheduling_priorities = np.zeros(num_active_instances, dtype=float)
    for i in range(num_active_instances):
        scheduling_priorities[i] = 1 / (active_instances[i].deadline + 1e-6)

    return scheduling_priorities, frequency_scales


class Tester(object):

    def __init__(self):

        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        np.set_printoptions(precision=4, suppress=True)

        self.environment = Environment()
        self.rl_agent = MADDPG(None)

        model_files = [d for d in os.listdir(SAVE_PATH) if d.startswith('ep_')]
        if not model_files:
            print(f"No saved models found in {SAVE_PATH}. Please train the RL agent first.")
            return 1
        
        model_files.sort(key=lambda x: int(x.split('_')[1]), reverse=True)
        latest_model_path = os.path.join(SAVE_PATH, model_files[0])
        print(f"Loading RL agent model from: {latest_model_path}")
        self.rl_agent.load_model(latest_model_path)

        self.results = {
            'rl': {'success_ratio': [], 'energy': [], 'completed': [], 'missed': []},
            'gedf': {'success_ratio': [], 'energy': [], 'completed': [], 'missed': []},
            'es-dvfs': {'success_ratio': [], 'energy': [], 'completed': [], 'missed': []}
        }
    

    def run_simulation(self, scheduler_type: str):
        """
        Runs a simulation for a given scheduler type.

        Args:
            scheduler_type (str): 'rl', 'gedf', or 'es-dvfs'.

        Returns:
            tuple: (
                success_ratio, total_energy_consumed,
                total_completed_in_episode,
                total_missed_in_episode
            )
        """
        episode_reward_sum = 0
        total_completed_in_episode = 0
        total_missed_in_episode = 0

        if scheduler_type not in ['rl', 'gedf', 'es-dvfs']:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        next_state = self.environment.get_state()

        while not self.environment.done():
            current_state_actor, current_state_critic = next_state
            active_instances = self.environment.active_instances

            if scheduler_type == 'rl':
                if len(active_instances) > 0:
                    action = self.rl_agent.policy_net.select_action(current_state_actor, noise_width=0.0)
                    scheduling_priorities = action[:, 0]
                    frequency_scales = convert_to_discrete_levels(action[:, 1])
                else:
                    scheduling_priorities = np.array([])
                    frequency_scales = np.array([])

            elif scheduler_type == 'gedf':
                scheduling_priorities, frequency_scales = gedf_scheduler(active_instances)

            elif scheduler_type == 'es-dvfs':
                scheduling_priorities, frequency_scales = es_dvfs_scheduler(active_instances)                

            transition = self.environment.step(scheduling_priorities, frequency_scales)
            global_reward, next_state, is_done, num_completed, num_missed, time_duration = transition

            episode_reward_sum += global_reward
            total_completed_in_episode += num_completed
            total_missed_in_episode += num_missed

            if is_done: break

        total_energy_consumed = self.environment.total_energy_consumed
        total_tasks_in_episode = total_completed_in_episode + total_missed_in_episode
        success_ratio = (total_completed_in_episode / total_tasks_in_episode * 100)
        return success_ratio, total_energy_consumed, total_completed_in_episode, total_missed_in_episode


    def plot_results(self):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Success ratio comparison (bar plot)
        success_means = [
            self.results['rl']['avg_success_ratio'],
            self.results['gedf']['avg_success_ratio'],
            self.results['es-dvfs']['avg_success_ratio']
        ]

        ax1.bar(['RL', 'GEDF', 'ES-DVFS'], success_means, color=plt.cm.tab10.colors[:3])
        ax1.set_title('Success Ratio Comparison', fontsize=14)
        ax1.set_ylabel('Success Ratio (%)', fontsize=12)
        ax1.grid(True, linestyle=':', alpha=0.7)

        # Energy consumption comparison (bar plot)
        energy_means = [
            self.results['rl']['avg_energy'],
            self.results['gedf']['avg_energy'],
            self.results['es-dvfs']['avg_energy']
        ]

        ax2.bar(['RL', 'GEDF', 'ES-DVFS'], energy_means, color=plt.cm.tab10.colors[:3])
        ax2.set_title('Energy Consumption Comparison', fontsize=14)
        ax2.set_ylabel('Energy Consumption', fontsize=12)
        ax2.grid(True, linestyle=':', alpha=0.7)

        plt.tight_layout()
        plot_path = os.path.join(SAVE_PATH, 'test_result.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Results plot saved to {plot_path}")


    def save_summary(self):

        summary_path = os.path.join(SAVE_PATH, 'test_summary.txt')
        
        with open(summary_path, 'w') as f:

            def write_and_print(line):
                print(line)
                f.write(line + "\n")
                
            write_and_print(f"Per-core Utilization: {CURRENT_LOAD:.4f}")
            write_and_print("\n--- Success Ratio Summary ---")
            write_and_print(f"{'Scheduler':<10} | {'Avg (%)':<10} | {'Std Dev':<10} | {'Min (%)':<10} | {'Max (%)':<10}")
            write_and_print("-" * 60)
            
            for scheduler in ['rl', 'gedf', 'es-dvfs']:
                avg = self.results[scheduler]['avg_success_ratio']
                std = self.results[scheduler]['std_success_ratio']
                min_val = min(self.results[scheduler]['success_ratio'])
                max_val = max(self.results[scheduler]['success_ratio'])
                
                write_and_print(f"{scheduler.upper():<10} | {avg:<10.2f} | {std:<10.2f} | {min_val:<10.2f} | {max_val:<10.2f}")
            
            write_and_print("\n--- Energy Consumption Summary ---")
            write_and_print(f"{'Scheduler':<10} | {'Avg':<10} | {'Std Dev':<10} | {'Min':<10} | {'Max':<10}")
            write_and_print("-" * 60)
            
            for scheduler in ['rl', 'gedf', 'es-dvfs']:
                avg = self.results[scheduler]['avg_energy']
                std = self.results[scheduler]['std_energy']
                min_val = min(self.results[scheduler]['energy'])
                max_val = max(self.results[scheduler]['energy'])

                write_and_print(f"{scheduler.upper():<10} | {avg:<10.2f} | {std:<10.2f} | {min_val:<10.2f} | {max_val:<10.2f}")
                
            # Calculate energy savings percentage
            rl_avg_energy = self.results['rl']['avg_energy']
            gedf_avg_energy = self.results['gedf']['avg_energy']
            es_dvfs_avg_energy = self.results['es-dvfs']['avg_energy']

            rl_vs_gedf = ((gedf_avg_energy - rl_avg_energy) / gedf_avg_energy) * 100
            rl_vs_es_dvfs = ((es_dvfs_avg_energy - rl_avg_energy) / es_dvfs_avg_energy) * 100
            
            write_and_print("\n--- Energy Savings ---")
            write_and_print(f"RL vs GEDF: {rl_vs_gedf:.2f}% savings")
            write_and_print(f"RL vs ES-DVFS: {rl_vs_es_dvfs:.2f}% savings")
            
            # Task completion statistics
            write_and_print("\n--- Task Completion Statistics ---")
            write_and_print(f"{'Scheduler':<10} | {'Avg Completed':<15} | {'Avg Missed':<15} | {'Success Rate':<15}")
            write_and_print("-" * 65)
            
            for scheduler in ['rl', 'gedf', 'es-dvfs']:
                avg_completed = self.results[scheduler]['avg_completed']
                avg_missed = self.results[scheduler]['avg_missed']
                success_rate = avg_completed / (avg_completed + avg_missed) * 100
                
                write_and_print(f"{scheduler.upper():<10} | {avg_completed:<15.2f} | {avg_missed:<15.2f} | {success_rate:<15.2f}%")
        
        print(f"Summary saved to {summary_path}")


    def run(self, num_test_runs=100):

        for i in range(num_test_runs):
            print(f"\nTest run {i+1}/{num_test_runs}")

            # Test RL agent
            self.environment.reset(CURRENT_LOAD)
            success_rl, energy_rl, completed_rl, missed_rl = self.run_simulation('rl')
            self.results['rl']['success_ratio'].append(success_rl)
            self.results['rl']['energy'].append(energy_rl)
            self.results['rl']['completed'].append(completed_rl)
            self.results['rl']['missed'].append(missed_rl)

            # Test GEDF
            self.environment.reset(CURRENT_LOAD)
            success_gedf, energy_gedf, completed_gedf, missed_gedf = self.run_simulation('gedf')
            self.results['gedf']['success_ratio'].append(success_gedf)
            self.results['gedf']['energy'].append(energy_gedf)
            self.results['gedf']['completed'].append(completed_gedf)
            self.results['gedf']['missed'].append(missed_gedf)
            
            # Test ES-DVFS
            self.environment.reset(CURRENT_LOAD)
            success_es_dvfs, energy_es_dvfs, completed_es_dvfs, missed_es_dvfs = self.run_simulation('es-dvfs')
            self.results['es-dvfs']['success_ratio'].append(success_es_dvfs)
            self.results['es-dvfs']['energy'].append(energy_es_dvfs)
            self.results['es-dvfs']['completed'].append(completed_es_dvfs)
            self.results['es-dvfs']['missed'].append(missed_es_dvfs)

            print(f"RL      - Success: {success_rl:.2f}%, Energy: {energy_rl:.2f}, Completed: {completed_rl}, Missed: {missed_rl}")
            print(f"GEDF    - Success: {success_gedf:.2f}%, Energy: {energy_gedf:.2f}, Completed: {completed_gedf}, Missed: {missed_gedf}")
            print(f"ES-DVFS - Success: {success_es_dvfs:.2f}%, Energy: {energy_es_dvfs:.2f}, Completed: {completed_es_dvfs}, Missed: {missed_es_dvfs}")

        # Calculate averages
        metrics_to_analyze = ['success_ratio', 'energy', 'completed', 'missed']
        for scheduler in self.results:
            for metric in metrics_to_analyze:
                self.results[scheduler][f'avg_{metric}'] = np.mean(self.results[scheduler][metric])
                self.results[scheduler][f'std_{metric}'] = np.std(self.results[scheduler][metric])

        self.plot_results()
        self.save_summary()


if __name__ == "__main__":

    tester = Tester()
    tester.run()