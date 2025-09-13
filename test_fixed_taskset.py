import os
import numpy as np
import matplotlib.pyplot as plt
from config import *
from env import Environment, Task
from ddpg_torch import MADDPG
from test import gedf_scheduler, es_dvfs_scheduler

np.set_printoptions(precision=4, suppress=True)


class TaskSetEnvironment(Environment):
    """
    Extended Environment class that uses a fixed task set but
    regenerates arrival times for each episode.
    """
    def reset(self, task_utilizations, task_periods):
        """
        Reset the environment using the fixed task set parameters
        but with new arrival times.
        
        Args:
            task_utilizations: Array of utilizations for each task
            task_periods: Array of periods for each task
        """
        self.time = 0.0
        self.instance_arrival_count = 0
        self.active_instances = []
        self.total_energy_consumed = 0.0

        self.task_set = [
            Task(idx, task_utilizations[idx], task_periods[idx])
            for idx in range(self.task_count)
        ]

        self.event_queue.reset()
        for task in self.task_set:
            instance = task.create_instance(self.time)
            self.push_instance_to_event_queue(instance)

        self.time = self.event_queue.peek_next_timestamp()
        self.process_events_at_current_time()
        self.update_env_stats()


def run_simulation(
        environment: TaskSetEnvironment,
        scheduler_type: str,
        rl_agent: MADDPG = None,
    ):
    """
    Runs a simulation for a given scheduler type.

    Args:
        environment (TaskSetEnvironment): The simulation environment.
        scheduler_type (str): 'rl', 'gedf', or 'es-dvfs'.
        rl_agent (MADDPG, optional): The RL agent, required if scheduler_type is 'rl'.

    Returns:
        tuple: (success_ratio, total_energy_consumed)
    """
    episode_reward_sum = 0
    total_completed_in_episode = 0
    total_missed_in_episode = 0

    next_state = environment.get_state()

    while not environment.done():
        current_state_actor, current_state_critic = next_state
        num_active_instances = len(environment.active_instances)

        if scheduler_type == 'rl':
            if num_active_instances > 0:
                action = rl_agent.policy_net.select_action(current_state_actor, noise_width=0.0)
                scheduling_priorities = action[:, 0]
                frequency_scales = action[:, 1]

                num_levels = len(DVFS_LEVELS)
                level_indices = np.floor(frequency_scales * num_levels).astype(int)
                level_indices = np.clip(level_indices, 0, num_levels - 1)
                frequency_scales = np.array([DVFS_LEVELS[i] for i in level_indices]).astype(np.float32)
            else:
                scheduling_priorities = np.array([])
                frequency_scales = np.array([])

        elif scheduler_type == 'gedf':
            scheduling_priorities, frequency_scales = gedf_scheduler(environment.active_instances)

        elif scheduler_type == 'es-dvfs':
            scheduling_priorities, frequency_scales = es_dvfs_scheduler(environment.active_instances)

        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        transition = environment.step(scheduling_priorities, frequency_scales)
        global_reward, next_state, is_done, num_completed, num_missed, time_duration = transition

        episode_reward_sum += global_reward
        total_completed_in_episode += num_completed
        total_missed_in_episode += num_missed

        if is_done:
            break

    total_tasks_in_episode = total_completed_in_episode + total_missed_in_episode
    success_ratio = (total_completed_in_episode / total_tasks_in_episode * 100) if total_tasks_in_episode > 0 else 0

    return success_ratio, environment.total_energy_consumed, total_completed_in_episode, total_missed_in_episode


def test_fixed_taskset(num_test_runs=30):
    """
    Test the RL agent against baseline schedulers on the fixed task set.
    
    Args:
        num_test_runs (int): Number of test runs to perform with different arrival times
    """
    print("Starting fixed task set performance comparison...")
    
    # Load the fixed task set parameters
    task_params_file = os.path.join(SAVE_PATH, 'task_set_params.npz')
    if not os.path.exists(task_params_file):
        print(f"Task set parameters file not found at {task_params_file}. Please train the model first.")
        return
        
    task_params = np.load(task_params_file)
    task_utilizations = task_params['utilizations']
    task_periods = task_params['periods']
    per_core_utilization = task_params['per_core_utilization']
    
    print(f"Loaded fixed task set with per-core utilization: {per_core_utilization:.4f}")
    print(f"Testing with {num_test_runs} different arrival time patterns")
    
    # Load the trained RL model
    environment = TaskSetEnvironment()
    rl_agent = MADDPG(None)
    
    model_files = [d for d in os.listdir(SAVE_PATH) if d.startswith('ep_')]
    if not model_files:
        print(f"No saved models found in {SAVE_PATH}. Please train the RL agent first.")
        return
        
    model_files.sort(key=lambda x: int(x.split('_')[1]), reverse=True)
    latest_model_path = os.path.join(SAVE_PATH, model_files[0])
    print(f"Loading RL agent model from: {latest_model_path}")
    rl_agent.load_model(latest_model_path)
    
    # Initialize results storage
    results = {
        'rl': {'success_ratio': [], 'energy': [], 'completed': [], 'missed': []},
        'gedf': {'success_ratio': [], 'energy': [], 'completed': [], 'missed': []},
        'es-dvfs': {'success_ratio': [], 'energy': [], 'completed': [], 'missed': []}
    }
    
    # Run tests
    for i in range(num_test_runs):
        print(f"\nTest run {i+1}/{num_test_runs}")
        
        # Test RL agent
        environment.reset(task_utilizations, task_periods)
        success_rl, energy_rl, completed_rl, missed_rl = run_simulation(environment, 'rl', rl_agent=rl_agent)
        results['rl']['success_ratio'].append(success_rl)
        results['rl']['energy'].append(energy_rl)
        results['rl']['completed'].append(completed_rl)
        results['rl']['missed'].append(missed_rl)
        
        # Test GEDF
        environment.reset(task_utilizations, task_periods)
        success_gedf, energy_gedf, completed_gedf, missed_gedf = run_simulation(environment, 'gedf')
        results['gedf']['success_ratio'].append(success_gedf)
        results['gedf']['energy'].append(energy_gedf)
        results['gedf']['completed'].append(completed_gedf)
        results['gedf']['missed'].append(missed_gedf)
        
        # Test ES-DVFS
        environment.reset(task_utilizations, task_periods)
        success_es_dvfs, energy_es_dvfs, completed_es_dvfs, missed_es_dvfs = run_simulation(environment, 'es-dvfs')
        results['es-dvfs']['success_ratio'].append(success_es_dvfs)
        results['es-dvfs']['energy'].append(energy_es_dvfs)
        results['es-dvfs']['completed'].append(completed_es_dvfs)
        results['es-dvfs']['missed'].append(missed_es_dvfs)
        
        print(f"RL   - Success: {success_rl:.2f}%, Energy: {energy_rl:.2f}, Completed: {completed_rl}, Missed: {missed_rl}")
        print(f"GEDF - Success: {success_gedf:.2f}%, Energy: {energy_gedf:.2f}, Completed: {completed_gedf}, Missed: {missed_gedf}")
        print(f"ES-DVFS - Success: {success_es_dvfs:.2f}%, Energy: {energy_es_dvfs:.2f}, Completed: {completed_es_dvfs}, Missed: {missed_es_dvfs}")
    
    # Calculate averages
    metrics_to_analyze = ['success_ratio', 'energy', 'completed', 'missed']
    for scheduler in results:
        for metric in metrics_to_analyze:
            results[scheduler][f'avg_{metric}'] = np.mean(results[scheduler][metric])
            results[scheduler][f'std_{metric}'] = np.std(results[scheduler][metric])
    
    # Plot results
    plot_fixed_taskset_results(results)
    
    # Save summary
    save_fixed_taskset_summary(results, per_core_utilization)
    
    return results


def plot_fixed_taskset_results(results):
    """
    Plot the results of the fixed task set testing.
    
    Args:
        results (dict): Dictionary containing all test results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Success ratio comparison (bar plot)
    success_means = [
        results['rl']['avg_success_ratio'],
        results['gedf']['avg_success_ratio'],
        results['es-dvfs']['avg_success_ratio']
    ]

    ax1.bar(['RL', 'GEDF', 'ES-DVFS'], success_means, color=plt.cm.tab10.colors[:3])
    ax1.set_title('Success Ratio Comparison', fontsize=14)
    ax1.set_ylabel('Success Ratio (%)', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Energy consumption comparison (bar plot)
    energy_means = [
        results['rl']['avg_energy'],
        results['gedf']['avg_energy'],
        results['es-dvfs']['avg_energy']
    ]

    ax2.bar(['RL', 'GEDF', 'ES-DVFS'], energy_means, color=plt.cm.tab10.colors[:3])
    ax2.set_title('Energy Consumption Comparison', fontsize=14)
    ax2.set_ylabel('Energy Consumption', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(SAVE_PATH, 'fixed_taskset_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Results plot saved to {plot_path}")


def save_fixed_taskset_summary(results, utilization):
    """
    Save a summary of the fixed task set test results.
    
    Args:
        results (dict): Dictionary containing all test results
        utilization (float): The per-core utilization of the fixed task set
    """
    summary_path = os.path.join(SAVE_PATH, 'fixed_taskset_summary.txt')
    
    with open(summary_path, 'w') as f:
        def write_and_print(line):
            print(line)
            f.write(line + "\n")
            
        write_and_print(f"Fixed Task Set Test Results - Per-core Utilization: {utilization:.4f}")
        write_and_print("\n--- Success Ratio Summary ---")
        write_and_print(f"{'Scheduler':<10} | {'Avg (%)':<10} | {'Std Dev':<10} | {'Min (%)':<10} | {'Max (%)':<10}")
        write_and_print("-" * 60)
        
        for scheduler in ['rl', 'gedf', 'es-dvfs']:
            avg = results[scheduler]['avg_success_ratio']
            std = results[scheduler]['std_success_ratio']
            min_val = min(results[scheduler]['success_ratio'])
            max_val = max(results[scheduler]['success_ratio'])
            
            write_and_print(f"{scheduler.upper():<10} | {avg:<10.2f} | {std:<10.2f} | {min_val:<10.2f} | {max_val:<10.2f}")
        
        write_and_print("\n--- Energy Consumption Summary ---")
        write_and_print(f"{'Scheduler':<10} | {'Avg':<10} | {'Std Dev':<10} | {'Min':<10} | {'Max':<10}")
        write_and_print("-" * 60)
        
        for scheduler in ['rl', 'gedf', 'es-dvfs']:
            avg = results[scheduler]['avg_energy']
            std = results[scheduler]['std_energy']
            min_val = min(results[scheduler]['energy'])
            max_val = max(results[scheduler]['energy'])
            
            write_and_print(f"{scheduler.upper():<10} | {avg:<10.2f} | {std:<10.2f} | {min_val:<10.2f} | {max_val:<10.2f}")
            
        # Calculate energy savings percentage
        rl_avg_energy = results['rl']['avg_energy']
        gedf_avg_energy = results['gedf']['avg_energy']
        es_dvfs_avg_energy = results['es-dvfs']['avg_energy']
        
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
            avg_completed = results[scheduler]['avg_completed']
            avg_missed = results[scheduler]['avg_missed']
            success_rate = avg_completed / (avg_completed + avg_missed) * 100
            
            write_and_print(f"{scheduler.upper():<10} | {avg_completed:<15.2f} | {avg_missed:<15.2f} | {success_rate:<15.2f}%")
    
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    test_fixed_taskset(100)
