import os
import numpy as np
import matplotlib.pyplot as plt
from config import *
from env import Environment, Task
from ddpg_torch import MADDPG

np.set_printoptions(precision=4, suppress=True)


def gedf_scheduler(active_instances: list):
    """
    Schedules tasks based on Global Earliest Deadline First (GEDF).
    Always uses maximum frequency for scheduled tasks.

    Args:
        active_instances (list): A list of active Instance objects.

    Returns:
        tuple: (scheduling_priorities, frequency_scales)
            scheduling_priorities (np.ndarray): Array of scheduling priorities for each instance (higher is better).
            frequency_scales (np.ndarray): Array of frequency levels for each instance (max frequency).
    """
    num_active_instances = len(active_instances)
    scheduling_priorities = np.zeros(num_active_instances, dtype=int)
    frequency_scales = np.full(num_active_instances, DVFS_LEVELS[-1], dtype=np.float32)

    if num_active_instances == 0:
        return scheduling_priorities, frequency_scales

    for i in range(num_active_instances):
        scheduling_priorities[i] = 1 / active_instances[i].deadline

    return scheduling_priorities, frequency_scales


def run_simulation(
        environment: Environment,
        scheduler_type: str,
        rl_agent: MADDPG = None,
    ):
    """
    Runs a simulation for a given scheduler type and utilization level.

    Args:
        environment (Environment): The simulation environment.
        scheduler_type (str): 'rl' or 'gedf'.
        rl_agent (MADDPG, optional): The RL agent, required if scheduler_type is 'rl'.
        utilization_level (float, optional): The target utilization level for the environment reset.

    Returns:
        tuple: (success_ratio, total_energy_consumed)
    """
    episode_reward_sum = 0
    total_completed_in_episode = 0
    total_missed_in_episode = 0

    next_state = environment.get_state()

    for step in range(MAX_STEPS):

        current_state = next_state
        num_active_instances = len(environment.active_instances)

        if scheduler_type == 'rl':

            if rl_agent is None:
                raise ValueError("RL agent must be provided for 'rl' scheduler type.")

            if num_active_instances > 0:
                action = rl_agent.target_policy_net.select_action(current_state, noise_std=0.0)
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
            if num_active_instances > 0:
                scheduling_priorities, frequency_scales = gedf_scheduler(environment.active_instances)
            else:
                scheduling_priorities = np.array([])
                frequency_scales = np.array([])

        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


        transition = environment.step(scheduling_priorities, frequency_scales)
        global_reward, next_state, is_done, num_completed, num_missed = transition

        episode_reward_sum += global_reward
        total_completed_in_episode += num_completed
        total_missed_in_episode += num_missed

        if is_done:
            break

    total_tasks_in_episode = environment.task_count * INSTANCES_PER_TASK
    success_ratio = (total_completed_in_episode / total_tasks_in_episode * 100)

    return success_ratio, environment.total_energy_consumed


def main():

    print("Starting performance comparison...")
    environment = Environment()

    rl_agent = MADDPG(
        None, DISCOUNT_RATE, STATE_DIM, ACTION_DIM, HIDDEN_DIM,
        Q_LEARNING_RATE, POLICY_LEARNING_RATE, TARGET_UPDATE_DELAY
    )

    model_files = [d for d in os.listdir(MODEL_PATH) if d.startswith('ep_')]
    if not model_files:
        print(f"No saved models found in {MODEL_PATH}. Please train the RL agent first.")
        return 1

    model_files.sort(key=lambda x: int(x.split('_')[1]), reverse=True)
    latest_model_path = os.path.join(MODEL_PATH, model_files[0])
    print(f"Loading RL agent model from: {latest_model_path}")
    rl_agent.load_model(latest_model_path)

    utilization_levels = list(np.arange(0.5, 1.5, 0.1))

    results = {
        'rl': {'success_ratio': [], 'energy': []},
        'gedf': {'success_ratio': [], 'energy': []}
    }

    num_runs_per_utilization = 5

    for util in utilization_levels:

        print(f"\n--- Testing Utilization: {util:.2f} ---")
        
        rl_success_temp = []
        rl_energy_temp = []
        gedf_success_temp = []
        gedf_energy_temp = []

        for i in range(num_runs_per_utilization):

            print(f"  Run {i+1}/{num_runs_per_utilization}")
            task_set_for_run = [Task(util) for _ in range(environment.task_count)]
            environment.reset()
            environment.task_set = task_set_for_run
            environment.active_instances = []
            environment.arrive_instances()
            environment.update_env_stats()

            success_rl, energy_rl = run_simulation(environment, 'rl', rl_agent=rl_agent)
            rl_success_temp.append(success_rl)
            rl_energy_temp.append(energy_rl)

            for task in task_set_for_run:
                task.instance_count = 0

            environment.reset()
            environment.task_set = task_set_for_run
            environment.active_instances = []
            environment.arrive_instances()
            environment.update_env_stats()

            success_gedf, energy_gedf = run_simulation(environment, 'gedf')
            gedf_success_temp.append(success_gedf)
            gedf_energy_temp.append(energy_gedf)

        avg_rl_success = np.mean(rl_success_temp)
        avg_rl_energy = np.mean(rl_energy_temp)
        avg_gedf_success = np.mean(gedf_success_temp)
        avg_gedf_energy = np.mean(gedf_energy_temp)

        results['rl']['success_ratio'].append(avg_rl_success)
        results['rl']['energy'].append(avg_rl_energy)
        results['gedf']['success_ratio'].append(avg_gedf_success)
        results['gedf']['energy'].append(avg_gedf_energy)

        print(f"  Avg RL   - Success: {avg_rl_success:.2f}%, Energy: {avg_rl_energy:.2f}")
        print(f"  Avg GEDF - Success: {avg_gedf_success:.2f}%, Energy: {avg_gedf_energy:.2f}")

    # Normalize energy values for plotting
    max_energy = np.max(results['rl']['energy'] + results['gedf']['energy'])
    results['rl']['normalized_energy'] = [e / max_energy for e in results['rl']['energy']]
    results['gedf']['normalized_energy'] = [e / max_energy for e in results['gedf']['energy']]

    # Plotting results
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('System Utilization (Load)', fontsize=14)
    ax1.set_ylabel('Success Ratio (%)', color=color, fontsize=14)
    ax1.plot(
        utilization_levels, results['rl']['success_ratio'],
        marker='o', linestyle='-', color=color, label='RL Success Ratio'
    )
    ax1.plot(
        utilization_levels, results['gedf']['success_ratio'],
        marker='x', linestyle='--', color=color, alpha=0.7, label='GEDF Success Ratio'
    )
    ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.grid(True, linestyle=':', alpha=0.7)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Normalized Total Energy Consumed', color=color, fontsize=14)
    ax2.plot(
        utilization_levels, results['rl']['normalized_energy'],
        marker='o', linestyle='-', color=color, label='RL Energy (Normalized)'
    )
    ax2.plot(
        utilization_levels, results['gedf']['normalized_energy'],
        marker='x', linestyle='--', color=color, alpha=0.7, label='GEDF Energy (Normalized)'
    )
    ax2.tick_params(axis='y', labelcolor=color, labelsize=12)

    fig.tight_layout()
    plt.title('Performance Comparison: RL Agent vs GEDF', fontsize=16)
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best', fontsize=12)

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nPlot saved to plots/performance_comparison.png")

    print("\n--- Summary of Results ---")
    print(
        f"{'Utilization':<15} | {'RL Success (%)':<18} | "
        f"{'GEDF Success (%)':<18} | {'RL Energy':<15} | "
        f"{'GEDF Energy':<15}"
    )
    print("-" * 90)
    for i, util in enumerate(utilization_levels):
        print(
            f"{util:<15.2f} | "
            f"{results['rl']['success_ratio'][i]:<18.2f} | "
            f"{results['gedf']['success_ratio'][i]:<18.2f} | "
            f"{results['rl']['energy'][i]:<15.2f} | "
            f"{results['gedf']['energy'][i]:<15.2f}"
        )


if __name__ == '__main__':
    main()
