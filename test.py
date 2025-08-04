import os
import numpy as np
import matplotlib.pyplot as plt
from config import *
from env import Environment, Task
from ddpg_torch import MADDPG
from task_gen import StaffordRandFixedSum, gen_periods

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
    scheduling_priorities = np.zeros(num_active_instances, dtype=float)
    frequency_scales = np.full(num_active_instances, DVFS_LEVELS[-1], dtype=np.float32)

    if num_active_instances == 0:
        return scheduling_priorities, frequency_scales

    for i in range(num_active_instances):
        scheduling_priorities[i] = 1 / active_instances[i].deadline

    return scheduling_priorities, frequency_scales


def es_dvfs_scheduler(active_instances: list):
    """
    Schedules tasks based on the ES-DVFS algorithm for a single core.
    Calculates speed based on workload and intensity, and schedules by EDF.

    Args:
        active_instances (list): A list of active Instance objects.

    Returns:
        tuple: (scheduling_priorities, frequency_scales)
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
    speed = min(DVFS_LEVELS, key=lambda x: abs(x - speed))

    scheduling_priorities = np.zeros(num_active_instances, dtype=float)
    for i in range(num_active_instances):
        scheduling_priorities[i] = 1 / (active_instances[i].deadline + 1e-6)

    frequency_scales = np.full(num_active_instances, speed, dtype=np.float32)

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

        elif scheduler_type == 'es-dvfs':
            scheduling_priorities, frequency_scales = es_dvfs_scheduler(environment.active_instances)

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


def plot_results(utilization_levels, results, save_path):
    """
    Plots the success ratios and energy consumption.

    Args:
        utilization_levels (list): List of utilization levels.
        results (dict): Dictionary containing success ratios and energy data.
        save_path (str): Path to save the plot.
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))

    x = np.arange(len(utilization_levels))  # the label locations
    width = 0.25  # the width of the bars

    # Energy Consumption (Bar Plots with reduced opacity)
    ax1.bar(x - width, results['rl']['normalized_energy'], width,
        label='RL Energy (Normalized)', color='tab:red', alpha=0.5)

    ax1.bar(x, results['gedf']['normalized_energy'], width,
        label='GEDF Energy (Normalized)', color='tab:green', alpha=0.5)

    ax1.bar(x + width, results['es-dvfs']['normalized_energy'], width,
        label='ES-DVFS Energy (Normalized)', color='tab:blue', alpha=0.5)

    ax1.set_xlabel('System Utilization (Load)', fontsize=14)
    ax1.set_ylabel('Normalized Total Energy Consumed', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{util:.2f}" for util in utilization_levels], fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Success Ratios (Line Plots)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Success Ratio (%)', fontsize=14)
    ax2.plot(
        x, results['rl']['success_ratio'],
        marker='o', linestyle='-', color='tab:red', label='RL Success Ratio'
    )
    ax2.plot(
        x, results['gedf']['success_ratio'],
        marker='x', linestyle='-', color='tab:green', label='GEDF Success Ratio'
    )
    ax2.plot(
        x, results['es-dvfs']['success_ratio'],
        marker='s', linestyle='-', color='tab:blue', label='ES-DVFS Success Ratio'
    )
    ax2.tick_params(axis='y', labelsize=12)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best', fontsize=12)

    fig.tight_layout()
    plt.title('Performance Comparison: RL Agent vs GEDF vs ES-DVFS', fontsize=16)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {save_path}")


def save_summary(utilization_levels, results, summary_path):
    """
    Prints and saves the summary of results to a file.

    Args:
        utilization_levels (list): List of utilization levels.
        results (dict): Dictionary containing success ratios and energy data.
        summary_path (str): Path to save the summary.
    """
    with open(summary_path, 'w') as f:

        def write_and_print(line):
            print(line)
            f.write(line + "\n")

        write_and_print("\n--- Success Ratio Summary ---")
        write_and_print(
            f"{'Utilization':<15} | "
            f"{'RL Success (%)':<20} | {'GEDF Success (%)':<20} | {'ES-DVFS Success (%)':<20}"
        )
        write_and_print("-" * 85)
        for i, util in enumerate(utilization_levels):
            write_and_print(
                f"{util:<15.2f} | "
                f"{results['rl']['success_ratio'][i]:<20.2f} | "
                f"{results['gedf']['success_ratio'][i]:<20.2f} | "
                f"{results['es-dvfs']['success_ratio'][i]:<20.2f}"
            )

        write_and_print("\n--- Energy Consumption Summary ---")
        write_and_print(
            f"{'Utilization':<15} | "
            f"{'RL Energy':<15} | {'GEDF Energy':<15} | {'ES-DVFS Energy':<15}"
        )
        write_and_print("-" * 70)
        for i, util in enumerate(utilization_levels):
            write_and_print(
                f"{util:<15.2f} | "
                f"{results['rl']['energy'][i]:<15.2f} | "
                f"{results['gedf']['energy'][i]:<15.2f} | "
                f"{results['es-dvfs']['energy'][i]:<15.2f}"
            )

    print(f"\nSummary saved to {summary_path}")


def main():

    print("Starting performance comparison...")
    environment = Environment()
    environment.reset()

    rl_agent = MADDPG(
        None, DISCOUNT_RATE, STATE_DIM, ACTION_DIM, HIDDEN_DIM,
        Q_LEARNING_RATE, POLICY_LEARNING_RATE, TARGET_UPDATE_DELAY
    )

    model_files = [d for d in os.listdir(SAVE_PATH) if d.startswith('ep_')]
    if not model_files:
        print(f"No saved models found in {SAVE_PATH}. Please train the RL agent first.")
        return 1

    model_files.sort(key=lambda x: int(x.split('_')[1]), reverse=True)
    latest_model_path = os.path.join(SAVE_PATH, model_files[0])
    print(f"Loading RL agent model from: {latest_model_path}")
    rl_agent.load_model(latest_model_path)

    utilization_levels = list(np.arange(0.5, 1.5, 0.1))

    results = {
        'rl': {'success_ratio': [], 'energy': []},
        'gedf': {'success_ratio': [], 'energy': []},
        'es-dvfs': {'success_ratio': [], 'energy': []}
    }

    num_runs_per_utilization = 5

    for util in utilization_levels:

        print(f"\n--- Testing Utilization: {util:.2f} ---")
        target_system_utilization = util * environment.processor_count

        rl_success_temp, rl_energy_temp = [], []
        gedf_success_temp, gedf_energy_temp = [], []
        es_dvfs_success_temp, es_dvfs_energy_temp = [], []

        for i in range(num_runs_per_utilization):

            print(f"  Run {i+1}/{num_runs_per_utilization}")
            utilizations = StaffordRandFixedSum(environment.task_count, target_system_utilization, 1).flatten()
            periods = gen_periods(environment.task_count, 1, MIN_PERIOD, MAX_PERIOD, 1.0, "logunif").flatten()
            periods = periods.round().astype(int) # periods are integers for now

            task_set_for_run = []
            for utilization, period in zip(utilizations, periods):
                task_set_for_run.append(Task(utilization, period))

            environment.reset()
            environment.task_set = task_set_for_run
            environment.active_instances = []
            environment.arrive_instances()
            environment.update_env_stats()
            print("Debug: actual utilization:", environment.calc_utilization())

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

            for task in task_set_for_run:
                task.instance_count = 0

            environment.reset()
            environment.task_set = task_set_for_run
            environment.active_instances = []
            environment.arrive_instances()
            environment.update_env_stats()

            success_es_dvfs, energy_es_dvfs = run_simulation(environment, 'es-dvfs')
            es_dvfs_success_temp.append(success_es_dvfs)
            es_dvfs_energy_temp.append(energy_es_dvfs)

        avg_rl_success = np.mean(rl_success_temp)
        avg_rl_energy = np.mean(rl_energy_temp)
        avg_gedf_success = np.mean(gedf_success_temp)
        avg_gedf_energy = np.mean(gedf_energy_temp)
        avg_es_dvfs_success = np.mean(es_dvfs_success_temp)
        avg_es_dvfs_energy = np.mean(es_dvfs_energy_temp)

        results['rl']['success_ratio'].append(avg_rl_success)
        results['rl']['energy'].append(avg_rl_energy)
        results['gedf']['success_ratio'].append(avg_gedf_success)
        results['gedf']['energy'].append(avg_gedf_energy)
        results['es-dvfs']['success_ratio'].append(avg_es_dvfs_success)
        results['es-dvfs']['energy'].append(avg_es_dvfs_energy)

        print(f"  Avg RL   - Success: {avg_rl_success:.2f}%, Energy: {avg_rl_energy:.2f}")
        print(f"  Avg GEDF - Success: {avg_gedf_success:.2f}%, Energy: {avg_gedf_energy:.2f}")
        print(f"  Avg ES-DVFS - Success: {avg_es_dvfs_success:.2f}%, Energy: {avg_es_dvfs_energy:.2f}")

    # Normalize energy values for plotting
    max_energy = np.max(results['rl']['energy'] + results['gedf']['energy'] + results['es-dvfs']['energy'])
    results['rl']['normalized_energy'] = [e / max_energy for e in results['rl']['energy']]
    results['gedf']['normalized_energy'] = [e / max_energy for e in results['gedf']['energy']]
    results['es-dvfs']['normalized_energy'] = [e / max_energy for e in results['es-dvfs']['energy']]

    # Plotting results
    plot_path = os.path.join(SAVE_PATH, 'test_result.png')
    plot_results(utilization_levels, results, plot_path)

    # Save summary to file
    summary_path = os.path.join(SAVE_PATH, 'test_summary.txt')
    save_summary(utilization_levels, results, summary_path)


if __name__ == '__main__':
    main()
