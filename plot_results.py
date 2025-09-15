import numpy as np
from matplotlib import pyplot as plt
import re
import os


def plot_results(x_axis, x_label, results, save_path):
    """
    Plots the success ratios and energy consumption.

    Args:
        x_axis (list): List of x-axis values (e.g., utilization levels).
        x_label (str): Label for the x-axis.
        results (dict): Dictionary containing success ratios and energy data.
        save_path (str): Path to save the plot.
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))

    x = np.arange(len(x_axis))  # the label locations
    width = 0.25  # the width of the bars

    # Energy Consumption (Bar Plots with reduced opacity)
    ax1.bar(x - width, results['rl']['normalized_energy'], width,
        label='RL Energy (Normalized)', color='tab:red', alpha=0.5)

    ax1.bar(x, results['gedf']['normalized_energy'], width,
        label='GEDF Energy (Normalized)', color='tab:green', alpha=0.5)

    ax1.bar(x + width, results['es-dvfs']['normalized_energy'], width,
        label='ES-DVFS Energy (Normalized)', color='tab:blue', alpha=0.5)

    ax1.set_xlabel(x_label, fontsize=14)
    ax1.set_ylabel('Normalized Total Energy Consumed', fontsize=14)
    ax1.set_xticks(x)
    if isinstance(x_axis[0], float):
        ax1.set_xticklabels([f"{tick:.2f}" for tick in x_axis], fontsize=12)
    else:
        ax1.set_xticklabels([str(tick) for tick in x_axis], fontsize=12)

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
    plt.title('Performance Comparison', fontsize=16)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_overall_result(dir_name):

    dir_path = os.path.join('results', dir_name)

    x_axis = []
    results = {
        'rl': {'success_ratio': [], 'energy': []},
        'gedf': {'success_ratio': [], 'energy': []},
        'es-dvfs': {'success_ratio': [], 'energy': []}
    }

    def read_stat(path):

        with open(path) as file:
            text = file.read()

        pattern = r'^RL\s*\|\s*([^\|\s]*)'
        matching_lines = re.findall(pattern, text, re.MULTILINE)
        results['rl']['success_ratio'].append(float(matching_lines[0]))
        results['rl']['energy'].append(float(matching_lines[1]))

        pattern = r'^GEDF\s*\|\s*([^\|\s]*)'
        matching_lines = re.findall(pattern, text, re.MULTILINE)
        results['gedf']['success_ratio'].append(float(matching_lines[0]))
        results['gedf']['energy'].append(float(matching_lines[1]))

        pattern = r'^ES-DVFS\s*\|\s*([^\|\s]*)'
        matching_lines = re.findall(pattern, text, re.MULTILINE)
        results['es-dvfs']['success_ratio'].append(float(matching_lines[0]))
        results['es-dvfs']['energy'].append(float(matching_lines[1]))

    for sub_dir in sorted(os.listdir(dir_path)):

        if not os.path.isdir(os.path.join(dir_path, sub_dir)):
            continue

        path = f'{dir_path}/{sub_dir}/test_summary.txt'
        x_axis.append(sub_dir.replace('_', '.'))
        read_stat(path)

    max_energy = np.max(results['rl']['energy'] + results['gedf']['energy'] + results['es-dvfs']['energy'])
    results['rl']['normalized_energy'] = [e / max_energy for e in results['rl']['energy']]
    results['gedf']['normalized_energy'] = [e / max_energy for e in results['gedf']['energy']]
    results['es-dvfs']['normalized_energy'] = [e / max_energy for e in results['es-dvfs']['energy']]

    plot_results(x_axis, dir_name, results, f'{dir_path}/experiment_results.png')


if __name__ == '__main__':
    for dir_name in os.listdir('results'):
        plot_overall_result(dir_name)
