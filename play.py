import matplotlib.pyplot as plt
import os

# Data from the results
utilization_levels = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40]
rl_success = [94.92, 94.65, 95.13, 93.02, 93.92, 91.08, 88.71, 86.08, 85.83, 83.88]
gedf_success = [95.79, 95.12, 96.00, 94.04, 95.04, 91.08, 88.71, 86.12, 85.88, 83.88]
rl_energy = [19230.40, 21459.60, 28585.40, 35684.80, 39373.20, 36865.40, 42386.00, 45074.40, 54663.80, 51621.80]
gedf_energy = [21741.40, 25469.60, 30585.40, 36086.60, 39376.00, 36872.60, 42393.40, 45083.60, 54672.00, 51625.20]

# Normalize energy values
max_energy = max(max(rl_energy), max(gedf_energy))
rl_energy = [e / max_energy for e in rl_energy]
gedf_energy = [e / max_energy for e in gedf_energy]

# Plotting results
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('System Utilization (Load)', fontsize=14)
ax1.set_ylabel('Success Ratio (%)', color=color, fontsize=14)
ax1.plot(
    utilization_levels, rl_success,
    marker='o', linestyle='-', color=color, label='RL Success Ratio'
)
ax1.plot(
    utilization_levels, gedf_success,
    marker='x', linestyle='--', color=color, alpha=0.7, label='GEDF Success Ratio'
)
ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.grid(True, linestyle=':', alpha=0.7)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Normalized Total Energy Consumed', color=color, fontsize=14)
ax2.plot(
    utilization_levels, rl_energy,
    marker='o', linestyle='-', color=color, label='RL Energy'
)
ax2.plot(
    utilization_levels, gedf_energy,
    marker='x', linestyle='--', color=color, alpha=0.7, label='GEDF Energy'
)
ax2.tick_params(axis='y', labelcolor=color, labelsize=12)

fig.tight_layout()
plt.title('Performance Comparison: RL Agent vs GEDF', fontsize=16)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower right', fontsize=12)

os.makedirs('plots', exist_ok=True)
plt.savefig('plots/performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nPlot saved to plots/performance_comparison.png")