import os
import numpy as np
from config import *
from env import Environment
from ddpg_torch import ReplayBuffer, MADDPG
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)


environment = Environment()
replay_buffer = ReplayBuffer(BUFFER_SIZE)

algorithm = MADDPG(
    replay_buffer, DISCOUNT_RATE, STATE_DIM, ACTION_DIM, HIDDEN_DIM,
    Q_LEARNING_RATE, POLICY_LEARNING_RATE, TARGET_UPDATE_DELAY
)

noise_scale = 0.0
noise_decay = 0.998

rewards_log = []

for i_episode in range(1, MAX_EPISODES+1):

    environment.reset()
    next_state = environment.get_state() 

    print(f"--- Episode {i_episode} ---")
    util = environment.calc_utilization()
    print(f"Utilization: {util if util is not None else 'N/A':.4f}")

    q_loss_list = []
    policy_loss_list = []
    episode_reward_sum = 0
    total_completed_in_episode = 0
    total_missed_in_episode = 0

    for step in range(MAX_STEPS):

        current_state = next_state
        num_active_instances = len(environment.active_instances)
        noise_scale *= noise_decay

        if num_active_instances > 0 and i_episode > EXPLORATION_EPISODES:
            action = algorithm.policy_net.select_action(current_state, noise_std=noise_scale)
        else:
            action = np.random.uniform(low=0.0, high=1.0, size=(num_active_instances, ACTION_DIM))
            action = action.astype(np.float32)

        scheduling_priorities = action[:, 0]
        frequency_scales = action[:, 1]

        if frequency_scales.size > 0:
            num_levels = len(DVFS_LEVELS)
            level_indices = np.floor(action[:, 1] * num_levels).astype(int)
            level_indices = np.clip(level_indices, 0, num_levels - 1)
            frequency_scales = np.array([DVFS_LEVELS[i] for i in level_indices]).astype(np.float32)

        transition = environment.step(scheduling_priorities, frequency_scales)
        global_reward, next_state, is_done, num_completed, num_missed = transition

        if num_active_instances > 0:
             replay_buffer.push(current_state, action, global_reward, next_state, is_done)

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

    total_tasks_in_episode = environment.task_count * INSTANCES_PER_TASK
    success_ratio = total_completed_in_episode / total_tasks_in_episode * 100
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

    if i_episode % CHECKPOINT_INTERVAL == 0:
        print(f"Saving model at episode {i_episode}...")
        algorithm.save_model(os.path.join(MODEL_PATH, f"ep_{i_episode}"))


print("Training finished.")

plt.plot(rewards_log)
plt.title("Episode Reward Trend")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("plots/episode_rewards.png", dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved to plots/episode_rewards.png")