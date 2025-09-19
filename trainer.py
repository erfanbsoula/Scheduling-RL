import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from environment import Environment
from ddpg_torch import ReplayBuffer, MADDPG
from config import (
    SAVE_PATH,
    RANDOM_SEED,
    ACTION_DIM,
    BUFFER_SIZE,
    MAX_EPISODES,
    MAX_STEPS,
    UPDATE_INTERVAL,
    UPDATE_REPEAT_COUNT,
    BATCH_SIZE,
    CHECKPOINT_INTERVAL,
    START_NOISE_WIDTH,
    END_NOISE_WIDTH,
)


class Trainer(object):

    def __init__(self):

        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        np.set_printoptions(precision=4, suppress=True)

        os.makedirs(SAVE_PATH, exist_ok=True)

        self.curr_noise_width = START_NOISE_WIDTH
        self.noise_decay = (END_NOISE_WIDTH / START_NOISE_WIDTH) ** (1 / MAX_EPISODES)

        self.environment = Environment()
        self.environment.generate_new_taskset()
        self.environment.save_task_set(
            os.path.join(SAVE_PATH, "taskset.pkl")
        )
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.algorithm = MADDPG(self.replay_buffer)

        self.rewards_log = []
        self.success_ratio_log = []
        self.energy_consumption_log = []


    def train_episode(self):

        self.environment.reset()
        next_state = self.environment.get_state()
        self.curr_noise_width *= self.noise_decay

        q_loss_list = []
        policy_loss_list = []
        episode_reward_sum = 0
        total_completed_in_episode = 0
        total_missed_in_episode = 0

        for step in range(MAX_STEPS):
            current_state_actor, current_state_critic = next_state
            num_active_instances = len(self.environment.active_instances)

            if num_active_instances > 0:
                action = self.algorithm.policy_net.select_action(
                    current_state_actor, noise_width=self.curr_noise_width)
            else:
                action = np.zeros((1, ACTION_DIM), dtype=np.float32)

            scheduling_priorities = action[:, 0]
            frequency_scales = action[:, 1]

            transition = self.environment.step(scheduling_priorities, frequency_scales)
            global_reward, next_state, is_done, num_completed, num_missed, time_duration = transition
            next_state_actor, next_state_critic = next_state

            self.replay_buffer.push(
                current_state_actor, current_state_critic, action, global_reward,
                next_state_actor, next_state_critic, is_done, time_duration
            )

            episode_reward_sum += global_reward
            total_completed_in_episode += num_completed
            total_missed_in_episode += num_missed

            if len(self.replay_buffer) > BATCH_SIZE and (step + 1) % UPDATE_INTERVAL == 0:
                for _ in range(UPDATE_REPEAT_COUNT):
                    q_loss, policy_loss = self.algorithm.update()
                    q_loss_list.append(q_loss)
                    policy_loss_list.append(policy_loss)

            if is_done: break
        
        total_tasks_in_episode = total_completed_in_episode + total_missed_in_episode
        success_ratio = total_completed_in_episode / total_tasks_in_episode * 100

        self.rewards_log.append(episode_reward_sum)
        self.success_ratio_log.append(success_ratio)
        self.energy_consumption_log.append(self.environment.total_energy_consumed)

        avg_q_loss = np.mean(q_loss_list) if q_loss_list else 0
        avg_policy_loss = np.mean(policy_loss_list) if policy_loss_list else 0

        return {
            "episode_reward": episode_reward_sum,
            "success_ratio": success_ratio,
            "total_completed": total_completed_in_episode,
            "total_missed": total_missed_in_episode,
            "total_tasks": total_tasks_in_episode,
            "total_energy": self.environment.total_energy_consumed,
            "avg_q_loss": avg_q_loss,
            "avg_policy_loss": avg_policy_loss,
            "steps": step + 1
        }
    

    def plot_train_metrics(self):

        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(self.rewards_log)
        plt.title("Episode Reward Trend")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        plt.subplot(3, 1, 2)
        plt.plot(self.success_ratio_log)
        plt.title("Success Ratio Trend")
        plt.xlabel("Episode")
        plt.ylabel("Success Ratio (%)")

        plt.subplot(3, 1, 3)
        plt.plot(self.energy_consumption_log)
        plt.title("Energy Consumption Trend")
        plt.xlabel("Episode")
        plt.ylabel("Energy")

        plt.tight_layout()
        plot_path = os.path.join(SAVE_PATH, "training_metrics.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()


    def run(self):

        for i_episode in range(1, MAX_EPISODES+1):

            print(f"--- Episode {i_episode} ---")
            stats = self.train_episode()

            print(f"Episode Reward: {stats['episode_reward']}")
            print(f"Episode Success Ratio: {stats['success_ratio']:.2f}%")
            print(f"Total Completed: {stats['total_completed']}/{stats['total_tasks']}")
            print(f"Total Missed: {stats['total_missed']}/{stats['total_tasks']}")
            print(f"Total Energy Consumed: {stats['total_energy']:.2f}")
            print(f"Avg Q_Loss: {stats['avg_q_loss']:.4f}")
            print(f"Avg Policy_Loss: {stats['avg_policy_loss']:.4f}")
            print(f"Noise Scale: {self.curr_noise_width:.4f}")
            print(f"Replay Buffer Size: {len(self.replay_buffer)}")
            print(f"Total Steps: {stats['steps']}")

            if i_episode % CHECKPOINT_INTERVAL == 0 or i_episode == MAX_EPISODES:

                self.plot_train_metrics()

                model_path = os.path.join(SAVE_PATH, f"ep_{i_episode}")
                self.algorithm.save_model(model_path)
                print(f"Models saved to {model_path}")

                np.savez(
                    os.path.join(SAVE_PATH, "training_metrics.npz"),
                    rewards=np.array(self.rewards_log),
                    success_ratios=np.array(self.success_ratio_log),
                    energy_consumption=np.array(self.energy_consumption_log)
                )

        print("Training finished.\n")


if __name__ == "__main__":

    trainer = Trainer()
    trainer.run()