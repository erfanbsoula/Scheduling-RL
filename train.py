import numpy as np
from config import *
from env import Env
from ddpg_torch import ReplayBuffer, DDPG


def GlobalEDF(instance, no_processor):
    deadline = []
    action = np.zeros(len(instance))
    for i in instance:
        deadline.append(i.deadline)
    executable = np.argsort(deadline)
    for i in range(no_processor):
        if i < len(executable):
            action[executable[i]] = 1
    return action


def GlobalLSF(instance, no_processor):
    laxity = []
    action = np.zeros(len(instance))
    for i in instance:
        laxity.append(i.laxity_time)
    executable = np.argsort(laxity)
    for i in range(no_processor):
        if i < len(executable):
            action[executable[i]] = 1
    return action


if __name__ == '__main__':

    np.set_printoptions(precision=2, suppress=True)

    env = Env()
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    alg = DDPG(replay_buffer, STATE_DIM, ACTION_DIM, HIDDEN)

    # hyper-parameters
    noise = 3
    frame_idx = 0
    update_frequency = 100

    rewards = []
    mean_rewards = []

    for i_episode in range(MAX_EPISODES):

        print(f"episode {i_episode} ...")

        env.reset()
        q_loss_list = []
        policy_loss_list = []

        episode_reward = 0
        completed = 0
        missed = 0

        for step in range(MAX_STEPS):

            frame_idx += 1
            noise *= 0.99686
            instance_count = len(env.instance)
            states = np.ones((instance_count, STATE_DIM), dtype=np.float32)
            actions = np.ones((instance_count, 1), dtype=np.float32)

            for i in range(instance_count):
                state = env.observation(env.instance[i])
                if i_episode > EXPLORATION_EPISODES:
                    action = alg.policy_net.select_action(state, noise)
                else:
                    action = alg.policy_net.sample_action()

                states[i] = state
                actions[i] = action

            reward, done, next_state, info = env.step(np.squeeze(actions, 1))
            if instance_count > 0:
                replay_buffer.push(states, actions, reward, next_state, done)

            completed += info[0]
            missed += info[1]

            if len(replay_buffer) > BATCH_SIZE and frame_idx % update_frequency == 0:
                for i in range(5):
                    q_loss, policy_loss = alg.update(BATCH_SIZE)
                    q_loss_list.append(q_loss)
                    policy_loss_list.append(policy_loss)

            if env.done():
                break

        episode_reward = completed
        mean_rewards.append(episode_reward * 100 / (env.no_task * FREQUENCY))

        if i_episode % 2 == 0 and i_episode >= EXPLORATION_EPISODES:
            rewards.append(np.mean(mean_rewards))
            mean_rewards = []
            alg.plot(rewards, [], [])
            alg.save_model(MODEL_PATH)

        # print('Eps: ', i_episode, ' | Update: ', update_frequency, '| Successful: %.2f' % (completed * 100/ (completed + missed)),
        #       '& %.2f'% (edf_completed * 100/ (edf_completed + edf_missed)),'& %.2f'% (lsf_completed * 100/ (lsf_completed + lsf_missed)),
        #       ' | Loss: %.2f' % np.average(q_loss_list),'%.2f' % np.average(policy_loss_list),
        #       '| Completed & Missed:', int(completed), int(missed), '| Utilization: %.2f' % env.utilization())