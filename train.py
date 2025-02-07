import numpy as np
from config import *
from env import Environment
from ddpg_torch import ReplayBuffer, DDPG

np.set_printoptions(precision=2, suppress=True)



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



env = Environment()
replay_buffer = ReplayBuffer(BUFFER_SIZE)
alg = DDPG(replay_buffer, STATE_DIM, ACTION_DIM, HIDDEN)

noise = 0

rewards = []
mean_rewards = []

for i_episode in range(MAX_EPISODES):

    env.reset()
    next_state = env.get_state()
    print(f"episode {i_episode} ...")
    print("utilization:", env.calc_utilization())

    q_loss_list = []
    policy_loss_list = []
    episode_reward = 0
    total_completed = 0
    total_missed = 0

    for step in range(MAX_STEPS):

        noise *= 0.99686
        instance_count = len(env.active_instances)
        state = next_state

        if instance_count > 0 and i_episode > EXPLORATION_EPISODES:
            action = alg.policy_net.select_action(state, noise/2)
        else:
            normal = np.random.normal(loc=0.5, scale=1.0, size=(instance_count,1))
            action = np.clip(normal, 0.001, 1).astype(np.float32)

        reward, next_state, done, completed, missed = env.step(np.squeeze(action, 1))
        if instance_count > 0:
            replay_buffer.push(state, action, reward, next_state, done)

        total_completed += completed
        total_missed += missed

        if len(replay_buffer) > BATCH_SIZE and (step + 1) % UPDATE_INTERVAL == 0:
            for i in range(5):
                q_loss, policy_loss = alg.update(BATCH_SIZE)
                q_loss_list.append(q_loss)
                policy_loss_list.append(policy_loss)

        if env.done():
            break

    episode_reward = total_completed
    rewards.append(episode_reward * 100 / (env.task_count * INSTANCES_PER_TASK))
    print("episode success ratio:", rewards[i_episode])

    # if i_episode % 2 == 0 and i_episode >= EXPLORATION_EPISODES:
    #     rewards.append(np.mean(mean_rewards))
    #     mean_rewards = []
    #     alg.plot(rewards, [], [])
    #     alg.save_model(MODEL_PATH)

    # print('Eps: ', i_episode, ' | Update: ', update_frequency, '| Successful: %.2f' % (completed * 100/ (completed + missed)),
    #       '& %.2f'% (edf_completed * 100/ (edf_completed + edf_missed)),'& %.2f'% (lsf_completed * 100/ (lsf_completed + lsf_missed)),
    #       ' | Loss: %.2f' % np.average(q_loss_list),'%.2f' % np.average(policy_loss_list),
    #       '| Completed & Missed:', int(completed), int(missed), '| Utilization: %.2f' % env.utilization())