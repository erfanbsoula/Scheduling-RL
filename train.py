#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/8

from ddpg_torch import *
from LSF import *
from env import *
from EDF import *
if __name__ == '__main__':
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    np.set_printoptions(precision=2, suppress=True)
    alg = DDPG(replay_buffer, STATE_DIM, ACTION_DIM, HIDDEN)
    # alg.load_model(model_path)
    # hyper-parameters
    noise = 3
    frame_idx = 0
    update_frequency = 100
    mean_rewards = []
    mean_edf_rewards = []
    mean_lsf_rewards = []
    rewards = []
    env = Env()
    edf_rewards = []
    lsf_rewards = []

    for i_episode in range(MAX_EPISODES):
        q_loss_list = []
        policy_loss_list = []
        env.reset()
        episode_reward = 0
        edf_episode_reward = 0
        completed = 0
        edf_completed = 0
        lsf_completed = 0
        lsf_episode_reward = 0
        total = 0
        count = 0
        missed = 0
        edf_missed = 0
        lsf_missed = 0
        if EDF:
            env.save()
            for step in range(MAX_STEPS):
                # env.arrive()
                actions = GlobalEDF(env.instance, env.no_processor)
                reward, _, _, info = env.step(actions)
                edf_completed += info[0]
                edf_missed += info[1]
                edf_episode_reward = edf_completed
                if env.done():
                    break
            env.load()
            env.save()
            for step in range(MAX_STEPS):
                # env.arrive()
                # actions = GlobalLSF(env.instance, env.no_processor)
                actions = np.zeros(len(env.instance))
                a = random.sample(range(len(env.instance)), min(env.no_processor, len(env.instance)))
                for i in range(len(env.instance)):
                    if i in a:
                        actions[i] = 1
                reward, _, _, info = env.step(actions)
                lsf_completed += info[0]
                lsf_missed += info[1]
                lsf_episode_reward = lsf_completed
                if env.done():
                    break
            env.load()
        for step in range(MAX_STEPS):
            # env.arrive()
            no_instance = len(env.instance)
            actions = np.ones((no_instance, 1), dtype=np.float32)
            states = np.ones((no_instance, STATE_DIM), dtype=np.float32)
            #if no_instance > env.no_processor:
            for i in range(no_instance):
                state = env.observation(env.instance[i])
                if i_episode > EXPLORATION_EPISODES:
                    action = alg.policy_net.select_action(state, noise)
                else:
                    action = alg.policy_net.sample_action(action_range=1)
                actions[i] = action
                states[i] = state
            noise *= 0.99686

            reward, done, next_state, info = env.step(np.squeeze(actions, 1))
            # info[0]: 执行成功数 info[1]：执行失败数
            completed += info[0]
            missed += info[1]
            if no_instance > 0:
                replay_buffer.push(states, actions, reward, next_state, done)
            episode_reward = completed
            frame_idx += 1

            if len(replay_buffer) > BATCH_SIZE and frame_idx % update_frequency == 0:
                for i in range(5):
                    q_loss, policy_loss = alg.update(BATCH_SIZE)
                    q_loss_list.append(q_loss)
                    policy_loss_list.append(policy_loss)
            if env.done():
                break
        mean_rewards.append(episode_reward*100/(env.no_task*FREQUENCY))
        mean_edf_rewards.append(edf_episode_reward*100/(env.no_task*FREQUENCY))
        mean_lsf_rewards.append(lsf_episode_reward * 100 / (env.no_task * FREQUENCY))
        if i_episode % 2 == 0 and i_episode >= EXPLORATION_EPISODES:
            rewards.append(np.mean(mean_rewards))
            edf_rewards.append(np.mean(mean_edf_rewards))
            lsf_rewards.append(np.mean(mean_lsf_rewards))
            if rewards[-1] > edf_rewards[-1]:
                update_frequency = round(update_frequency + 50)
            mean_edf_rewards = []
            mean_rewards = []
            mean_lsf_rewards = []
            if i_episode % 1 == 0:
                alg.plot(rewards, edf_rewards, lsf_rewards)
                alg.save_model(MODEL_PATH)
        print('Eps: ', i_episode, ' | Update: ', update_frequency, '| Successful: %.2f' % (completed * 100/ (completed + missed)),
              '& %.2f'% (edf_completed * 100/ (edf_completed + edf_missed)),'& %.2f'% (lsf_completed * 100/ (lsf_completed + lsf_missed)),
              ' | Loss: %.2f' % np.average(q_loss_list),'%.2f' % np.average(policy_loss_list),
              '| Completed & Missed:', int(completed), int(missed), '| Utilization: %.2f' % env.utilization())