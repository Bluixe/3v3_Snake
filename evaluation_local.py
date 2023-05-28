import numpy as np
import torch
import random
import agent.maddpg.submission as maddpg
import agent.greedy.submission as greedy
import agent.greedy_old.submission as greedy_old
import agent.QMIX.submission as QMIX
import agent.mappo.submission as mappo
import agent.mappo_sil.submission as mappo_new
import agent.sp.submission as sp
import agent.pbt.submission as pbt
from env.chooseenv import make
from tabulate import tabulate
import argparse
from torch.distributions import Categorical
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_actions(state, algo, indexs):

    # random agent
    actions = np.random.randint(4, size=3)

    # rl agent
    if algo == 'maddpg':
        obs = maddpg.get_observations(state[0], indexs, obs_dim=122, height=10, width=20)
        logits = maddpg.agent.choose_action(obs)
        logits = torch.Tensor(logits)
        actions = np.array([Categorical(out).sample().item() for out in logits])
    if algo == "greedy":
        actions = greedy.evaluation(state)
    if algo == "QMIX":
        obs = QMIX.get_observations(state[0], indexs, obs_dim=122, height=10, width=20)
        actions = QMIX.agent.choose_action(obs)
    if algo == "greedy_old":
        actions = greedy_old.evaluation(state)
    if algo == "mappo":
        obs = mappo.get_observations(state[0], indexs, obs_dim=122, height=10, width=20)
        logits = mappo.agent.choose_action(obs)
        logits = torch.Tensor(logits)
        actions = np.array([Categorical(out).sample().item() for out in logits])

    if algo == "mappo_sil":
        obs = mappo_new.get_observations(state[0], indexs, obs_dim=122, height=10, width=20)
        logits = mappo_new.agent.choose_action(obs)
        logits = torch.Tensor(logits)
        actions = np.array([Categorical(out).sample().item() for out in logits])
    if algo == "sp":
        obs = sp.get_observations(state[0], indexs, obs_dim=122, height=10, width=20)
        logits = sp.agent.choose_action(obs)
        logits = torch.Tensor(logits)
        actions = np.array([Categorical(out).sample().item() for out in logits])
    if algo == 'pbt':
        obs = pbt.get_observations(state[0], indexs, obs_dim=122, height=10, width=20)
        logits = pbt.agent.choose_action(obs)
        logits = torch.Tensor(logits)
        actions = np.array([Categorical(out).sample().item() for out in logits])
    return actions


def get_join_actions(obs, algo_list):
    obs_2_evaluation = obs.copy()
    indexs = [0,1,2,3,4,5]
    first_action = get_actions(obs_2_evaluation[:3], algo_list[0], indexs[:3])
    second_action = get_actions(obs_2_evaluation[3:], algo_list[1], indexs[3:])
    actions = np.zeros(6)
    actions[:3] = first_action[:]
    actions[3:] = second_action[:]
    return actions


def run_game(env, algo_list, episode, verbose=False):

    total_reward = np.zeros(6)
    num_win = np.zeros(3)

    for i in range(1, episode + 1):
        episode_reward = np.zeros(6)

        state = env.reset()

        step = 0

        # QMIX.agent.init_hidden(1)

        while True:
            joint_action = get_join_actions(state, algo_list)

            next_state, reward, done, _, info = env.step(env.encode(joint_action))
            reward = np.array(reward)
            episode_reward += reward

            if done:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    num_win[0] += 1
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
                    num_win[1] += 1
                else:
                    num_win[2] += 1

                if not verbose:
                    # print('.', end='')
                    print(i)
                    if i % 100 == 0 or i == episode:
                        print()
                break

            state = next_state
            step += 1
        print("episode_reward: ", episode_reward)
        total_reward += episode_reward

    # calculate results
    total_reward /= episode
    print("total_reward: ", total_reward)
    print(f'\nResult base on {episode} ', end='')
    print('episode:') if episode == 1 else print('episodes:')

    header = ['Name', algo_list[0], algo_list[1]]
    data = [['score', np.round(np.sum(total_reward[:3]), 2), np.round(np.sum(total_reward[3:]), 2)],
            ['win', num_win[0], num_win[1]]]
    print(tabulate(data, headers=header, tablefmt='pretty'))


if __name__ == "__main__":
    env_type = 'snakes_3v3'

    game = make(env_type, conf=None)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--my_ai", default="rl", help="rl/random")
    parser.add_argument("--opponent", default="random", help="rl/random")
    parser.add_argument("--episode", default=100)
    args = parser.parse_args()

    agent_list = [args.my_ai, args.opponent]
    run_game(game, algo_list=agent_list, episode=args.episode, verbose=False)
