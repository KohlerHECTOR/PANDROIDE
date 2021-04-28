import os
# import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import lzma
import gym
from chrono import Chrono
from simu import make_simu_from_params
from policies import GenericNet,BernoulliPolicy, NormalPolicy, SquashedGaussianPolicy, DiscretePolicy, PolicyWrapper
from arguments import get_args
from numpy.random import random

def evaluate_policy(params, env, weights):
    policy = NormalPolicy(env.observation_space.shape[0], 24, 36, 1, params.lr_actor)
    policy.set_weights(weights)
    scores=np.zeros(int(args.nb_evals))
    for j in range(int(args.nb_evals)):
        state = env.reset()
        total_reward = 0
        for t in range(params.max_episode_steps):
            action = policy.select_action(state, params.deterministic_eval)
                # print("action", action)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

            if done:
                scores[j]=total_reward
                break
    return scores
def load_policies(folder):
    """
     :param: folder : name of the folder containing policies
     Output : none (policies of the folder stored in self.env_dict)
     """
    listdir = os.listdir(folder)
    policies = []
    for policy_file in listdir:
        pw = PolicyWrapper(GenericNet(),0, "", "", "", 0)
        policy,_ = pw.load(directory+policy_file)
        policy = policy.get_weights()
        policies.append(policy)
    return policies

if __name__ == '__main__':
    args = get_args()
    print(args)
    directory = os.getcwd() + '/Models/'
    all_weights = load_policies(directory)
    distances = np.zeros(len(all_weights)-1)
    for i in range(len(distances)):

        distances[i]=np.linalg.norm(all_weights[i+1]-all_weights[i])
        print(distances[i])
    plt.plot(range(len(distances)),distances,label=str(args.lr_actor))

    plt.ylabel("distance")
    plt.xlabel("policy")
    plt.title("distance between successive policies")
    title="PG_zoom_distance.pdf"
    plt.savefig(title, format='pdf')
    plt.show()
