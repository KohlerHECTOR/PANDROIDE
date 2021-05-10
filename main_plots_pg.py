import os
# import torch
from chrono import Chrono
from simu import make_simu_from_params
from policies import GenericNet,BernoulliPolicy, NormalPolicy, SquashedGaussianPolicy, DiscretePolicy, PolicyWrapper
from arguments import get_args
from numpy.random import random
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


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
    rewards = []
    for policy_file in listdir:
        pw = PolicyWrapper(GenericNet(),0, "", "", "", 0)
        policy,reward = pw.load(directory+policy_file)
        policy = policy.get_weights()
        policies.append(policy)
        rewards.append(reward)
    return policies,rewards

def plot_evals_on_segment(args,policies,rewards):
    pol_t0 = policies[0]
    pol_t1 = policies[1]
    distance = np.linalg.norm(pol_t1-pol_t0)
    print("distance between policies: ",distance)
    nb_steps=args.nb_intervals
    step_size=1/nb_steps
    line=pol_t1-pol_t0
    policies_on_segment=np.zeros((nb_steps-1,len(pol_t0)))
    for i in range(nb_steps-1):
        policies_on_segment[i]=pol_t0+step_size*(i+1)*line


    line_small = policies_on_segment[0] - pol_t0
    small_steps=1/(nb_steps+1)
    policies_in_small_interval = np.zeros((nb_steps,len(pol_t0)))
    for i in range(nb_steps):
        policies_in_small_interval[i] = pol_t0 + (i+1) * line_small * small_steps

    all_policies_on_segment=np.append(policies_in_small_interval,policies_on_segment,axis=0)
    coordinates_on_segment = np.zeros(len(all_policies_on_segment))
    for i in range(len(coordinates_on_segment)):
        coordinates_on_segment[i]=np.linalg.norm(all_policies_on_segment[i]-pol_t0)
    coordinates_on_segment=np.append(0,coordinates_on_segment)
    coordinates_on_segment= np.append(coordinates_on_segment,distance)


    means=[]
    quantile_25 = []
    quantile_75 = []
    for policy in all_policies_on_segment:
        scores=evaluate_policy(args,env,policy)
        means.append(scores.mean(axis=0))
        quantile_25.append(np.quantile(scores, 0.25))
        quantile_75.append(np.quantile(scores, 0.75))
    means=np.array(means)
    means= np.append(rewards[0],means)
    means=np.append(means,rewards[-1])
    # means= np.append(rewards[idx_best-1],means)
    # means=np.append(means,rewards[idx_best])
    quantile_25=np.array(quantile_25)
    quantile_25=np.append(0,quantile_25)
    quantile_25=np.append(quantile_25,0)
    quantile_75=np.array(quantile_75)
    quantile_75=np.append(0,quantile_75)
    quantile_75=np.append(quantile_75,0)
    markers_on=[0,len(coordinates_on_segment)-1]
    plt.plot(coordinates_on_segment,means,markevery=markers_on,marker = 'o',markersize=12)
    plt.fill_between(coordinates_on_segment,quantile_75,quantile_25,alpha=0.3)
    title="CEM_landscape_best_bestplus1.pdf"
    
    plt.ylabel("reward")
    plt.xlabel("distance to initial policy")
    plt.title("rewards landscape between two policies")
    plt.savefig(title, format='pdf')
    plt.show()


if __name__ == '__main__':
    args = get_args()
    print(args)
    env = gym.make(args.env_name)
    directory = os.getcwd() + '/Models/'
    policies,rewards=load_policies(directory)
    print(rewards)
    print(policies)
    plot_evals_on_segment(args,policies,rewards)
    env.close()

    # env = gym.make(args.env_name)
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    # max_action = int(env.action_space.high[0])
