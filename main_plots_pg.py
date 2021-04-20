import os
# import torch
from chrono import Chrono
from simu import make_simu_from_params
from policies import BernoulliPolicy, NormalPolicy, SquashedGaussianPolicy, DiscretePolicy, PolicyWrapper
from critics import VNetwork, QNetworkContinuous
from arguments import get_args
from visu.visu_critics import plot_critic
from visu.visu_policies import plot_policy
from visu.visu_results import exploit_total_reward_cem
from visu.visu_results import exploit_reward_full
from visu.visu_results import exploit_total_reward_cem_vs_pg
from visu.visu_weights import plot_normal_histograms
from numpy.random import random
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import *
from matplotlib import rc
from sklearn.manifold import TSNE

def create_data_folders() -> None:
    """
    Create folders where to save output files if they are not already there
    :return: nothing
    """
    if not os.path.exists("data/save"):
        os.mkdir("./data")
        os.mkdir("./data/save")
    if not os.path.exists('data/policies/'):
        os.mkdir('data/policies/')
    if not os.path.exists('data/results/'):
        os.mkdir('data/results/')


def set_files(study_name, env_name):
    """
    Create files to save the reward by duration
    :param study_name: the name of the study
    :param env_name: the name of the environment
    :return:
    """
    reward_name = "data/save/total_reward_" + "cem" + '_' + env_name + '.txt'
    reward_file = open(reward_name, "w")
    return reward_file

def set_files_pg(study_name, env_name):
    """
    Create files to save the policy loss and the critic loss
    :param study_name: the name of the study
    :param env_name: the name of the environment
    :return:
    """
    policy_loss_name = "data/save/policy_loss_" + study_name + '_' + env_name + ".txt"
    policy_loss_file = open(policy_loss_name, "w")
    critic_loss_name = "data/save/critic_loss_" + study_name + '_' + env_name + ".txt"
    critic_loss_file = open(critic_loss_name, "w")
    return policy_loss_file, critic_loss_file


def study_cem(params) -> None:
    """
    Start a sum study of cem
    :param params: the parameters of the study
    :return: nothing
    """

    assert params.policy_type in ['normal'], 'unsupported policy type'
    # cuda = torch.device('cuda')
    study = params.gradients
    simu = make_simu_from_params(params)
    simu.env.set_file_name(study[0] + '_' + simu.env_name)
    reward_file = set_files(study[0], simu.env_name)
    print("study : ", study)

    # defixed layers
    params.fix_layers = False

    print("cem study") # cem study
    chrono_cem = Chrono()
    for j in range(params.nb_repet):
        simu.env.reinit()
        if params.policy_type=="normal":
            policy = NormalPolicy(simu.obs_size, 24, 36, 1)
        pw = PolicyWrapper(policy, params.policy_type, simu.env_name, j,params.team_name, params.max_episode_steps)
        all_cem_weights,best_cem_weights,all_rewards=simu.train(pw, params, policy, False, reward_file, "", study[0], 0, True)
    cem_time = chrono_cem.stop()
    return cem_time,all_cem_weights,best_cem_weights,all_rewards

def study_pg(params):
    """
    Start a sum study of pg
    :param params: the parameters of the study
    :return: nothing
    """

    assert params.policy_type in ['normal'], 'unsupported policy type'
    # cuda = torch.device('cuda')
    study = params.gradients
    simu = make_simu_from_params(params)
    simu.env.set_file_name(study[0] + '_' + simu.env_name)
    print("study : ", study)

    # defixed layers
    params.fix_layers = False

    print("pg study") # pg study
    chrono_pg = Chrono()
    simu.env.set_file_name(study[0] + '_' + simu.env_name)
    policy_loss_file, critic_loss_file = set_files_pg(study[0], simu.env_name)
    for j in range(params.nb_repet):
        simu.env.reinit()
        if params.policy_type == "normal":
            policy = NormalPolicy(simu.obs_size, 24, 36, 1, params.lr_actor)
        critic = VNetwork(simu.obs_size, 24, 36, 1, params.lr_critic)
        pw = PolicyWrapper(policy, params.policy_type, simu.env_name,j, params.team_name, params.max_episode_steps)
        all_pg_weights,best_pg_weights,all_rewards,idx_best=simu.train(pw,params, policy, critic, policy_loss_file, critic_loss_file, study[0])
    pg_time = chrono_pg.stop()
    return all_pg_weights,all_rewards,idx_best

def evaluate_policy(params, env, weights):
    policy = NormalPolicy(env.observation_space.shape[0], 24, 36, 1, params.lr_actor)
    policy.set_weights(weights)
    scores=np.zeros(int(args.nb_eval))
    for j in range(int(args.nb_eval)):
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

if __name__ == '__main__':
    args = get_args()
    print(args)
    env = gym.make(args.env_name)
    all_pg_weights,all_rewards,idx_best = study_pg(args)

    weight_t_0 = all_pg_weights[idx_best-1]
    weight_t_1 = all_pg_weights[idx_best]
    weight_t_2 = all_pg_weights[idx_best+1]
################################################################################

    distance = np.linalg.norm(weight_t_1-weight_t_0)
    print("distance between policies: ",distance)
    nb_steps=50
    step_size=1/nb_steps
    line=weight_t_1-weight_t_0
    policies_on_segment=np.zeros((nb_steps-1,len(weight_t_0)))
    for i in range(nb_steps-1):
        policies_on_segment[i]=weight_t_0+step_size*(i+1)*line


    line_small = policies_on_segment[0] - weight_t_0
    small_steps=1/(nb_steps+1)
    policies_in_small_interval = np.zeros((nb_steps,len(weight_t_0)))
    for i in range(nb_steps):
        policies_in_small_interval[i] = weight_t_0 + (i+1) * line_small * small_steps

    all_policies_on_segment=np.append(policies_in_small_interval,policies_on_segment,axis=0)
    coordinates_on_segment = np.zeros(len(all_policies_on_segment))
    for i in range(len(coordinates_on_segment)):
        coordinates_on_segment[i]=np.linalg.norm(all_policies_on_segment[i]-weight_t_0)
    coordinates_on_segment=np.append(0,coordinates_on_segment)
    coordinates_on_segment= np.append(coordinates_on_segment,distance)


    means=[]
    std=[]
    for policy in all_policies_on_segment:
        scores=evaluate_policy(args,env,policy)
        means.append(scores.mean(axis=0))
        std.append(scores.std(axis=0))
    means=np.array(means)
    means= np.append(all_rewards[idx_best-1],means)
    means=np.append(means,all_rewards[idx_best])
    std=np.array(std)
    std=np.append(0,std)
    std=np.append(std,0)

################################################################################
################################################################################
    distance2 = np.linalg.norm(weight_t_2-weight_t_1)
    print("distance between policies: ",distance2)
    nb_steps=50
    step_size=1/nb_steps
    line2=weight_t_2-weight_t_1
    policies_on_segment2=np.zeros((nb_steps-1,len(weight_t_1)))
    for i in range(nb_steps-1):
        policies_on_segment2[i]=weight_t_1+step_size*(i+1)*line2


    line_small2 = policies_on_segment2[0] - weight_t_1
    small_steps=1/(nb_steps+1)
    policies_in_small_interval2 = np.zeros((nb_steps,len(weight_t_1)))
    for i in range(nb_steps):
        policies_in_small_interval2[i] = weight_t_1 + (i+1) * line_small2 * small_steps

    all_policies_on_segment2=np.append(policies_in_small_interval2,policies_on_segment2,axis=0)
    coordinates_on_segment2 = np.zeros(len(all_policies_on_segment2))
    for i in range(len(coordinates_on_segment2)):
        coordinates_on_segment2[i]=np.linalg.norm(all_policies_on_segment2[i]-weight_t_1)
    coordinates_on_segment2=np.append(0,coordinates_on_segment2)
    coordinates_on_segment2= np.append(coordinates_on_segment2,distance2)


    means2=[]
    std2=[]
    for policy in all_policies_on_segment2:
        scores=evaluate_policy(args,env,policy)
        means2.append(scores.mean(axis=0))
        std2.append(scores.std(axis=0))
    means2=np.array(means2)
    means2= np.append(all_rewards[idx_best],means2)
    means2=np.append(means2,all_rewards[idx_best+1])
    std2=np.array(std2)
    std2=np.append(0,std2)
    std2=np.append(std2,0)
################################################################################
    markers_on=[0,len(coordinates_on_segment)-1]
    plt.plot(coordinates_on_segment,means,markevery=markers_on,marker = 'o',markersize=12)
    plt.fill_between(coordinates_on_segment,means+std,means-std,alpha=0.3)
    plt.show()
    plt.clf()
    markers_on=[0,len(coordinates_on_segment2)-1]
    plt.plot(coordinates_on_segment2,means2,markevery=markers_on,marker = 'o',markersize=12)
    plt.fill_between(coordinates_on_segment2,means2+std2,means2-std2,alpha=0.3)
    plt.show()

    # title='PG_'+'lr_'+str(args.lr_actor)+'_evals_'+str(args.nb_eval)+'.svg'
    # plt.savefig(title, format='svg',dpi=1200)
    env.close()

    # env = gym.make(args.env_name)
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    # max_action = int(env.action_space.high[0])
