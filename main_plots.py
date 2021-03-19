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
import matplotlib.pyplot as plt
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
        simu.train(pw, params, policy, False, reward_file, "", study[0], 0, True)
    cem_time = chrono_cem.stop()
    return cem_time

def study_cem_fixed(params) -> None:
    """
    Start a sum study of cem fixed
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

    # fixed layers
    params.fix_layers = True

    print("cem fixed study") # cem fixed study
    chrono_cem_fixed = Chrono()
    for j in range(params.nb_repet):
        simu.env.reinit()
        if params.policy_type=="normal":
            policy = NormalPolicy(simu.obs_size, 24, 36, 1)
        pw = PolicyWrapper(policy, params.policy_type, simu.env_name, params.team_name, params.max_episode_steps)
        simu.train(pw,params, policy, False, reward_file, "", study[0], 0, True)
    cem_fixed_time = chrono_cem_fixed.stop()
    return cem_fixed_time

def study_pg(params) -> None:
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
        pw = PolicyWrapper(policy, params.policy_type, simu.env_name, params.team_name, params.max_episode_steps)
        simu.train(pw,params, policy, critic, policy_loss_file, critic_loss_file, study[0])
    pg_time = chrono_pg.stop()
    return pg_time

if __name__ == '__main__':
    args = get_args()
    print(args)
    if args.plot_mode == "all":
        #init dummy policy and dummy simulati
        # simu = make_simu_from_params(args)
        # policy=NormalPolicy(simu.obs_size, 24, 36, 1)
        # starting_weights=np.array(3*np.random.randn(args.nb_repet,policy.get_weights_dim(False)))
        # X_embedded = TSNE(n_components=2).fit_transform(starting_weights)
        # print(np.shape(X_embedded))
        # print(X_embedded)
        # plt.scatter(*zip(*X_embedded))
        # plt.show()
        #
        # policy=None
        # simu=None
        create_data_folders()
        cem_time = study_cem(args)
        exploit_total_reward_cem(args)
        cem_fixed_time = study_cem_fixed(args)
        exploit_total_reward_cem(args)
        pg_time = study_pg(args)
        exploit_reward_full(args)
        print("\n")
        print("====================TIME===================")
        print("Time cem : " + cem_time)
        print("Time cem_fixed : " + cem_fixed_time)
        print("Time pg : " + pg_time)
        print("===========================================")
        print("\n")
        #To make plot to compare pg and cem
        exploit_total_reward_cem_vs_pg(args)
    elif args.plot_mode == "cem":
        create_data_folders()
        cem_time = study_cem(args)
        exploit_total_reward_cem(args)
        print("\n")
        print("====================TIME===================")
        print("Time cem : " + cem_time)
        print("===========================================")
        print("\n")
    elif args.plot_mode == "cem_fixed":
        create_data_folders()
        cem_fixed_time = study_cem_fixed(args)
        exploit_total_reward_cem(args)
        print("\n")
        print("====================TIME===================")
        print("Time cem_fixed : " + cem_fixed_time)
        print("===========================================")
        print("\n")
    elif args.plot_mode == "pg":
        create_data_folders()
        pg_time = study_pg(args)
        exploit_reward_full(args)
        print("\n")
        print("====================TIME===================")
        print("Time pg : " + pg_time)
        print("===========================================")
        print("\n")
    elif args.plot_mode == "plot_only":
        #To make plot to compare pg and cem
        exploit_total_reward_cem_vs_pg(args)
