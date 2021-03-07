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
from visu.visu_results import exploit_duration_full_cem
from visu.visu_results import exploit_reward_full_cem
from visu.visu_results import exploit_total_reward_cem_vs_pg
from numpy.random import random
import gym


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


def study_cem(params) -> None:
    """
    Start a study of the policy gradient algorithms
    :param params: the parameters of the study
    :return: nothing
    """
    assert params.policy_type in ['bernoulli', 'normal'], 'unsupported policy type'
    chrono = Chrono()
    # cuda = torch.device('cuda')
    study = params.gradients
    simu = make_simu_from_params(params)
    for i in range(1): #len(study) Only sum here
        simu.env.set_file_name(study[i] + '_' + simu.env_name)
        reward_file = set_files(study[i], simu.env_name)
        print("study : ", study[i])
        for j in range(params.nb_repet):
            simu.env.reinit()
            if params.policy_type == "bernoulli":
                policy = BernoulliPolicy(simu.obs_size, 24, 36, 1)
            if params.policy_type=="normal":
                policy = NormalPolicy(simu.obs_size, 24, 36, 1)
            pw = PolicyWrapper(policy, params.policy_type, simu.env_name, params.team_name, params.max_episode_steps)
            #plot_policy(policy, simu.env, True, simu.env_name, study[i], '_ante_', j, plot=False)
            simu.train(pw, params, policy, False, reward_file, "", study[i], 0, True)
            #plot_policy(policy, simu.env, True, simu.env_name, study[i], '_post_', j, plot=False)
    chrono.stop()

if __name__ == '__main__':
    args = get_args()
    print(args)
    create_data_folders()
    study_cem(args)
    #exploit_reward_full_cem(args)
    #exploit_duration_full_cem(args)
    exploit_total_reward_cem(args)
    #To make plot to compare pg and cem
    #exploit_duration_full_cem_vs_pg(args, study, env)
    exploit_total_reward_cem_vs_pg(args)
