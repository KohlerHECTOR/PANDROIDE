import os
# import torch
from chrono import Chrono
from simu import make_simu_from_params
from policies import NormalCEM,BernoulliCEM,BernoulliPolicy, NormalPolicy, SquashedGaussianPolicy, DiscretePolicy, PolicyWrapper
from critics import VNetwork, QNetworkContinuous
from arguments import get_args
from visu.visu_critics import plot_critic
from visu.visu_policies import plot_policy
from visu.visu_results import exploit_total_reward_cem
from visu.visu_results import exploit_duration_full_cem
from visu.visu_results import plot_results
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
    Create files to save the policy loss and the critic loss
    :param study_name: the name of the study
    :param env_name: the name of the environment
    :return:
    """
    policy_loss_name = "data/save/policy_loss_" + study_name + '_' + env_name + ".txt"
    policy_loss_file = open(policy_loss_name, "w")
    return policy_loss_file


def study_cem(params) -> None:
    """
    Start a study of the policy gradient algorithms
    :param params: the parameters of the study
    :return: nothing
    """
    assert params.policy_type in ['bernoulliCEM', 'normalCEM','normalSimple'], 'unsupported policy type'
    chrono = Chrono()
    # cuda = torch.device('cuda')
    simu = make_simu_from_params(params)
    study = "cem"
    print("study : ", study)
    simu.env.set_file_name(str(study) + '_' + simu.env_name)
    policy_loss_file = set_files(str(study), simu.env_name)
    for j in range(params.nb_repet):
        simu.env.reinit()
        if params.policy_type == "bernoulliCEM":
            policy = BernoulliCEM(simu.obs_size, 24, 36, 1)
        if params.policy_type=='normalCEM':
            policy = NormalCEM(simu.obs_size, 24, 36, 1)
        if params.policy_type=="normalSimple":
            policy = NormalSimple(random(simu.obs_size))
        pw = PolicyWrapper(policy, params.policy_type, simu.env_name, params.team_name, params.max_episode_steps)
        #plot_policy(policy, simu.env, True, simu.env_name, study, '_ante_', j, plot=False)
        simu.trainCEM(pw, params, policy, policy_loss_file, params.study_name)
        #plot_policy(policy, simu.env, True, simu.env_name, study, '_post_', j, plot=False)
    chrono.stop()

if __name__ == '__main__':
    args = get_args()
    print(args)
    create_data_folders()
    study_cem(args)
    #plot_results(args)
    exploit_total_reward_cem(args)
    exploit_duration_full_cem(args)
