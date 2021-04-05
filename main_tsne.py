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
        pw = PolicyWrapper(policy, params.policy_type, simu.env_name,j, params.team_name, params.max_episode_steps)
        all_pg_weights,best_pg_weights,all_rewards=simu.train(pw,params, policy, critic, policy_loss_file, critic_loss_file, study[0])
    pg_time = chrono_pg.stop()
    return pg_time,all_pg_weights,best_pg_weights,all_rewards

def normalize_reward(reward,reward_min,reward_max):
    print(reward_max,reward_min)
    return round((0.1+(0.9999-0.1)*(reward-reward_min)/(reward_max-reward_min)),5)

def get_embedding_of_weights(weights):
    return np.array(TSNE(n_components=2).fit_transform(weights))

def get_x_y_positions(embedding_of_weights):
    return embedding_of_weights[:,0],embedding_of_weights[:,1]

def get_min_max_of_axes(array_of_embeddings_of_weights):
    x_min=np.inf
    x_max=-np.inf
    y_min=np.inf
    y_max=-np.inf
    for array in array_of_embeddings_of_weights:
        x,y=get_x_y_positions(array)
        if np.amin(x)<x_min:
            x_min=np.amin(x)
        if np.amax(x)>x_max:
            x_max=np.amax(x)
        if np.amin(y)<y_min:
            y_min=np.amin(y)
        if np.amax(y)>y_max:
            y_max=np.amax(y)
    return x_min,x_max,y_min,y_max

def get_list_of_colours(array_of_rewards):
    all_rewards=[]
    for reward_array in array_of_rewards:
        all_rewards=np.append(all_rewards,reward_array)
    reward_min=np.amin(all_rewards)
    reward_max=np.amax(all_rewards)
    COLORSpg = np.ones((len(array_of_rewards[0]),4))*(1,0,0,0)
    COLORScem = np.ones((len(array_of_rewards[0]),4))*(0,0,1,0)
    COLORS=np.append(COLORSpg,COLORScem,axis=0)
    for i in range(np.shape(COLORS)[0]):
        COLORS[i,3]=normalize_reward(all_rewards[i],reward_min,reward_max)
    print(COLORS)
    return COLORS




if __name__ == '__main__':
    args = get_args()
    print(args)
    create_data_folders()

    ## Get weights and rewards and some plots
    cem_time,all_cem_weights,best_cem_weights,all_rewards_cem = study_cem(args)
    exploit_total_reward_cem(args)
    pg_time,all_pg_weights,best_pg_weights,all_rewards_pg = study_pg(args)
    exploit_reward_full(args)

    ## Reduce dimensions of the NN weights to 2 dimensions
    embedded_pg = get_embedding_of_weights(all_pg_weights)
    embedded_cem = get_embedding_of_weights(all_cem_weights)


    ## Get range of axis so the animation is centered
    x_min,x_max,y_min,y_max = get_min_max_of_axes(np.array(([embedded_pg,embedded_cem])))

    matplotlib.rcParams['toolbar'] = 'None'
    fig, ax = plt.subplots()
    scat = ax.scatter([x_min,x_max],[y_min,y_max])
    nb_frames=np.shape(embedded_pg)[0]+np.shape(embedded_cem)[0]
    print(nb_frames)

    POSITIONS=np.append(embedded_pg,embedded_cem,axis=0)
    COLORS=get_list_of_colours(np.array([all_rewards_pg,all_rewards_cem]))
    SIZES=np.ones(nb_frames)*40

    def animate(frame):
        global COLORS, POSITIONS, SIZES
        scat.set_offsets(POSITIONS[:frame+1,:])
        scat.set_color(COLORS[:frame+1])
        scat.set_sizes(SIZES[:frame+1])
        return scat,
        # plt.show()

    anim=FuncAnimation(fig, animate, frames=nb_frames, blit=False, interval=300, repeat=False)
        #plt.show()

    anim.save("visuPOIDScempg_trajpg_" +str(args.nb_trajs_pg)+"_lr_"+str(args.lr_actor) ".gif")
