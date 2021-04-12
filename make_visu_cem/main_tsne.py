import os
# import torch
from chrono import Chrono
from simu import make_simu_from_params
from policies import NormalPolicy,PolicyWrapper
from arguments import get_args
from numpy.random import random
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import *
from matplotlib import rc
from sklearn.manifold import TSNE
import math






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
    reward_file = None
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
        all_weights,all_rewards,all_pops,all_pops_scores,is_kept=simu.train(pw, params, policy, False, reward_file, "", study[0], 0, True)
    cem_time = chrono_cem.stop()
    return all_weights,all_rewards,all_pops,all_pops_scores,is_kept


def normalize_reward(reward,reward_min,reward_max):
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
    reward_min=np.amin(array_of_rewards)
    reward_max=np.amax(array_of_rewards)
    # COLORS_centroids = np.ones((len(array_of_rewards[0]),4))*(1,0,0,0)
    COLORS_offsprings = np.ones((len(array_of_rewards),4))*(1,0,0,0)
    # COLORS=np.append(COLORSpg,COLORScem,axis=0)
    for i in range(len(array_of_rewards)):
        COLORS_offsprings[i,3]=normalize_reward(array_of_rewards[i],reward_min,reward_max)
    return COLORS_offsprings

def get_list_of_colours_centroids(array_of_rewards):
    reward_min=np.amin(array_of_rewards)
    reward_max=np.amax(array_of_rewards)
    # COLORS_centroids = np.ones((len(array_of_rewards[0]),4))*(1,0,0,0)
    COLORS_centroids = np.ones((len(array_of_rewards),4))*(0,0,1,0)
    # COLORS=np.append(COLORSpg,COLORScem,axis=0)
    for i in range(len(array_of_rewards)):
        COLORS_centroids[i,3]=normalize_reward(array_of_rewards[i],reward_min,reward_max)
    return COLORS_centroids


















if __name__ == '__main__':
    args = get_args()
    print(args)

    ## Get weights and rewards and some plots
    all_weights,all_rewards,all_pops,all_pops_scores,is_kept = study_cem(args)

    dimension0=np.shape(all_pops)[0]
    dimension1=np.shape(all_pops)[1]
    dimension2=np.shape(all_pops)[2]
    ## Reduce dimensions of the NN weights to 2 dimensions
    # embedded_centroids = get_embedding_of_weights(all_weights)

    # embedded_offsprings=np.zeros((dimension0,dimension1,2))

    all_weights_and_pops=np.append(all_weights,all_pops.reshape((dimension0*dimension1,dimension2)),axis=0)
    embedded=get_embedding_of_weights(all_weights_and_pops)
    embedded_centroids=embedded[:dimension0+1]
    embedded_offsprings=embedded[-dimension0*dimension1:]
    embedded_offsprings=embedded_offsprings.reshape((dimension0,dimension1,2))
    # for i in range(dimension0):
    #     embedded_offsprings[i] = get_embedding_of_weights(all_pops[i])

    COLORS_offsprings=np.zeros((dimension0,dimension1,4))
    for i in range(dimension0):
        COLORS_offsprings[i]=get_list_of_colours(all_pops_scores[i])



    COLORS_centroids=get_list_of_colours_centroids(all_rewards)

    print('shape colors centres', np.shape(COLORS_centroids))
    print('shape colors offsprings', np.shape(COLORS_offsprings))




    ## Get range of axis so the animation is centered
    x_min,x_max,y_min,y_max = get_min_max_of_axes(np.array(([embedded_centroids,embedded_offsprings.reshape((dimension0*dimension1,2))])))

    matplotlib.rcParams['toolbar'] = 'None'
    fig, ax = plt.subplots()
    scat = ax.scatter([x_min,x_max],[y_min,y_max],label='0')
    # scat = ax.scatter([],[],label='0')
    nb_frames=int(3*args.nb_cycles)

    POSITIONS_centroids=embedded_centroids
    print('shape centroids ', np.shape(POSITIONS_centroids))
    POSITIONS_offsprings=embedded_offsprings
    print('shape offsprings ', np.shape(POSITIONS_offsprings))
    # COLORS_centroids,COLORS_offsprings=get_list_of_colours(np.array([all_rewards,all_pops_scores.flatten()]))
    # print('shape colors centres', np.shape(COLORS_centroids))
    # print('shape colors offsprings', np.shape(COLORS_offsprings))
    SIZES=np.ones(1000)*9
    count=-1
    i1=3
    i2=2
    i3=1
    def animate(frame):
        global COLORS_centroids, COLORS_offsprings, POSITIONS_centroids, POSITIONS_offsprings, SIZES, i1,i2,i3,is_kept,dimension1, nb_frames,args,count,all_rewards,legend
        index=math.floor((count+1)/3)


        if i1%3==0:


            scat.set_offsets(np.append([POSITIONS_centroids[index]],POSITIONS_offsprings[index],axis=0))
            scat.set_color(np.append([COLORS_centroids[index]],COLORS_offsprings[index],axis=0))
            scat.set_sizes(np.append([40],SIZES[:dimension1]))
            scat.set_label('reward '+str(int(all_rewards[index])))
            plt.legend()

        if i2%3==0:


            scat.set_offsets(POSITIONS_offsprings[index][is_kept[index].astype(int)])
            scat.set_color(COLORS_offsprings[index][is_kept[index].astype(int)])
            scat.set_sizes(SIZES[:len(is_kept[index])])

        if i3%3== 0:


            scat.set_offsets(np.append([POSITIONS_centroids[index+1]],POSITIONS_offsprings[index][is_kept[index].astype(int)],axis=0))
            scat.set_color(np.append([COLORS_centroids[index+1]],COLORS_offsprings[index][is_kept[index].astype(int)],axis=0))
            scat.set_sizes(np.append([40],SIZES[:len(is_kept[index])]))
            scat.set_label('reward '+str(int(all_rewards[index+1])))
            plt.legend()


        count+=1
        i1+=1
        i2+=1
        i3+=1
        return scat,
        plt.show()

    anim=FuncAnimation(fig, animate, frames=nb_frames-1, blit=False, interval=1200, repeat=False)
        #plt.show()

    anim.save("visuCEM_labels_new_pop25_10EVAL.gif")
