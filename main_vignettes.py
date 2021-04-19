# coding: utf-8
import os


import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import lzma
import gym

from progress.bar import Bar

from savedVignette import SavedVignette
from slowBar import SlowBar
from vector_util import *
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

def evaluate_policy(params, env, weights):
    policy = NormalPolicy(env.observation_space.shape[0], 24, 36, 1, params.lr_actor)
    policy.set_weights(weights)
    average_tot_score=0
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
                average_tot_score+=total_reward/args.nb_eval
                break
    return average_tot_score


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
        all_cem_weights,best_cem_weights,all_rewards,index_best=simu.train(pw, params, policy, False, reward_file, "", study[0], 0, True)
    cem_time = chrono_cem.stop()
    return pg_time,all_pg_weights,best_pg_weights,all_rewards,index_best

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
        all_pg_weights,best_pg_weights,all_rewards,index_best=simu.train(pw,params, policy, critic, policy_loss_file, critic_loss_file, study[0])
    pg_time = chrono_pg.stop()
    return pg_time,all_pg_weights,best_pg_weights,all_rewards,index_best

if __name__ == '__main__':
    args = get_args()
    print(args)
    pg_time,all_pg_weights,best_pg_weights,all_rewards_pg,idx_best = study_pg(args)
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])

    theta0 = best_pg_weights
    num_params = len(theta0)
    base_vect = theta0
    D = getDirectionsMuller(args.nb_lines,num_params)
    policies=[all_pg_weights[idx_best-1],all_pg_weights[idx_best]]

	# Compute fitness over these directions :
    previous_theta = None # Remembers theta
    count=1
    for policy in policies:

		# Change which model to load
        filename = args.saved_file_name+str(count)
        count+=1

		# Load the model


		# Get the new parameters
        theta0 = policy
        base_vect = theta0 if previous_theta is None else theta0 - previous_theta
        previous_theta = theta0
        print("Loaded parameters")

		# Processing the provided policies
		# 	Distance of each policy along their directions, directions taken by the policies
        policyDistance, policyDirection = [], []
        with SlowBar('Computing the directions to input policies', max=len(policies)) as bar:
            for p in policies:
                if not (p==policy).all():
                    distance = euclidienne(base_vect, p);	direction = (p - base_vect) / distance

					# Storing the directions to remove them from those already sampled
                    policyDirection.append(direction)
					# Storing the distances to the model
                    policyDistance.append(distance)
					# 	Remove the closest direction in those sampled
                    del D[np.argmin([euclidienne(direction, dirK) for dirK in D])]
                    bar.next()

		# 	Adding the provided policies
        print(policyDirection)
        D += policyDirection
		# 	Ordering the directions
        D = order_all_by_proximity(D)
		#	Keeping track of which directions stem from a policy
        copyD = [list(direction) for direction in D]
        indicesPolicies = [copyD.index(list(direction)) for direction in policyDirection]
        print(indicesPolicies)
        del copyD

		# Evaluate the Model : mean, std
        print("Evaluating the model...")
        init_score = evaluate_policy(args,env,policy)
        print("Model initial fitness : "+str(init_score))

		# Study the geometry around the model
        print("Starting study around the model...")
        theta_plus_scores, theta_minus_scores = [], []
        image, base_image = [], []

		#	Norm of the model
        length_dist = euclidienne(base_vect, np.zeros(np.shape(base_vect)))
		# 		Direction taken by the model (normalized)
        d = np.zeros(np.shape(base_vect)) if length_dist ==0 else base_vect / length_dist

		# Iterating over all directions, -1 is the direction that was initially taken by the model
        newVignette = SavedVignette(D, policyDistance=policyDistance, indicesPolicies=indicesPolicies,stepalpha=args.stepalpha, pixelWidth=args.pixelWidth, pixelHeight=args.pixelHeight,x_diff=args.x_diff, y_diff=args.y_diff)
        for step in range(-1,len(D)):
            print("\nDirection ", step, "/", len(D)-1)
			# New parameters following the direction
			#	Changing the range and step of the Vignette if the optional input policies are beyond that range
            min_dist, max_dist = (args.minalpha, max(max(policyDistance), args.maxalpha))
            step_dist = args.stepalpha * (max_dist - min_dist) / (args.maxalpha - args.minalpha)
            newVignette.stepalpha = step_dist
			# 	Sampling new models' parameters following the direction
            theta_plus, theta_minus = getPointsDirection(theta0, num_params, min_dist, max_dist, step_dist, d)

			# Get the next direction
            if step != -1:	d = D[step]

			# Evaluate using new parameters
            scores_plus, scores_minus = [], []
            with SlowBar('Evaluating along the direction', max=len(theta_plus)) as bar:
                for param_i in range(len(theta_plus)):
					# 	Go forward in the direction

					#		Get the new performance
                    scores_plus.append(evaluate_policy(args,env,theta_plus[param_i]))
					# 	Go backward in the direction

					#		Get the new performance
                    scores_minus.append(evaluate_policy(args,env,theta_minus[param_i]))
                    bar.next()

			# Inverting scores for a symetrical Vignette (theta_minus going left, theta_plus going right)
            scores_minus = scores_minus[::-1]
            line = scores_minus + [init_score] + scores_plus
			# 	Adding the line to the image
            if step == -1:	newVignette.baseLines.append(line)
            else:	newVignette.lines.append(line)

        computedImg = None
        try:
			# Computing the 2D Vignette
            if args.save2D is True:	computedImg = newVignette.plot2D()
			# Computing the 3D Vignette
            if args.save3D is True: newVignette.plot3D()
        except Exception as e:
            newVignette.saveInFile("{}/temp/{}".format(args.directoryFile, filename))
            print(e)

		# Saving the Vignette
        angles3D = [20,45,50,65] # angles at which to save the plot3D
        elevs= [0, 30, 60]
        newVignette.saveAll(filename, saveInFile=args.saveInFile, save2D=args.save2D, save3D=args.save3D,directoryFile=args.directoryFile, directory2D=args.directory2D, directory3D=args.directory3D,computedImg=computedImg, angles3D=angles3D, elevs=elevs)


    env.close()
