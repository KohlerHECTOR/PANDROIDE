# coding: utf-8
import os


import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import lzma
import gym

from progress.bar import Bar

from savedGradient import SavedGradient
from slowBar import SlowBar
from vector_util import *
# import torch
from chrono import Chrono
from simu import make_simu_from_params
from policies import GenericNet,BernoulliPolicy, NormalPolicy, SquashedGaussianPolicy, DiscretePolicy, PolicyWrapper
from arguments import get_args
from numpy.random import random




def evaluate_policy(params, env, weights):
    policy = NormalPolicy(env.observation_space.shape[0], 24, 36, 1, params.lr_actor)
    policy.set_weights(weights)
    average_tot_score=0
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
                average_tot_score+=total_reward/args.nb_evals
                break
    return average_tot_score

def load_policies(folder):
    """
     :param: folder : name of the folder containing policies
     Output : none (policies of the folder stored in self.env_dict)
     """
    listdir = os.listdir(folder)
    policies = []
    for policy_file in listdir:
        pw = PolicyWrapper(GenericNet(), "", "", "", 0)
        policy,_ = pw.load(directory+policy_file)
        policy = policy.get_weights()
        policies.append(policy)
    return policies

if __name__ == '__main__':
    args = get_args()
    print(args)
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])
    directory = os.getcwd() + '/Models/'
    policies=load_policies(directory)

    if len(np.shape(policies))>1:
        theta0 = policies[0]
    else:
        theta0 = policies
        policies = [policies]
    num_params = len(theta0)
    base_vect = theta0
    D = getDirectionsMuller(args.nb_lines,num_params)
    D = order_all_by_proximity(D)

	# Compute fitness over these directions :
    previous_theta = None # Remembers theta
    newGradient = SavedGradient(directions=[], results=[], red_markers=[], green_markers=[],nbLines=args.line_height, pixelWidth=args.pixelWidth, pixelHeight=args.pixelHeight, maxValue=args.maxValue,dotText=args.dotText, dotWidth=args.dotWidth, xMargin=args.xMargin, yMargin=int(args.pixelHeight/2))

    for policy in policies:
        filename = args.saved_file_name+".png"


		# Load the model


		# Get the new parameters
        theta0 = policy
        init_score=evaluate_policy(args,env,policy)
        base_vect = theta0 if previous_theta is None else theta0 - previous_theta
        previous_theta = theta0
        print("Loaded parameters")
        length_dist = euclidienne(base_vect, np.zeros(np.shape(base_vect)))
        d = np.zeros(np.shape(base_vect)) if length_dist ==0 else base_vect / length_dist
        newGradient.directions.append(d)
		#		New parameters following the direction
        theta_plus, theta_minus = getPointsDirection(theta0, num_params, args.minalpha, args.maxalpha, args.stepalpha, d)

		# Processing the provided policies
		# 	Distance of each policy along their directions, directions taken by the policies
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

        scores_minus = scores_minus[::-1]
        line = scores_minus + [init_score] + scores_plus

    		# Adding the results
        last_params_marker = int(length_dist/args.stepalpha)
    		#	Mark two consecutive positions on the line
        marker_actor = int((len(line)-1)/2)
        marker_last = max(marker_actor-last_params_marker, 0)
    		#		A list of the markers, previous will be shown in red and current in green
        newGradient.red_markers.append(marker_last)

        newGradient.green_markers.append(marker_actor)
    		# 	Putting it all together
        newGradient.results.append(line)

    try:
    		# Assembling the image, saving it if asked
        newGradient.computeImage(saveImage=args.saveImage, filename=filename, directory=args.directoryImage)
    except Exception as e:
        newGradient.saveGradient(filename=filename, directory=args.directoryFileGrad)

    	# Saving the SavedGradient if asked
    if args.saveFile is True: newGradient.saveGradient(filename=filename, directory=args.directoryFileGrad)
    env.close()
