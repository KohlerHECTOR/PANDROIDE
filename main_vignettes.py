# coding: utf-8
import os

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import lzma
import gym
import ray

from progress.bar import Bar

from savedVignette import SavedVignette
from slowBar import SlowBar
from vector_util import *
# import torch
from chrono import Chrono
from simu import make_simu_from_params
from policies import GenericNet, BernoulliPolicy, NormalPolicy, SquashedGaussianPolicy, DiscretePolicy, PolicyWrapper
from arguments import get_args
from numpy.random import random


def create_data_folders() -> None:
    """
    Create folders where to put politics if they are not already there
    :return: nothing
    """
    if not os.path.exists("Models"):
        os.mkdir("./Models")


class Simulator(object):
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.env.reset()

    def step(self, action):
        return self.env.step(action)


def evaluate_policy(params, env, weights):
    """
    Perform an episode using the policy parameter and return the obtained reward
    Used to evaluate an already trained policy, without storing data for further training
    :return: the total reward collected during the episode
    """
    if params.multi_threading:
        ray.init(include_dashboard=False)

        @ray.remote
        def eval(params, nb_evals, sim):
            average_tot_score = 0
            for j in range(nb_evals):
                state = sim.env.reset()
                total_reward = 0
                for t in range(params.max_episode_steps):
                    action = policy.select_action(state,
                                                  params.deterministic_eval)
                    # print("action", action)
                    next_state, reward, done, _ = sim.env.step(action)
                    total_reward += reward
                    state = next_state
                    if done:
                        # print(total_reward)
                        average_tot_score += total_reward
                        break
            return average_tot_score / nb_evals

        policy = NormalPolicy(env.observation_space.shape[0], 24, 36, 1,
                              params.lr_actor)
        policy.set_weights(weights)
        workers = min(16, os.cpu_count() + 4)
        evals = int(params.nb_evals / workers)
        sim_list = []
        for i in range(workers):
            sim_list.append(Simulator(params.env_name))
        futures = [eval.remote(params, evals, sim) for sim in sim_list]
        returns = ray.get(futures)
        # print(returns)
        ray.shutdown()
        average_tot_score = np.sum(returns) / workers
        return average_tot_score
    else:
        policy = NormalPolicy(env.observation_space.shape[0], 24, 36, 1,
                              params.lr_actor)
        policy.set_weights(weights)
        average_tot_score = 0
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
                    average_tot_score += total_reward / args.nb_evals
                    break
        return average_tot_score


def load_policies(folder, params):
    """
     :param: folder : name of the folder containing policies
     Output : none (policies of the folder stored in self.env_dict)
     """
    listdir = os.listdir(folder)
    policies = []
    listdir.sort(key=lambda x: x.split('#')[3])
    colors = []
    for policy_file in listdir:
        if policy_file.split('#')[1] == 'CEM':
            colors.append("#9467bd")
        if policy_file.split('#')[1] == 'PG':
            colors.append("#d62728")
        pw = PolicyWrapper(GenericNet(), 0, "", "", "", 0)
        policy, _ = pw.load(directory + policy_file)
        policy = policy.get_weights()
        policies.append(policy)
    return policies, colors


if __name__ == '__main__':
    args = get_args()
    print(args)
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])
    create_data_folders()
    directory = os.getcwd() + '/Models/'
    policies, colors = load_policies(directory, args)

    if len(np.shape(policies)) > 1:
        theta0 = policies[0]
    else:
        theta0 = policies
        policies = [policies]
    num_params = len(theta0)
    base_vect = theta0
    D = getDirectionsMuller(args.nb_lines, num_params)

    # Compute fitness over these directions :
    previous_theta = None  # Remembers theta
    count = 1
    for policy in policies:

        # Change which model to load
        filename = args.saved_file_name + str(count)
        count += 1

        # Load the model

        # Get the new parameters
        theta0 = policy
        base_vect = theta0 if previous_theta is None else theta0 - previous_theta
        previous_theta = theta0
        print("Loaded parameters")

        # Processing the provided policies
        # 	Distance of each policy along their directions, directions taken by the policies
        policyDistance, policyDirection = [], []
        with SlowBar('Computing the directions to input policies',
                     max=len(policies) - 1) as bar:
            for p in policies:
                if not (p == policy).all():
                    distance = euclidienne(base_vect, p)
                    direction = (p - base_vect) / distance

                    # Storing the directions to remove them from those already sampled
                    policyDirection.append(direction)
                    # Storing the distances to the model
                    policyDistance.append(distance)
                    # 	Remove the closest direction in those sampled
                    del D[np.argmin(
                        [euclidienne(direction, dirK) for dirK in D])]
                    bar.next()

# 	Adding the provided policies
        D += policyDirection
        # 	Ordering the directions
        D = order_all_by_proximity(D)
        #	Keeping track of which directions stem from a policy
        copyD = [list(direction) for direction in D]
        #print(len(copyD))
        #print(copyD.index(list(direction)))
        indicesPolicies = [
            copyD.index(list(direction)) for direction in policyDirection
        ]
        print("indicesPolicies : " + str(indicesPolicies))
        del copyD

        # Evaluate the Model : mean, std
        print("Evaluating the model...")
        init_score = evaluate_policy(args, env, policy)
        print("Model initial fitness : " + str(init_score))

        # Study the geometry around the model
        print("Starting study around the model...")
        theta_plus_scores, theta_minus_scores = [], []
        image, base_image = [], []

        #	Norm of the model
        length_dist = euclidienne(base_vect, np.zeros(np.shape(base_vect)))
        # 		Direction taken by the model (normalized)
        d = np.zeros(np.shape(
            base_vect)) if length_dist == 0 else base_vect / length_dist

        # Print the number of workers with the multi-thread
        if args.multi_threading:
            workers = min(16, os.cpu_count() + 4)
            evals = int(args.nb_evals / workers)
            print("\n Multi-Threading Evaluations : " + str(workers) +
                  " workers with each " + str(evals) + " evaluations to do")

# Iterating over all directions, -1 is the direction that was initially taken by the model
        newVignette = SavedVignette(D,
                                    policyDistance=policyDistance,
                                    indicesPolicies=indicesPolicies,
                                    stepalpha=args.stepalpha,
                                    pixelWidth=args.pixelWidth,
                                    pixelHeight=args.pixelHeight,
                                    x_diff=args.x_diff,
                                    y_diff=args.y_diff,
                                    colors=colors)
        for step in range(0, len(D)):
            print("\nDirection ", step + 1, "/", len(D))
            # New parameters following the direction
            #	Changing the range and step of the Vignette if the optional input policies are beyond that range
            if len(policyDistance) > 0:
                print("Changing the range to reach the input policies to " +
                      str(max(max(policyDistance), args.maxalpha)) +
                      " instead of " + str(args.maxalpha))
                min_dist, max_dist = (args.minalpha,
                                      max(max(policyDistance), args.maxalpha))
            else:
                min_dist = args.minalpha
                max_dist = args.maxalpha
            step_dist = args.stepalpha * (max_dist - min_dist) / (
                args.maxalpha - args.minalpha)
            newVignette.stepalpha = step_dist
            # 	Sampling new models' parameters following the direction
            theta_plus, theta_minus = getPointsDirection(
                theta0, num_params, min_dist, max_dist, step_dist, d)

            # Get the next direction
            d = D[step]

            # Evaluate using new parameters
            scores_plus, scores_minus = [], []
            with SlowBar('Evaluating along the direction',
                         max=len(theta_plus)) as bar:
                for param_i in range(len(theta_plus)):
                    # 	Go forward in the direction

                    #		Get the new performance
                    scores_plus.append(
                        evaluate_policy(args, env, theta_plus[param_i]))
                    # 	Go backward in the direction

                    #		Get the new performance
                    scores_minus.append(
                        evaluate_policy(args, env, theta_minus[param_i]))
                    bar.next()

            # Inverting scores for a symetrical Vignette (theta_minus going left, theta_plus going right)
            scores_minus = scores_minus[::-1]
            line = scores_minus + [init_score] + scores_plus
            # 	Adding the line to the image
            newVignette.lines.append(line)

        computedImg = None
        computedImgCleaned = None
        try:
            # Computing the 2D Vignette
            if args.save2D is True:
                computedImg = newVignette.plot2D()
# Computing the 3D Vignette
#if args.save3D is True: newVignette.plot3DBand()
        except Exception as e:
            newVignette.saveInFile("{}/temp/{}".format(args.directoryFile,
                                                       filename))
            print(e)

# Saving the Vignette
        angles3D = [20, 45, 50, 65]  # angles at which to save the plot3D
        elevs = [0, 30, 60]
        newVignette.saveAll(filename,
                            saveInFile=args.saveInFile,
                            save2D=args.save2D,
                            save3D=args.save3D,
                            directoryFile=args.directoryFile,
                            directory2D=args.directory2D,
                            directory3D=args.directory3D,
                            computedImg=computedImg,
                            angles3D=angles3D,
                            elevs=elevs)
        break

    env.close()
