# coding: utf-8
import os

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import lzma
import gym
from numpy import ma
import ray
import sys

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
from environment import Simulator, make_env


def create_data_folders() -> None:
    """
    Create folders where to put politics if they are not already there
    :return: nothing
    """
    if not os.path.exists("Models"):
        os.mkdir("./Models")


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
                    if params.policy_type == "normal":
                        next_state, reward, done, _ = sim.env.step(action)
                    elif params.policy_type == "beta":
                        if params.env_name == "Pendulum-v0":
                            next_state, reward, done, _ = sim.env.step(2 * (2 * action - 1))
                        elif params.env_name == "CartPoleContinuous-v0":
                            print("test")
                            next_state, reward, done, _ = sim.env.step(2 * action - 1)
                    total_reward += reward
                    state = next_state
                    if done:
                        # print(total_reward)
                        average_tot_score += total_reward
                        break
            env.close()
            return average_tot_score / nb_evals

        if params.policy_type == "normal":
            policy = NormalPolicy(env.observation_space.shape[0], 32, 64, 1,params.lr_actor)
        if params.policy_type == "beta":
            print("test")
            policy = BetaPolicy(env.observation_space.shape[0], 32, 64, 1,params.lr_actor)
        policy.set_weights(weights)
        workers = min(16, os.cpu_count() + 4)
        evals = int(params.nb_evals / workers)
        sim_list = []
        for i in range(workers):
            sim_list.append(Simulator(params))
        futures = [eval.remote(params, evals, sim) for sim in sim_list]
        returns = ray.get(futures)
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
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
                if done:
                    average_tot_score += total_reward / args.nb_evals
                    break
        return average_tot_score


def load_policies(folder):
    """
    Sort the policies and add colors to each method
    :param: folder : name of the folder containing policies
    Output : array of policies sorted and array of colors
    """
    listdir = os.listdir(folder)
    policies = []
    listdir.sort(key=lambda x: x.split('#')[3])
    colors = []
    print("\nPolices loaded :")
    for policy_file in listdir:
        if policy_file.split('#')[1] == 'PG':
            colors.append("#ff7f0e")
        if policy_file.split('#')[1] == 'CEM':
            colors.append("#d62728")
        pw = PolicyWrapper(GenericNet(), 0, "", "", "", 0)
        policy, _ = pw.load(directory + policy_file)
        policy = policy.get_weights()
        policies.append(policy)
    print("\n")
    env = (policy_file.split('#')[0]).split('/')[-1]
    policy = policy_file.split('#')[5]
    max_episode_steps = policy_file.split('#')[6]
    return policies, colors[1:], policy, env, max_episode_steps


def compute_vignette(args, env, policies, colors):
    if len(np.shape(policies)) > 1:
        theta0 = policies[0]
    else:
        theta0 = policies
        policies = [policies]
    num_params = len(theta0)
    base_vect = theta0
    try:
        D = getDirectionsMuller(args.nb_lines, num_params)
    except Exception as e:
        print("/Models empty (Policies needed to compute a vignette)")
        sys.exit()
    # Compute fitness over these directions :
    policy = policies[0]

    # Change which model to load
    filename = args.filename

    # Load the model

    # Get the new parameters
    theta0 = policy
    base_vect = theta0
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
    indicesPolicies = [
        copyD.index(list(direction)) for direction in policyDirection
    ]
    del copyD

    # Evaluate the Model : mean, std
    print("Evaluating the model...")
    init_score = evaluate_policy(args, env, policy)
    print("Model initial fitness : " + str(init_score))

    # Study the geometry around the model
    print("Starting study around the model...")

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
                                colors=colors,
                                env=args.env_name,
                                policy=args.policy_type,
                                title=args.title)
    for step in range(0, len(D)):
        # Get the direction
        d = D[step]
        print("\nDirection ", step + 1, "/", len(D))
        # New parameters following the direction
        #	Changing the range and step of the Vignette if the optional input policies are beyond that range
        if len(policyDistance) > 0:
            min_dist, max_dist = (args.minalpha,
                                  max(max(policyDistance), args.maxalpha))
            if max(max(policyDistance),
                   args.maxalpha) == max(policyDistance):
                print(
                    "Changing the range to reach the input policies to " +
                    str(max(policyDistance)) + " instead of " +
                    str(args.maxalpha))
        else:
            min_dist = args.minalpha
            max_dist = args.maxalpha
        step_dist = args.stepalpha * (max_dist - min_dist) / (
            args.maxalpha - args.minalpha)
        newVignette.stepalpha = step_dist
        # 	Sampling new models' parameters following the direction
        theta_plus, theta_minus = getPointsDirection(
            theta0, num_params, min_dist, max_dist, step_dist, d)

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

    env.close()


if __name__ == '__main__':

    args = get_args()
    create_data_folders()
    directory = os.getcwd() + '/Models/'
    policies, colors, policy_name, env_name, max_episode_steps = load_policies(
        directory)
    args.env_name = env_name
    args.policy_type = policy_name
    args.max_episode_steps = int(max_episode_steps)
    print(args)
    env = make_env(args.env_name, args.policy_type, args.max_episode_steps,
                   args.env_obs_space_name)
    compute_vignette(args, env, policies, colors)
