import numpy as np
from itertools import count
from batch import Episode, Batch
from environment import make_env
from algo import Algo
from visu.visu_trajectories import plot_trajectory
from visu.visu_weights import plot_weight_histograms, plot_normal_histograms
import math
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

def make_simu_from_params(params):
    """
    Creates the environment, adding the required wrappers
    :param params: the hyper-parameters of the run, specified in arguments.py or in the command line
    :return: a simulation object
    """
    env_name = params.env_name
    env = make_env(env_name, params.policy_type, params.max_episode_steps, params.env_obs_space_name)
    return Simu(env, env_name)


def make_simu_from_wrapper(pw, params):
    """
    Creates the environment, adding the required wrappers
    Used when loading an agent from an external file, through a policy wrapper
    :param pw: the policy wrapper specifying the environment
    :param params: the hyper-parameters of the run, specified in arguments.py or in the command line
    :return: a simulation object
    """
    env_name = pw.env_name
    params.env_name = env_name
    env = make_env(env_name, params.policy_type, params.max_episode_steps, params.env_obs_space_name)
    return Simu(env, env_name)


class Simu:
    """
    The class implements the interaction between the agent represented by its policy and the environment
    """
    def __init__(self, env, env_name):
        self.cpt = 0
        self.best_reward = -1e38
        self.env = env
        self.env_name = env_name
        self.obs_size = env.observation_space.shape[0]
        self.discrete = not env.action_space.contains(np.array([0.5]))
        self.list_weights = None
        self.best_weights = None
        self.best_weights_idx = None
        self.list_rewards = None

    def reset(self, render):
        """
        Reinitialize the state of the agent in the environment
        :param render: whether the step is displayed or not (True or False)
        :return: the new state
        """
        state = self.env.reset()
        if render:
            self.env.render(mode='rgb_array')
        return state

    def take_step(self, state, action, episode, render=False):
        """
        Perform one interaction step between the agent and the environment
        :param state: the current state of the agent
        :param action: the action selected by the agent
        :param episode: the structure into which the resulting sample will be added
        :param render: whether the step is displayed or not (True or False)
        :return: the resulting next state, reward, and whether the episode ended
        """
        next_state, reward, done, _ = self.env.step(action)
        if render:
            self.env.render(mode='rgb_array')
        episode.add(state, action, reward, done, next_state)
        return next_state, reward, done

    def evaluate_episode(self, policy, deterministic, params, render=False):
        """
         Perform an episode using the policy parameter and return the obtained reward
         Used to evaluate an already trained policy, without storing data for further training
         :param policy: the policy controlling the agent
         :param deterministic: whether the evaluation should use a deterministic policy or not
         :param render: whether the episode is displayed or not (True or False)
         :return: the total reward collected during the episode
         """
        average_reward=0
        for i in range(params.nb_evals):
            total_reward=0
            done=False
            state = self.reset(render)
            while not done:
                action = policy.select_action(state, deterministic)
            # print("action", action)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
            average_reward += total_reward
        average_reward /= params.nb_evals
        return average_reward





    def train_cem(self,pw, params,policy) -> None:
        """
        The main function to learn policies using the Cross Enthropy Method
        return: nothing
        """

        #Initialize variables
        self.list_weights = np.zeros((int(params.nb_cycles),policy.get_weights_dim(False)))
        self.best_weights = np.zeros(policy.get_weights_dim(False))
        self.list_rewards = np.zeros((int(params.nb_cycles)))

        print("Shape of weights vector is: ", np.shape(self.best_weights))

        #Init the first centroid randomly
        centroid = np.array(params.sigma*np.random.randn(policy.get_weights_dim(False)))
        #Set the weights with this random centroid
        policy.set_weights(centroid, False)
        #Init the noise matrix
        noise=np.diag(np.ones(policy.get_weights_dim(False))*params.sigma)
        #Init the covariance matrix
        var=np.diag(np.ones(policy.get_weights_dim(False))*np.var(centroid))+noise
        #Init the rng
        rng = np.random.default_rng()
        #Training Loop
        for cycle in range(params.nb_cycles):
            rewards = np.zeros(params.population)
            weights=rng.multivariate_normal(centroid, var, params.population)
            for p in range(params.population):
                policy.set_weights(weights[p], False)
                batch=self.make_monte_carlo_batch(params.nb_trajs, params.render, policy, True)
                rewards[p] = batch.train_policy_cem(policy, params.bests_frac)

            elites_nb = int(params.elites_frac * params.population)
            elites_idxs = rewards.argsort()[-elites_nb:]
            elites_weights = [weights[i] for i in elites_idxs]
                #update the best weights
            centroid = np.array(elites_weights).mean(axis=0)
            var = np.cov(elites_weights,rowvar=False)+noise

                #print(best_weights)
                # policy evaluation part
            policy.set_weights(centroid, False)

            self.list_weights[cycle] = policy.get_weights()

            # policy evaluation part
            if ((cycle%params.eval_freq)==0):
                total_reward = self.evaluate_episode(policy, params.deterministic_eval, params)
                print(total_reward)
                #write and store reward_
                self.env.write_reward(cycle,total_reward)
                self.list_rewards[cycle] = total_reward
                # plot_trajectory(batch2, self.env, cycle+1)

            # save best reward agent (no need for averaging if the policy is deterministic)
            if self.best_reward < total_reward:
                self.best_reward = total_reward
                self.best_weights = self.list_weights[cycle]
                self.best_weights_idx = cycle
            #Save the best policy obtained
            if ((cycle%params.save_freq)==0):
                pw.save(method = "CEM", cycle = cycle,score = total_reward)

        pw.rename_best(method="CEM",best_cycle=self.best_weights_idx,score=self.best_reward)




    def train_pg(self, pw, params, policy, critic, policy_loss_file, critic_loss_file, study_name, beta=0) -> None:
        """
        The main function for training and evaluating a policy
        Repeats training and evaluation params.nb_cycles times
        Stores the value and policy losses at each cycle
        When the reward is greater than the best reward so far, saves the corresponding policy
        :param pw: a policy wrapper, used to save the best policy into a file
        :param params: the hyper-parameters of the run, specified in arguments.py or in the command line
        :param policy: the trained policy
        :param critic: the corresponding critic (not always used)
        :param policy_loss_file: the file to record successive policy loss values
        :param critic_loss_file: the file to record successive critic loss values
        :param study_name: the name of the studied gradient algorithm
        :param beta: a specific parameter for beta-parametrized values
        :return: nothing
        """

        #Initialize variables
        self.list_weights = np.zeros((int(params.nb_cycles),policy.get_weights_dim(False)))
        self.best_weights = np.zeros(policy.get_weights_dim(False))
        self.list_rewards = np.zeros((int(params.nb_cycles)))

        print("Shape of weights vector is: ", np.shape(self.best_weights))


        for cycle in range(params.nb_cycles):
            batch = self.make_monte_carlo_batch(params.nb_trajs, params.render, policy)

            # Update the policy
            batch2 = batch.copy_batch()
            algo = Algo(study_name, params.critic_estim_method, policy, critic, params.gamma, beta, params.nstep)
            algo.prepare_batch(batch)
            policy_loss = batch.train_policy_td(policy)

            # Update the critic
            assert params.critic_update_method in ['batch', 'dataset'], 'unsupported critic update method'
            if params.critic_update_method == "dataset":
                critic_loss = algo.train_critic_from_dataset(batch2, params)
            elif params.critic_update_method == "batch":
                critic_loss = algo.train_critic_from_batch(batch2)
            critic_loss_file.write(str(cycle) + " " + str(critic_loss) + "\n")
            policy_loss_file.write(str(cycle) + " " + str(policy_loss) + "\n")

            # add the new weights to the list of weights
            self.list_weights[cycle] = policy.get_weights()

            # policy evaluation part
            if ((cycle%params.eval_freq)==0):
                total_reward = self.evaluate_episode(policy, params.deterministic_eval, params)
                print(total_reward)
                #wrote and store reward
                self.env.write_reward(cycle,total_reward)
                self.list_rewards[cycle] = total_reward
                # plot_trajectory(batch2, self.env, cycle+1)

            # save best reward agent (no need for averaging if the policy is deterministic)
            if self.best_reward < total_reward:
                self.best_reward = total_reward
                self.best_weights = self.list_weights[cycle]
                self.best_weights_idx = cycle
        #Save the best policy obtained
            if ((cycle%params.save_freq)==0):
                pw.save(cycle = cycle,score = total_reward)

        pw.rename_best(method="PG",best_cycle=self.best_weights_idx,best_score=self.best_reward)

    def get_weights_data(self):
        """
        Simple function to get the list of weights obtained during a training.
        return: list of weights, the weights giving the best reward, the vector of rewards,
        and the index of the best weights.
        """
        return self.list_weights,self.best_weights,self.list_rewards,self.best_weights_idx


    def train_on_one_episode(self, policy, deterministic, render=False):
        """
        Perform an episode using the policy parameter and return the corresponding samples into an episode structure
        :param policy: the policy controlling the agent
        :param deterministic: whether the evaluation should use a deterministic policy or not
        :param render: whether the episode is displayed or not (True or False)
        :return: the samples stored into an episode
        """
        episode = Episode()
        state = self.reset(render)
        for t in count():
            action = policy.select_action(state,deterministic)
            next_state, _, done = self.take_step(state, action, episode, render)
            state = next_state

            if done:
                # print("train nb steps:", t)
                return episode

    def make_monte_carlo_batch(self, nb_episodes, render, policy, weights_flag=False, weights=None):
        """
        Create a batch of episodes with a given policy
        Used in Monte Carlo approaches
        :param nb_episodes: the number of episodes in the batch
        :param render: whether the episode is displayed or not (True or False)
        :param policy: the policy controlling the agent
        :return: the resulting batch of episodes
        """
        if weights_flag == True:
            batch = Batch(weights)
        else:
            batch = Batch()
        self.env.set_reward_flag(False)
        self.env.set_duration_flag(False)
        for e in range(nb_episodes):
            episode = self.train_on_one_episode(policy, False, render)
            batch.add_episode(episode)
        return batch
                #For saving top 10 policies obtained
                # self.evaluate_episode(policy, params.deterministic_eval)
                # top_ten_policies= [init_weights for i in range(10)]
                # score=self.evaluate_episode(policy, params.deterministic_eval)
                # top_ten_scores=[score for i in range(10)]
                # print(np.shape(top_ten_policies))
                # print(np.shape(top_ten_scores))


                #policy.set_weights(init_weights[0,:], False)
    # def evaluate_episode_clean(self, policy, deterministic, render=False):
    #     """
    #      Perform an episode using the policy parameter and return the obtained reward
    #      Used to evaluate an already trained policy, without storing data for further training
    #      :param policy: the policy controlling the agent
    #      :param deterministic: whether the evaluation should use a deterministic policy or not
    #      :param render: whether the episode is displayed or not (True or False)
    #      :return: the total reward collected during the episode
    #      """
    #     state = self.reset(render)
    #     total_reward=0
    #     for t in count():
    #         action = policy.select_action(state, deterministic)
    #         # print("action", action)
    #         next_state, reward, done, _ = self.env.step(action)
    #         total_reward += reward
    #         state = next_state
    #
    #         if done:
    #             return total_reward

    def train(self, pw,params, policy, critic, policy_loss_file, critic_loss_file, study_name, beta=0, is_cem=False):
        all_weights=np.zeros((int(params.nb_cycles),policy.get_weights_dim(False)))
        print(np.shape(all_weights))
        all_rewards=np.zeros(params.nb_cycles)
        best_reward=-np.inf
        best_weights=np.zeros(policy.get_weights_dim(False))
        fixed=params.fix_layers
        idx_best=0
        if is_cem == False:
            if fixed:
                print(fixed)
                fc1_w, fc1_b, fc2_w, fc2_b = policy.get_weights_pg()
                # print(fc1_w)
                # print(policy.test())

        if is_cem == True:
            all_weights=np.zeros((int(params.nb_cycles),policy.get_weights_dim(fixed)))
            best_weights=np.zeros(policy.get_weights_dim(fixed))
            #random init of the neural network.
            #so far, all the layers are initialized with the same gaussian.
            init_weights = np.array(params.sigma*np.random.randn(policy.get_weights_dim(False)))
            #print(np.shape(init_weights))
            #start_weights=np.array(3*np.random.randn(policy.get_weights_dim(False)))
            policy.set_weights(init_weights, False)

            print(fixed)
            #print(params.fix_layers)
            #print(policy.get_weights_dim(params.fix_layers))
            study = params.study_name
            noise=np.diag(np.ones(policy.get_weights_dim(fixed))*params.sigma)
            #print(np.shape(noise))
            #var=np.cov(init_weights[:,-policy.get_weights_dim(fixed):],rowvar=False) + noise
            #mu=init_weights[:,-policy.get_weights_dim(fixed):].mean(axis=0)

            var=np.diag(np.ones(policy.get_weights_dim(fixed))*np.var(init_weights))+noise
            print(np.shape(var))
            mu=init_weights[-policy.get_weights_dim(fixed):]
            print(np.shape(mu))
            rng = np.random.default_rng()

            #we can draw the last layer from a different gaussian
            #mu=params.sigma_bis*np.random.randn(policy.get_weights_dim(params.fix_layers))
        for cycle in range(params.nb_cycles):
            if is_cem == True:
                rewards = np.zeros(params.population)
                weights=rng.multivariate_normal(mu, var, params.population)
                for p in range(params.population):
                    policy.set_weights(weights[p], fixed)
                    batch=self.make_monte_carlo_batch(params.nb_trajs_cem, params.render, policy, True)
                    rewards[p] = batch.train_policy_cem(policy, params.bests_frac)

                elites_nb = int(params.elites_frac * params.population)
                elites_idxs = rewards.argsort()[-elites_nb:]
                elites_weights = [weights[i] for i in elites_idxs]
                #update the best weights
                mu = np.array(elites_weights).mean(axis=0)
                var = np.cov(elites_weights,rowvar=False)+noise

                #print(best_weights)
                # policy evaluation part
                policy.set_weights(mu, fixed)

                total_reward = self.evaluate_episode(policy, params.deterministic_eval)
                if total_reward>best_reward:
                    best_weights=mu
                    best_reward=total_reward
                    idx_best=cycle
                all_rewards[cycle]=total_reward
                # if total_reward>np.min(top_ten_scores):
                #     temp_min=np.argmin(top_ten_scores)
                #     top_ten_scores[temp_min]=total_reward
                #     top_ten_policies[temp_min]=mu

                # Update the file for the plot
                reward_file = policy_loss_file
                reward_file.write(str(cycle) + " " + str(total_reward) + "\n")
                # if (cycle+1)%3==0:
                    # all_weights[int((cycle+1)/3)-1]=mu
                all_weights[cycle]=mu

            elif is_cem == False:
                batch = self.make_monte_carlo_batch(params.nb_trajs_pg, params.render, policy)

                # Update the policy
                batch2 = batch.copy_batch()
                algo = Algo(study_name, params.critic_estim_method, policy, critic, params.gamma, beta, params.nstep)
                algo.prepare_batch(batch)
                policy_loss = batch.train_policy_td(policy)
                # print(np.shape(policy.get_gradient()))
                # if (cycle+1)%3==0:
                #     all_weights[int((cycle+1)/3)-1]=policy.get_weights_as_numpy()
                all_weights[cycle]=policy.get_weights_as_numpy()
                # grad=policy.get_gradient()
                # print(np.linalg.norm(grad-all_weights[cycle])*params.lr_actor)
                #print(policy_loss)

                # Update the critic
                assert params.critic_update_method in ['batch', 'dataset'], 'unsupported critic update method'
                if params.critic_update_method == "dataset":
                    critic_loss = algo.train_critic_from_dataset(batch2, params)
                elif params.critic_update_method == "batch":
                    critic_loss = algo.train_critic_from_batch(batch2)
                critic_loss_file.write(str(cycle) + " " + str(critic_loss) + "\n")
                policy_loss_file.write(str(cycle) + " " + str(policy_loss) + "\n")
                plot_trajectory(batch2, self.env, cycle+1)

                # policy evaluation part
                if fixed:
                    policy.set_weights_pg(fc1_w, fc1_b, fc2_w, fc2_b)
                total_reward = self.evaluate_episode(policy, params.deterministic_eval)
                all_rewards[cycle]=total_reward


                # if cycle > 0:
                #     distance=np.linalg.norm(all_weights[cycle]-all_weights[cycle-1])
                #     print("distance between pol"+str(cycle-1)+" and pol"+str(cycle)+" : "+str(distance))
                if total_reward>best_reward:
                    best_weights=policy.get_weights_as_numpy()
                    best_reward=total_reward
                    idx_best=cycle
            print(total_reward)
        # X_embedded = TSNE(n_components=2).fit_transform(all_cem_weights)
        # # print(np.shape(X_embedded))
        # # print(X_embedded)
        # plt.scatter(*zip(*X_embedded))
        return all_weights,best_weights,all_rewards,idx_best
