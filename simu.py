import numpy as np
from itertools import count
from batch import Episode, Batch
from environment import make_env
from algo import Algo
from algoCEM import AlgoCEM
from visu.visu_trajectories import plot_trajectory
from visu.visu_weights import plot_weight_histograms, plot_normal_histograms
import math
import os

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

    def evaluate_episode(self, policy, deterministic, render=False):
        """
         Perform an episode using the policy parameter and return the obtained reward
         Used to evaluate an already trained policy, without storing data for further training
         :param policy: the policy controlling the agent
         :param deterministic: whether the evaluation should use a deterministic policy or not
         :param render: whether the episode is displayed or not (True or False)
         :return: the total reward collected during the episode
         """
        self.env.set_reward_flag(True)
        self.env.set_duration_flag(True)
        state = self.reset(render)
        total_reward = 0
        for t in count():
            action = policy.select_action(state, deterministic)
            # print("action", action)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            state = next_state

            if done:
                return total_reward

    def evaluate_episode_CEM(self, policy, deterministic, render=False):
        """
         Perform an episode using the policy parameter and return the obtained reward
         Used to evaluate an already trained policy, without storing data for further training
         :param policy: the policy controlling the agent
         :param deterministic: whether the evaluation should use a deterministic policy or not
         :param render: whether the episode is displayed or not (True or False)
         :return: the total reward collected during the episode
         """
        self.env.set_reward_flag(False)
        self.env.set_duration_flag(False)
        state = self.reset(render)
        total_reward = 0
        #final_t=0
        for t in count():
            action = policy.select_action(state, deterministic)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            state = next_state

            if done:
                return total_reward

    def trainCEM(self, pw, params, policy, policy_loss_file, study_name, beta=0, fix_layers=True) -> None:
        #random init of the neural network.
        init_weights = params.sigma*np.random.randn(policy.get_weights_dim())
        policy.set_weights(init_weights)

        #make data files to plot the sum of reward at each episode.
        path = os.getcwd() + "/data/save"
        study = params.study_name
        total_reward_file = open(path + "/total_reward_" + study + '_' + params.env_name + '.txt', 'w')
        #best_reward_file = open(path + "/best_reward_" + study + '_' + params.env_name + '.txt', 'w')

        #We learn all the weights of the neural network
        if not fix_layers:
            best_weights=init_weights
            for cycle in range(params.nb_cycles):
                batches = []
                rewards = np.zeros(params.population)
                weights = []

                for p in range(params.population):
                    weights.append(best_weights + (params.sigma*np.random.randn(policy.get_weights_dim())))
                for p in range(params.population):
                    policy.set_weights(weights[p])
                    batches.append(self.make_monte_carlo_batch(params.nb_trajs, params.render, policy, True))

                    # Update the policy
                for p in range(params.population):
                    rewards[p] = batches[p].train_policy_cem(policy, params.bests_frac)

                elites_nb = int(params.elites_frac * params.population)
                elites_idxs = rewards.argsort()[-elites_nb:]
                elites_weights = [weights[i] for i in elites_idxs]

                #update the best weights
                best_weights = np.array(elites_weights).mean(axis=0)

                # policy evaluation part
                policy.set_weights(best_weights)
                total_reward = self.evaluate_episode_CEM(policy, params.deterministic_eval)
                total_reward_file.write(str(cycle) + ' ' + str(total_reward) + '\n')


                # save best reward agent (no need for averaging if the policy is deterministic)
                print("best :", self.best_reward, "| new :", total_reward, "| test :", self.best_reward < total_reward)
                if self.best_reward < total_reward:
                    self.best_reward = total_reward
                    #best_reward_file.write(str(cycle) + ' ' + str(self.best_reward) + '\n')
                    #best_weights = mean_elites_weights
                    print("Weights changed :", self.best_reward)
                pw.save(self.best_reward)
            total_reward_file.close()
                #best_reward_file.close()

            #We learn only the weights of the last layer.
        else:
            print("ok")
            best_weights=init_weights[-policy.get_last_layer_dim():]
            for cycle in range(params.nb_cycles):
                batches = []
                rewards = np.zeros(params.population)
                weights = []

                for p in range(params.population):
                    weights.append(best_weights + (params.sigma*np.random.randn(policy.get_last_layer_dim())))
                for p in range(params.population):
                    policy.set_last_layer_weights(weights[p])
                    batches.append(self.make_monte_carlo_batch(params.nb_trajs, params.render, policy, True))

                # Update the policy
                for p in range(params.population):
                    rewards[p] = batches[p].train_policy_cem(policy, params.bests_frac)

                elites_nb = int(params.elites_frac * params.population)
                elites_idxs = rewards.argsort()[-elites_nb:]
                elites_weights = [weights[i] for i in elites_idxs]

                #update the best weight
                best_weights = np.array(elites_weights).mean(axis=0)

                # policy evaluation part
                policy.set_last_layer_weights(best_weights)
                total_reward = self.evaluate_episode_CEM(policy, params.deterministic_eval)
                total_reward_file.write(str(cycle) + ' ' + str(total_reward) + '\n')

                # save best reward agent (no need for averaging if the policy is deterministic)
                print("best :", self.best_reward, "| new :", total_reward, "| test :", self.best_reward < total_reward)
                if self.best_reward < total_reward:
                    self.best_reward = total_reward
                    #best_reward_file.write(str(cycle) + ' ' + str(self.best_reward) + '\n')
                    print("Weights changed :", self.best_reward)
                pw.save(self.best_reward)
            total_reward_file.close()
                #best_reward_file.close()


    def trainCEMbis(self, pw, params, policy, policy_loss_file, study_name, beta=0) -> None:
        sigma=params.sigma
        n_elite = int(params.elites_frac * params.population)
        best_weight = params.sigma*np.random.randn(policy.get_weights_dim())
        path = os.getcwd() + "/data/save"
        study = params.study_name
        total_reward_file = open(path + "/total_reward_" + study + '_' + params.env_name + '.txt', 'w')
        best_reward_file = open(path + "/best_reward_" + study + '_' + params.env_name + '.txt', 'w')

        for cycle in range(params.nb_cycles):
            weights_pop = [best_weight + (sigma*np.random.randn(policy.get_weights_dim())) for i in range(params.population)]
            #batch = self.make_monte_carlo_batch(params.nb_trajs, params.render, policy)
            #algo = AlgoCEM(study_name, policy, params.gamma, beta, params.nstep)
            #algo.prepare_batch(batch)
            #batch2 = batch.copy_batch()
            mean_total_rewards=np.zeros(params.population)
            for i in range(params.population):
                average_tot_sum_on_traj=0
                for j in range(params.nb_trajs):
                    average_tot_sum_on_traj+=self.evaluate_episode_CEM(policy, params.deterministic_eval, best_weight)
                mean_total_rewards[i]=average_tot_sum_on_traj/params.nb_trajs
            elite_idxs = mean_total_rewards.argsort()[-n_elite:]


            elite_weights = [weights_pop[i] for i in elite_idxs]
            #print(elite_weights)
            best_weight = np.array(elite_weights).mean(axis=0)

            # policy evaluation part
            total_reward = self.evaluate_episode_CEM(policy, params.deterministic_eval, best_weight)

            total_reward_file.write(str(cycle) + ' ' + str(total_reward) + '\n')

            # save best reward agent (no need for averaging if the policy is deterministic)
            print("best :", self.best_reward, "| new :", total_reward, "| test :", self.best_reward < total_reward)
            if self.best_reward < total_reward:
                self.best_reward = total_reward
                #best_reward_file.write(str(cycle) + ' ' + str(self.best_reward) + '\n')
            pw.save(self.best_reward)
        total_reward_file.close()
        #best_reward_file.close()

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
            action = policy.select_action(state, deterministic)
            next_state, _, done = self.take_step(state, action, episode, render)
            state = next_state

            if done:
                # print("train nb steps:", t)
                return episode

    def train(self, pw, params, policy, critic, policy_loss_file, critic_loss_file, study_name, beta=0) -> None:
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
        for cycle in range(params.nb_cycles):
            batch = self.make_monte_carlo_batch(params.nb_trajs, params.render, policy)

            # Update the policy
            batch2 = batch.copy_batch()
            algo = Algo(study_name, params.critic_estim_method, policy, critic, params.gamma, beta, params.nstep)
            algo.prepare_batch(batch)
            policy_loss = batch.train_policy_td(policy)
            #print(policy_loss)

            # Update the critic
            assert params.critic_update_method in ['batch', 'dataset'], 'unsupported critic update method'
            if params.critic_update_method == "dataset":
                critic_loss = algo.train_critic_from_dataset(batch2, params)
            elif params.critic_update_method == "batch":
                critic_loss = algo.train_critic_from_batch(batch2)
            critic_loss_file.write(str(cycle) + " " + str(critic_loss) + "\n")
            policy_loss_file.write(str(cycle) + " " + str(policy_loss) + "\n")

            # policy evaluation part
            total_reward = self.evaluate_episode(policy, params.deterministic_eval)
            print(total_reward)
            # plot_trajectory(batch2, self.env, cycle+1)

            # save best reward agent (no need for averaging if the policy is deterministic)
            #print(#self.best_reward)
            if self.best_reward < total_reward:
                self.best_reward = total_reward
                pw.save(self.best_reward)

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
