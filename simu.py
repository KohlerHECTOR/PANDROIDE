import os
import numpy as np
from itertools import count
from batch import Episode, Batch
from environment import make_env
from slowBar import SlowBar
import ray
import gym


def get_starting_weights(pw):
    directory = os.getcwd() + '/starting_policy/'
    listdir = os.listdir(directory)
    policies = []
    for policy_file in listdir:
        policy, _ = pw.load(directory + policy_file)
        policy = policy.get_weights()
        policies.append(policy)
    return policies[0]


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

    class Simulator(object):
        def __init__(self, env_name):
            self.env = gym.make(env_name)
            self.env.reset()

        def step(self, action):
            return self.env.step(action)

    def evaluate_episode(self, policy, deterministic, params, render=False):
        """
         Perform an episode using the policy parameter and return the obtained reward
         Used to evaluate an already trained policy, without storing data for further training
         :param policy: the policy controlling the agent
         :param deterministic: whether the evaluation should use a deterministic policy or not
         :param params: parameters of the run
         :param render: whether the episode is displayed or not (True or False)
         :return: the total reward collected during the episode
         """
        average_reward = 0
        for i in range(params.nb_evals):
            total_reward = 0
            done = False
            state = self.reset(render)
            while not done:
                action = policy.select_action(state, deterministic)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
            average_reward += total_reward
        average_reward /= params.nb_evals
        return average_reward

    def train_cem(self, pw, params, policy) -> None:
        """
        The main function to learn policies using the Cross Enthropy Method
        return: nothing
        """
        # Initialize variables
        self.list_weights = []
        self.best_weights = np.zeros(policy.get_weights_dim())
        self.list_rewards = np.zeros((int(params.nb_cycles)))
        self.best_reward = -1e38
        self.best_weights_idx = 0


        print("Shape of weights vector is: ", policy.get_weights_dim())

        if params.start_from_policy:
            starting_weights = get_starting_weights(pw)
            centroid = starting_weights

        # Init the first centroid
        elif params.start_from_same_policy:
            centroid = policy.get_weights()
        else:
            centroid = np.random.rand(policy.get_weights_dim())

        policy.set_weights(centroid)
        initial_score = self.evaluate_episode(policy, params.deterministic_eval, params)
        pw.save(cycle=0, method='CEM', score=initial_score)
        self.env.write_reward(cycle=0, reward=initial_score)
        self.list_weights.append(centroid)
        # Set the weights with this random centroid

        # Init the noise matrix
        noise = np.diag(np.ones(policy.get_weights_dim()) * params.sigma)
        # Init the covariance matrix
        var = np.diag(np.ones(policy.get_weights_dim()) * np.var(centroid)) + noise
        # var=np.diag(np.ones(policy.get_weights_dim())*params.lr_actor**2)+noise
        # Init the rng
        rng = np.random.default_rng()
        # Training Loop
        with SlowBar('Performing a repetition of CEM', max=params.nb_cycles) as bar:
            for cycle in range(params.nb_cycles):
                rewards = np.zeros(params.population)
                weights = rng.multivariate_normal(centroid, var, params.population)
                for p in range(params.population):
                    policy.set_weights(weights[p])
                    batch = self.make_monte_carlo_batch(params.nb_trajs, params.render, policy, True)
                    rewards[p] = batch.train_policy_cem(policy, params.bests_frac)

                elites_nb = int(params.elites_frac * params.population)
                elites_idxs = rewards.argsort()[-elites_nb:]
                elites_weights = [weights[i] for i in elites_idxs]
                # update the best weights
                centroid = np.array(elites_weights).mean(axis=0)
                var = np.cov(elites_weights, rowvar=False) + noise
                self.env.write_cov(cycle, np.linalg.norm(var))
                distance = np.linalg.norm(centroid - self.list_weights[-1])
                self.env.write_distances(cycle, distance)

                # policy evaluation part
                policy.set_weights(centroid)

                self.list_weights.append(policy.get_weights())
                self.write_angles_global(cycle)

                # policy evaluation part
                if (cycle % params.eval_freq) == 0:
                    total_reward = self.evaluate_episode(policy, params.deterministic_eval, params)
                    # write and store reward
                    self.env.write_reward(cycle + 1, total_reward)
                    self.list_rewards[cycle] = total_reward

                # Save best reward agent (no need for averaging if the policy is deterministic)
                if self.best_reward < total_reward:
                    self.best_reward = total_reward
                    self.best_weights = self.list_weights[-1]
                    self.best_weights_idx = cycle
                # Save the best policy obtained
                if (cycle % params.save_freq) == 0:
                    pw.save(method="CEM", cycle=cycle + 1, score=total_reward)
                bar.next()

        # pw.rename_best(method="CEM",best_cycle=self.best_weights_idx,best_score=self.best_reward)
        print("Best reward: ", self.best_reward)
        print("Best reward iter: ", self.best_weights_idx)

    def train_pg(self, pw, params, policy, policy_loss_file) -> None:
        """
        The main function for training and evaluating a policy
        Repeats training and evaluation params.nb_cycles times
        Stores the value and policy losses at each cycle
        When the reward is greater than the best reward so far, saves the corresponding policy
        :param pw: a policy wrapper, used to save the best policy into a file
        :param params: the hyper-parameters of the run, specified in arguments.py or in the command line
        :param policy: the trained policy
        :param policy_loss_file: the file to record successive policy loss values
        :return: nothing
        """
        # Initialize variables
        self.list_weights = []
        self.best_weights = np.zeros(policy.get_weights_dim())
        self.list_rewards = np.zeros((int(params.nb_cycles)))
        self.best_reward = -1e38
        self.best_weights_idx = 0

        self.list_weights.append(policy.get_weights())

        if params.start_from_policy:
            starting_weights = get_starting_weights(pw)
            policy.set_weights(starting_weights)


        print("Shape of weights vector is: ", np.shape(self.best_weights))
        initial_score = self.evaluate_episode(policy, params.deterministic_eval, params)
        pw.save(cycle=0, score=initial_score)
        self.env.write_reward(cycle=0, reward=initial_score)
        with SlowBar('Performing a repetition of PG', max=params.nb_cycles) as bar:
            for cycle in range(params.nb_cycles):
                batch = self.make_monte_carlo_batch(params.nb_trajs, params.render, policy)
                batch.sum_rewards()
                policy_loss = batch.train_policy_td(policy)
                # self.env.write_gradients(gradient_angles,cycle)
                policy_loss_file.write(str(cycle) + " " + str(policy_loss) + "\n")

                # add the new weights to the list of weights
                self.list_weights.append(policy.get_weights())
                distance = np.linalg.norm(self.list_weights[-1] - self.list_weights[-2])
                self.env.write_distances(cycle, distance)
                self.write_angles_global(cycle)

                # policy evaluation part
                if (cycle % params.eval_freq) == 0:
                    total_reward = self.evaluate_episode(policy, params.deterministic_eval, params)
                    # wrote and store reward
                    self.env.write_reward(cycle + 1, total_reward)
                    self.list_rewards[cycle] = total_reward
                    # plot_trajectory(batch2, self.env, cycle+1)

                # save best reward agent (no need for averaging if the policy is deterministic)
                if self.best_reward < total_reward:
                    self.best_reward = total_reward
                    self.best_weights = self.list_weights[-1]
                    self.best_weights_idx = cycle
                # Save the best policy obtained
                if (cycle % params.save_freq) == 0:
                    pw.save(cycle=cycle + 1, score=total_reward)
                bar.next()

        # pw.rename_best(method="PG",best_cycle=self.best_weights_idx,best_score=self.best_reward)
        print("Best reward: ", self.best_reward)
        print("Best reward iter: ", self.best_weights_idx)


    def write_angles_global(self, cycle):
        if cycle == 1:
            unit_vec_1 = self.list_weights[0] / np.linalg.norm(self.list_weights[0])
            norm = np.linalg.norm(self.list_weights[1] - self.list_weights[0])
            unit_vec_2 = (self.list_weights[1] - self.list_weights[0]) / norm
            angle = np.dot(unit_vec_1, unit_vec_2)
            self.env.write_angles(cycle, angle)
        elif cycle > 1:
            norm1 = np.linalg.norm(self.list_weights[cycle - 1] - self.list_weights[cycle - 2])
            norm2 = np.linalg.norm(self.list_weights[cycle] - self.list_weights[cycle - 1])
            unit_vec_1 = (self.list_weights[cycle - 1] - self.list_weights[cycle - 2]) / norm1
            unit_vec_2 = (self.list_weights[cycle] - self.list_weights[cycle - 1]) / norm2
            angle = np.dot(unit_vec_1, unit_vec_2)
            self.env.write_angles(cycle, angle)

    def get_weights_data(self):
        """
        Simple function to get the list of weights obtained during a training.
        return: list of weights, the weights giving the best reward, the vector of rewards,
        and the index of the best weights.
        """
        return self.list_weights, self.best_weights, self.list_rewards, self.best_weights_idx

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
        for _ in count():
            action = policy.select_action(state, deterministic)
            next_state, _, done = self.take_step(state, action, episode, render)
            state = next_state

            if done:
                return episode

    def make_monte_carlo_batch(self, nb_episodes, render, policy, weights_flag=False, weights=None):
        """
        Create a batch of episodes with a given policy
        Used in Monte Carlo approaches
        :param nb_episodes: the number of episodes in the batch
        :param render: whether the episode is displayed or not (True or False)
        :param policy: the policy controlling the agent
        :param weights_flag
        :param weights
        :return: the resulting batch of episodes
        """
        if weights_flag:
            batch = Batch(weights)
        else:
            batch = Batch()
        self.env.set_reward_flag(False)
        self.env.set_duration_flag(False)
        for e in range(nb_episodes):
            episode = self.train_on_one_episode(policy, False, render)
            batch.add_episode(episode)
        return batch
