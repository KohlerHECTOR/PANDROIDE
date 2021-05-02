import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal
from policies.generic_net import GenericNet
from copy import deepcopy


LOG_STD_MAX = 2
LOG_STD_MIN = -20


def log_prob(normal_distribution, action):
    """
    Compute the log probability of an action from a Gaussian distribution
    This function performs the necessary corrections in the computation
    to take into account the presence of tanh in the squashed Gaussian function
    see https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
    for details
    :param normal_distribution: the Gaussian distribution used to draw an action
    :param action: the action whose probability must be estimated
    :return: the obtained log probability
    """
    logp_pi = normal_distribution.log_prob(action).sum(axis=-1)
    val = func.softplus(-2 * action)
    logp_pi -= (2 * (np.log(2) - action - val)).sum(axis=1)
    return logp_pi


class SquashedGaussianPolicy(GenericNet):
    """
      A policy whose probabilistic output is drawn from a squashed Gaussian function
      """
    def __init__(self, l1, l2, l3, l4, learning_rate=None):
        super(SquashedGaussianPolicy, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(l1, l2)
        self.fc2 = nn.Linear(l2, l3)
        self.fc_mu = nn.Linear(l3, l4)
        self.fc_std = nn.Linear(l3, l4)
        self.tanh_layer = nn.Tanh()
        if (learning_rate != None):
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.s_size = l1
        self.h1_size = l2
        self.h2_size = l3
        self.fc_mu_size = l4
        self.fc_std_size = l4

    def to_numpy(self,var):
        return var.data.numpy()

    def set_weights(self, params):
        """
        Set the params of the network to the given parameters
        """
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            if torch.cuda.is_available():
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()).cuda())
            else:
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def get_weights(self):
        """
        Returns parameters of the actor
        """
        return deepcopy(np.hstack([self.to_numpy(v).flatten() for v in
                                   self.parameters()]))

    def get_grads(self):
        """
        Returns the current gradient
        """
        return deepcopy(np.hstack([self.to_numpy(v.grad).flatten() for v in self.parameters()]))

    def get_weights_dim(self):
        """
        Returns the number of parameters of the network
        """
        return self.get_weights().shape[0]

    def forward(self, state):
        """
        Compute the pytorch tensors resulting from sending a state or vector of states through the policy network
        The obtained tensors can be used to obtain an action by calling select_action
        :param state: the input state(s)
        :return: the resulting pytorch tensor (here the max and standard deviation of a Gaussian probability of action)
        """
        # To deal with numpy's poor behavior for one-dimensional vectors
        # Add batch dim of 1 before sending through the network
        if state.ndim == 1:
            state = np.reshape(state, (1, -1))
        state = torch.from_numpy(state).float()
        state = self.relu(self.fc1(state))
        state = self.relu(self.fc2(state))
        mu = self.fc_mu(state)
        std = self.fc_std(state)
        log_std = torch.clamp(std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    def select_action(self, state, deterministic=False):
        """
        Compute an action or vector of actions given a state or vector of states
        :param state: the input state(s)
        :param deterministic: whether the policy should be considered deterministic or not
        :return: the resulting action(s)
        """
        with torch.no_grad():
            # Forward pass
            mu, std = self.forward(state)
            pi_distribution = Normal(mu, std)

            if deterministic:
                # Only used for evaluating policy at test time.
                pi_action = mu
            else:
                pi_action = pi_distribution.rsample()

            # Finally applies tanh for squashing
            pi_action = torch.tanh(pi_action)
            if len(pi_action) == 1:
                pi_action = pi_action[0]
            return pi_action.data.numpy().astype(float)

    def train_pg(self, state, action, reward):
        """
        Train the policy using a policy gradient approach
        :param state: the input state(s)
        :param action: the input action(s)
        :param reward: the resulting reward
        :return: the loss applied to train the policy
        """
        act = torch.FloatTensor(action)
        rwd = torch.FloatTensor(reward)
        mu, std = self.forward(state)
        # Negative score function x reward
        # loss = -Normal(mu, std).log_prob(action) * reward
        normal_distribution = Normal(mu, std)
        loss = - log_prob(normal_distribution, act).sum(dim=-1) * rwd
        self.update(loss)
        return loss

    def train_regress(self, state, action, estimation_method='mse'):
        """
        Train the policy to perform the same action(s) in the same state(s) using regression
        :param state: the input state(s)
        :param action: the input action(s)
        :param estimation_method: whther we use mse or log_likelihood
        :return: the loss applied to train the policy
        """
        assert estimation_method in ['mse', 'log_likelihood'], 'unsupported estimation method'
        action = torch.FloatTensor(action)
        mu, std = self.forward(state)
        if estimation_method == 'mse':
            loss = func.mse_loss(mu, action)
        else:
            normal_distribution = Normal(mu, std)
            loss = -log_prob(normal_distribution, action.view(-1, 1))
        self.update(loss)
        return loss

    def train_regress_from_batch(self, batch) -> None:
        """
        Train the policy using a policy gradient approach from a full batch of episodes
        :param batch: the batch used for training
        :return: nothing
        """
        for j in range(batch.size):
            episode = batch.episodes[j]
            state = np.array(episode.state_pool)
            action = np.array(episode.action_pool)
            self.train_regress(state, action)

    # def set_weights(self, weights, fix_layers=False):
    #     if fix_layers: # last layers weights
    #         h2_size = self.h2_size
    #         fc_mu_size = self.fc_mu_size
    #         fc_std_size = self.fc_std_size
    #         fc_mu_end= (h2_size*fc_mu_size)+fc_mu_size
    #         fc_mu_W = torch.from_numpy(weights[:(h2_size*fc_mu_size)].reshape(h2_size, fc_mu_size))
    #         fc_mu_b = torch.from_numpy(weights[(h2_size*fc_mu_size):fc_mu_end])
    #         fc_std_W = torch.from_numpy(weights[fc_mu_end:fc_mu_end+(h2_size*fc_std_size)].reshape(h2_size, fc_std_size))
    #         fc_std_b = torch.from_numpy(weights[fc_mu_end+(h2_size*fc_std_size):])
    #         self.fc_mu.weight.data.copy_(fc_mu_W.view_as(self.fc_mu.weight.data))
    #         self.fc_mu.bias.data.copy_(fc_mu_b.view_as(self.fc_mu.bias.data))
    #         self.fc_std.weight.data.copy_(fc_std_W.view_as(self.fc_std.weight.data))
    #         self.fc_std.bias.data.copy_(fc_std_b.view_as(self.fc_std.bias.data))
    #     else:
    #         s_size = self.s_size
    #         h1_size = self.h1_size
    #         h2_size = self.h2_size
    #         fc_mu_size = self.fc_mu_size
    #         fc_std_size = self.fc_std_size
    #         # separate the weights for each layer
    #         fc1_end = (s_size*h1_size)+h1_size
    #         fc1_W = torch.from_numpy(weights[:s_size*h1_size].reshape(s_size, h1_size))
    #         fc1_b = torch.from_numpy(weights[s_size*h1_size:fc1_end])
    #         fc2_end = fc1_end+(h1_size*h2_size)+h2_size
    #         fc2_W = torch.from_numpy(weights[fc1_end:fc1_end+(h1_size*h2_size)].reshape(h1_size, h2_size))
    #         fc2_b = torch.from_numpy(weights[fc1_end+(h1_size*h2_size):fc2_end])
    #         fc_mu_end=fc2_end+(h2_size*fc_mu_size)+fc_mu_size
    #         fc_mu_W = torch.from_numpy(weights[fc2_end:fc2_end+(h2_size*fc_mu_size)].reshape(h2_size, fc_mu_size))
    #         fc_mu_b = torch.from_numpy(weights[fc2_end+(h2_size*fc_mu_size):fc_mu_end])
    #         fc_std_W = torch.from_numpy(weights[fc_mu_end:fc_mu_end+(h2_size*fc_std_size)].reshape(h2_size, fc_std_size))
    #         fc_std_b = torch.from_numpy(weights[fc_mu_end+(h2_size*fc_std_size):])
    #         # set the weights for each layer
    #         self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
    #         self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
    #         self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
    #         self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))
    #         self.fc_mu.weight.data.copy_(fc_mu_W.view_as(self.fc_mu.weight.data))
    #         self.fc_mu.bias.data.copy_(fc_mu_b.view_as(self.fc_mu.bias.data))
    #         self.fc_std.weight.data.copy_(fc_std_W.view_as(self.fc_std.weight.data))
    #         self.fc_std.bias.data.copy_(fc_std_b.view_as(self.fc_std.bias.data))
    # def get_weights(self):
    #     return torch.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()
