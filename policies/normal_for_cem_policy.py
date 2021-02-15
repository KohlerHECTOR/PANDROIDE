import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal
from policies.generic_net import GenericNet


class NormalCEM(GenericNet):
    """
    A policy whose probabilistic output is drawn from a Gaussian function
    """
    def __init__(self, l1, l2, l3, l4, learning_rate = 0.1):
        super(NormalCEM, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(l1, l2)
        self.fc2 = nn.Linear(l2, l3)
        self.fc_mu = nn.Linear(l3, l4)
        self.fc_std = nn.Linear(l3, l4)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.s_size = l1
        self.h1_size = l2
        self.h2_size = l3
        self.fc_mu_size = l4
        self.fc_std_size = l4

    def set_weights(self, weights):
        s_size = self.s_size
        h1_size = self.h1_size
        h2_size = self.h2_size
        fc_mu_size = self.fc_mu_size
        fc_std_size = self.fc_std_size
        # separate the weights for each layer
        fc1_end = (s_size*h1_size)+h1_size
        fc1_W = torch.from_numpy(weights[:s_size*h1_size].reshape(s_size, h1_size))
        fc1_b = torch.from_numpy(weights[s_size*h1_size:fc1_end])
        fc2_end = fc1_end+(h1_size*h2_size)+h2_size
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end+(h1_size*h2_size)].reshape(h1_size, h2_size))
        fc2_b = torch.from_numpy(weights[fc1_end+(h1_size*h2_size):fc2_end])
        fc_mu_W = torch.from_numpy(weights[fc2_end:fc2_end+(h2_size*fc_mu_size)].reshape(h2_size, fc_mu_size))
        fc_mu_b = torch.from_numpy(weights[fc2_end+(h2_size*fc_mu_size):])
        fc_std_W = torch.from_numpy(weights[fc2_end:fc2_end+(h2_size*fc_std_size)].reshape(h2_size, fc_std_size))
        fc_std_b = torch.from_numpy(weights[fc2_end+(h2_size*fc_std_size):])
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))
        self.fc_mu.weight.data.copy_(fc_mu_W.view_as(self.fc_mu.weight.data))
        self.fc_mu.bias.data.copy_(fc_mu_b.view_as(self.fc_mu.bias.data))
        self.fc_std.weight.data.copy_(fc_std_W.view_as(self.fc_std.weight.data))
        self.fc_std.bias.data.copy_(fc_std_b.view_as(self.fc_std.bias.data))



    def get_weights_dim(self):
        return (self.s_size+1)*self.h1_size + (self.h1_size+1)*self.h2_size + (self.h2_size+1)*self.fc_mu_size*self.fc_std_size

    def get_weights_dim_s_h1(self):
        return (self.s_size+1)*self.h1_size

    def get_weights_dim_h1_h2(self):
        return (self.h1_size+1)*self.h2_size

    def get_weights_dim_h2_mu_std(self):
        return (self.h2_size+1)*self.fc_mu_size*self.fc_std_size

    def forward(self, state):
        """
         Compute the pytorch tensors resulting from sending a state or vector of states through the policy network
         The obtained tensors can be used to obtain an action by calling select_action
         :param state: the input state(s)
         :return: the resulting pytorch tensor (here the max and standard deviation of a Gaussian probability of action)
         """
        state = torch.from_numpy(state).float()
        state = self.relu(self.fc1(state))
        state = self.relu(self.fc2(state))
        mu = self.fc_mu(state)
        std = self.fc_std(state)
        return mu, std

    def select_action(self, state, deterministic=True):
        """
        Compute an action or vector of actions given a state or vector of states
        :param state: the input state(s)
        :param deterministic: whether the policy should be considered deterministic or not
        :return: the resulting action(s)
        """
        with torch.no_grad():
            mu, std = self.forward(state)
            if deterministic:
                return mu.data.numpy().astype(float)
            else:
                n = Normal(mu, std)
                action = n.sample()
            return action.data.numpy().astype(float)