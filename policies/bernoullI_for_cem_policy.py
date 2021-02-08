import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Bernoulli
from policies.generic_net import GenericNet


def make_det_vec(values):
    """
    Transform the output vector of a Bernoulli policy into a vector of deterministic choices
    :param values: the Bernoulli policy output vector (turned into a numpy array)
    :return: the vector of binary choices
    """
    choices = []
    for v in values:
        if v > 0.5:
            choices.append(1.0)
        else:
            choices.append(0.0)
    return choices
class BernoulliCEM(GenericNet):
    """
    A policy whose probabilistic output is a boolean value for each state
    """
    def __init__(self, l1, l2, l3, l4):
        super(BernoulliCEM, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(l1, l2)
        # self.fc1.weight.data.uniform_(-1.0, 1.0)
        self.fc2 = nn.Linear(l2, l3)
        # self.fc2.weight.data.uniform_(-1.0, 1.0)
        self.fc3 = nn.Linear(l3, l4)  # Prob of Left
        # self.fc3.weight.data.uniform_(-1.0, 1.0)
        self.s_size = l1
        self.h1_size = l2
        self.h2_size = l3
        self.a_size = l4
    def set_weights(self, weights):
        s_size = self.s_size
        h1_size = self.h1_size
        h2_size = self.h2_size
        a_size = self.a_size
        # separate the weights for each layer
        fc1_end = (s_size*h1_size)+h1_size
        fc1_W = torch.from_numpy(weights[:s_size*h1_size].reshape(s_size, h1_size))
        fc1_b = torch.from_numpy(weights[s_size*h1_size:fc1_end])
        fc2_end = fc1_end+(h1_size*h2_size)+h2_size
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end+(h1_size*h2_size)].reshape(h1_size, h2_size))
        fc2_b = torch.from_numpy(weights[fc1_end+(h1_size*h2_size):fc2_end])
        fc3_W = torch.from_numpy(weights[fc2_end:fc2_end+(h2_size*a_size)].reshape(h2_size, a_size))
        fc3_b = torch.from_numpy(weights[fc2_end+(h2_size*a_size):])
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))
        self.fc3.weight.data.copy_(fc3_W.view_as(self.fc3.weight.data))
        self.fc3.bias.data.copy_(fc3_b.view_as(self.fc3.bias.data))


    def get_weights_dim(self):
        return (self.s_size+1)*self.h1_size + (self.h1_size+1)*self.h2_size + (self.h2_size+1)*self.a_size

    def forward(self, state):
        """
         Compute the pytorch tensors resulting from sending a state or vector of states through the policy network
         The obtained tensors can be used to obtain an action by calling select_action
         :param state: the input state(s)
         :return: the resulting pytorch tensor (here the probability of giving 0 or 1 as output)
         """
        state = torch.from_numpy(state).float()
        state = self.relu(self.fc1(state))
        state = self.relu(self.fc2(state))
        action = torch.sigmoid(self.fc3(state))
        return action #paramtre de la bernoulli

    def select_action(self, state, deterministic):
        """
        Compute an action or vector of actions given a state or vector of states
        :param state: the input state(s)
        :param deterministic: whether the policy should be considered deterministic or not
        :return: the resulting action(s)
        """
        with torch.no_grad():
            probs = self.forward(state)
            if deterministic:
                return make_det_vec(probs)
            else:
                action = Bernoulli(probs).sample()
            return action.data.numpy().astype(int)

    def train_cem(self, state, action, reward):
        """
        Train the policy using a policy gradient approach
        :param state: the input state(s)
        :param action: the input action(s)
        :param reward: the resulting reward
        :return: the loss applied to train the policy
        """
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        probs = self.forward(state)
        m = Bernoulli(probs)
        loss = -m.log_prob(action).sum(dim=-1) * reward  # Negative score function x reward
        return loss
