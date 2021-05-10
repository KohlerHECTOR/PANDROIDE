import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal
from policies.generic_net import GenericNet
from copy import deepcopy



class NormalPolicy(GenericNet):
    """
    A policy whose probabilistic output is drawn from a Gaussian function
    """
    def __init__(self, l1, l2, l3, l4, learning_rate=None):
        super(NormalPolicy, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(l1, l2)
        self.fc2 = nn.Linear(l2, l3)
        self.fc_mu = nn.Linear(l3, l4)
        self.fc_std = nn.Linear(l3, l4)
        self.sigmoid = nn.Sigmoid()
        if (learning_rate != None):
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.s_size = l1
        self.h1_size = l2
        self.h2_size = l3
        self.fc_mu_size = l4
        self.fc_std_size = l4

        # #To study the gradient
        # self.fc1Wgrad=None
        # self.fc1Bgrad=None
        # self.fc2Wfrad=None
        # self.fc2Bgrad=None
        # self.fc_muWgrad=None
        # self.fc_muBgrad=None
        # self.fc_stdWgrad=None
        # self.fc_stdBgrad=None
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
        # state_temp=state
        # print(state)
        state = torch.from_numpy(state).float()
        state = self.relu(self.fc1(state))
        state = self.relu(self.fc2(state))
        mu = self.fc_mu(state)
        std = self.sigmoid(self.fc_std(state))
        # print(std)

        # std = 0.2*np.ones(np.shape(mu.data.numpy().astype(float)))
        # std=torch.from_numpy(std).float()


        # print(mu.data.numpy()) # Note : Can be set to have better results with pg
        return mu, std

    def select_action(self, state, deterministic=False):
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
                # return np.clip(mu.data.numpy().astype(float),-2,2)
            else:
                n = Normal(mu, std)
                action = n.sample()
                return action.data.numpy().astype(float)
                # return np.clip(action.data.numpy().astype(float),-2,2)

    def train_pg(self, state, action, reward):
        """
        Train the policy using a policy gradient approach
        :param state: the input state(s)
        :param action: the input action(s)
        :param reward: the resulting reward
        :return: the loss applied to train the policy
        """
        action = torch.FloatTensor(action)

        reward = torch.FloatTensor(reward)
        mu, std = self.forward(state)
        # print(mu)
        # print(std)

        # Negative score function x reward
        loss = -Normal(mu, std).log_prob(action).sum(dim=-1)*reward
        # print(loss)
        if np.isnan(loss.data.numpy().any()):
            print('bad')
        self.update(loss)
        # self.fc1Wgrad=self.fc1.weight.grad.numpy().flatten()
        # self.fc1Bgrad=self.fc1.bias.grad.numpy().flatten()
        # self.fc2Wfrad=self.fc2.weight.grad.numpy().flatten()
        # self.fc2Bgrad=self.fc2.bias.grad.numpy().flatten()
        # self.fc_muWgrad=self.fc_mu.weight.grad.numpy().flatten()
        # self.fc_muBgrad=self.fc_mu.bias.grad.numpy().flatten()
        # self.fc_stdWgrad=self.fc_std.weight.grad.numpy().flatten()
        # self.fc_stdBgrad=self.fc_std.bias.grad.numpy().flatten()
        return loss

    def train_regress(self, state, action, estimation_method='log_likelihood'):
        """
         Train the policy to perform the same action(s) in the same state(s) using regression
         :param state: the input state(s)
         :param action: the input action(s)
         :return: the loss applied to train the policy
         """
        assert estimation_method in ['mse', 'log_likelihood'], 'unsupported estimation method'
        action = torch.FloatTensor(action)
        mu, std = self.forward(state)
        if estimation_method == 'mse':
            loss = func.mse_loss(mu, action)
        else:
            normal_distribution = Normal(mu, std)
            loss = -normal_distribution.log_prob(action)
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

    # def get_weights(self):
    #     return torch.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()
