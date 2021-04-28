import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal
from policies.generic_net import GenericNet



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
        if (learning_rate != None):
            self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0)
        self.s_size = l1
        self.h1_size = l2
        self.h2_size = l3
        self.fc_mu_size = l4
        self.fc_std_size = l4

        #To study the gradient
        self.fc1Wgrad=None
        self.fc1Bgrad=None
        self.fc2Wfrad=None
        self.fc2Bgrad=None
        self.fc_muWgrad=None
        self.fc_muBgrad=None
        self.fc_stdWgrad=None
        self.fc_stdBgrad=None

    def get_gradient(self):
        grad=np.concatenate((self.fc1Wgrad,self.fc1Bgrad,self.fc2Wfrad,self.fc2Bgrad,self.fc_muWgrad,self.fc_muBgrad,self.fc_stdWgrad,self.fc_stdBgrad),axis=None)
        return grad

    def set_weights_pg(self, fc1_w, fc1_b, fc2_w, fc2_b): #reset the weights after backpropagation
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_w.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_w.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))

    def get_weights_pg(self): #get the weights of every layers except the last one
        # get the weights for each layer
        fc1_w = self.fc1.weight.data.clone().detach()
        fc1_b = self.fc1.bias.data.clone().detach()
        fc2_w = self.fc2.weight.data.clone().detach()
        fc2_b = self.fc2.bias.data.clone().detach()
        #fc_mu_w = self.fc_mu.weight.data
        #fc_mu_b = self.fc_mu.bias.data
        #fc_std_w = self.fc_std.weight.data
        #fc_std_b = self.fc_std.bias.data
        return fc1_w, fc1_b, fc2_w, fc2_b

    def set_weights(self, weights, fix_layers=False):
        if fix_layers: # last layers weights
            h2_size = self.h2_size
            fc_mu_size = self.fc_mu_size
            fc_std_size = self.fc_std_size
            fc_mu_end= (h2_size*fc_mu_size)+fc_mu_size
            fc_mu_W = torch.from_numpy(weights[:(h2_size*fc_mu_size)].reshape(h2_size, fc_mu_size))
            fc_mu_b = torch.from_numpy(weights[(h2_size*fc_mu_size):fc_mu_end])
            fc_std_W = torch.from_numpy(weights[fc_mu_end:fc_mu_end+(h2_size*fc_std_size)].reshape(h2_size, fc_std_size))
            fc_std_b = torch.from_numpy(weights[fc_mu_end+(h2_size*fc_std_size):])
            self.fc_mu.weight.data.copy_(fc_mu_W.view_as(self.fc_mu.weight.data))
            self.fc_mu.bias.data.copy_(fc_mu_b.view_as(self.fc_mu.bias.data))
            self.fc_std.weight.data.copy_(fc_std_W.view_as(self.fc_std.weight.data))
            self.fc_std.bias.data.copy_(fc_std_b.view_as(self.fc_std.bias.data))
        else:
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
            fc_mu_end=fc2_end+(h2_size*fc_mu_size)+fc_mu_size
            fc_mu_W = torch.from_numpy(weights[fc2_end:fc2_end+(h2_size*fc_mu_size)].reshape(h2_size, fc_mu_size))
            fc_mu_b = torch.from_numpy(weights[fc2_end+(h2_size*fc_mu_size):fc_mu_end])
            fc_std_W = torch.from_numpy(weights[fc_mu_end:fc_mu_end+(h2_size*fc_std_size)].reshape(h2_size, fc_std_size))
            fc_std_b = torch.from_numpy(weights[fc_mu_end+(h2_size*fc_std_size):])
            # set the weights for each layer
            self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
            self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
            self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
            self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))
            self.fc_mu.weight.data.copy_(fc_mu_W.view_as(self.fc_mu.weight.data))
            self.fc_mu.bias.data.copy_(fc_mu_b.view_as(self.fc_mu.bias.data))
            self.fc_std.weight.data.copy_(fc_std_W.view_as(self.fc_std.weight.data))
            self.fc_std.bias.data.copy_(fc_std_b.view_as(self.fc_std.bias.data))

    def get_weights_dim(self, fix_layers):
        if fix_layers:
            return (self.h2_size+1)*2 # last layer
        else:
            return (self.s_size+1)*self.h1_size + (self.h1_size+1)*self.h2_size + (self.h2_size+1)*2#self.fc_mu_size*self.fc_std_size


    def forward(self, state):
        """
         Compute the pytorch tensors resulting from sending a state or vector of states through the policy network
         The obtained tensors can be used to obtain an action by calling select_action
         :param state: the input state(s)
         :return: the resulting pytorch tensor (here the max and standard deviation of a Gaussian probability of action)
         """
        # state_temp=state
        state = torch.from_numpy(state).float()
        state = self.relu(self.fc1(state))
        state = self.relu(self.fc2(state))
        mu = self.fc_mu(state)
        std = self.fc_std(state)
        # std = 2*np.ones(np.shape(mu.data.numpy().astype(float)))
        # std=torch.from_numpy(std).float()
        # std = torch.absolute(std)


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
                return np.clip(mu.data.numpy().astype(float),-2.0, 2.0)
            else:
                n = Normal(mu, std)
                action = n.sample()
            return np.clip(action.data.numpy().astype(float),-2.0,2.0)

    def train_pg(self, state, action, reward):
        """
        Train the policy using a policy gradient approach
        :param state: the input state(s)
        :param action: the input action(s)
        :param reward: the resulting reward
        :return: the loss applied to train the policy
        """
        action = torch.FloatTensor(action)

        # print(action)
        reward = torch.FloatTensor(reward)
        mu, std = self.forward(state)
        # Negative score function x reward
        loss = -Normal(mu, std).log_prob(action).sum(dim=-1)*reward
        self.update(loss)
        self.fc1Wgrad=self.fc1.weight.grad.numpy().flatten()
        self.fc1Bgrad=self.fc1.bias.grad.numpy().flatten()
        self.fc2Wfrad=self.fc2.weight.grad.numpy().flatten()
        self.fc2Bgrad=self.fc2.bias.grad.numpy().flatten()
        self.fc_muWgrad=self.fc_mu.weight.grad.numpy().flatten()
        self.fc_muBgrad=self.fc_mu.bias.grad.numpy().flatten()
        self.fc_stdWgrad=self.fc_std.weight.grad.numpy().flatten()
        self.fc_stdBgrad=self.fc_std.bias.grad.numpy().flatten()
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

    def get_weights(self):
        return torch.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()
