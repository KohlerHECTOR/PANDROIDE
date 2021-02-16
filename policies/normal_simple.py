import torch
from torch.distributions import Normal
from scipy.stats import multivariate_normal
import numpy

class NormalSimple:
    def __init__(self, mean):
        self.mean=mean
        self.cov=numpy.identity(len(self.mean))
        #self.cov=cov
        self.mean_size=len(self.mean)
        #self.cov_row_size=len(self.cov[0])
        #self.cov_matrix_size=self.cov_row_size*self.cov_row_size
    def get_weights_dim(self):
        return self.mean_size#+self.cov_matrix_size
    def set_weights(self,new_weights):
        self.mean=new_weights
        #self.cov=new_weights[self.mean_size:].reshape(self.cov_row_size,self.cov_row_size)

    def select_action(self, state, deterministic=False):
        """
        Compute an action or vector of actions given a state or vector of states
        :param state: the input state(s)
        :param deterministic: whether the policy should be considered deterministic or not
        :return: the resulting action(s)
        """
        if deterministic:
            return self.mean.data.numpy().astype(float)
        else:
            var = multivariate_normal(mean=self.mean, cov=self.cov)
            action=var.pdf(state)
            #print(action)
            return [action]
