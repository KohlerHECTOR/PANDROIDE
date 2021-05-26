import gym
from math import pi

class BetaWrapperPendul(gym.Wrapper):
    """
    Specific wrapper to scale the reward of the pendulum environment
    """
    def __init__(self, env):
        super(BetaWrapperPendul, self).__init__(env)
        self.min_reward_ = -(pi ** 2 + 0.1 * 8 ** 2 + 0.001 * 2 ** 2)

    def step(self, action):
        next_state, reward, done, y = self.env.step(2*(2*action-1))
        return next_state, reward, done, y
