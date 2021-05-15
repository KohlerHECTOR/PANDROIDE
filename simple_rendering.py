import numpy as np
import gym
import math
from policies import GenericNet,BernoulliPolicy, NormalPolicy, SquashedGaussianPolicy, DiscretePolicy, PolicyWrapper
from arguments import get_args
import os
import gym
import my_gym  # Necessary to see CartPoleContinuous, though PyCharm does not understand this
import numpy as np
from wrappers import FeatureInverter, BinaryShifter, BinaryShifterDiscrete, ActionVectorAdapter, \
    PerfWriter, PendulumWrapper, MountainCarContinuousWrapper
from gym.wrappers import TimeLimit
from environment import make_env
def load_policies(folder):
    """
     :param: folder : name of the folder containing policies
     Output : none (policies of the folder stored in self.env_dict)
     """
    listdir = os.listdir(folder)
    policies = []
    for policy_file in listdir:
        pw = PolicyWrapper(GenericNet(), 0,"", "", "", 0)
        policy,_ = pw.load(directory+policy_file)
        policy = policy.get_weights()
        policies.append(policy)
    return policies


def render_pol(params, env, weights):
    """
    Function to evaluate a policy over 900 episodes
    :param env: the evaluation environment
    :param policy: the evaluated policy
    :param deterministic: whether the evaluation uses a deterministic policy
    :return: the obtained vector of 900 scores
    """
    policy = SquashedGaussianPolicy(env.observation_space.shape[0], 24, 36, 1, params.lr_actor)
    policy.set_weights(weights)
    state = env.reset()
    env.render(mode='rgb_array')
    for i in range(1000):
        action = policy.select_action(state, deterministic = True)
        print(action)
        next_state, reward, done, _ = env.step(action)
        env.render(mode='rgb_array')
        state = next_state
    print('finished rendering')
    # print("team: ", policy.team_name, "mean: ", scores.mean(), "std:", scores.std())
if __name__ == '__main__':
    args = get_args()
    print(args)

    pw = PolicyWrapper(GenericNet(),0, "", "", "", 0)

    env = make_env(args.env_name, args.policy_type, args.max_episode_steps)
    env = gym.wrappers.Monitor(env, './videos/PG_fin')

    directory = os.getcwd() + '/Models/'
    weights_vecs=load_policies(directory)
    for weights_vec in weights_vecs:
        render_pol(args, env, weights_vec)
    env.close()
