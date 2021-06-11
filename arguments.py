import argparse

# the following functions are used to build file names for saving data and displaying results


def make_study_string(params):
    return params.env_name + '_' + params.study_name + '_' + params.critic_update_method \
           + '_' + params.critic_estim_method + '_eval_' + str(params.deterministic_eval)


def make_study_params_string(params):
    return 'cycles_' + str(params.nb_cycles) + '_trajs_' + str(params.nb_trajs) + '_batches_' + str(params.nb_batches)


def make_learning_params_string(params):
    return 'gamma_' + str(params.gamma) + '_nstep_' + str(params.nstep) + '_lr_act_' \
           + str(params.lr_actor) + '_lr_critic_' + str(params.lr_critic)


def make_full_string(params):
    return make_study_string(params) + '_' + make_study_params_string(params) + '_' \
           + make_learning_params_string(params)


def get_args():
    """
    Standard function to specify the default value of the hyper-parameters of all policy gradient algorithms
    and experimental setups
    :return: the complete list of arguments
    """
    parser = argparse.ArgumentParser()
    # environment setting
    parser.add_argument('--env_name', type=str, default='Pendulum-v0', help='the environment name')
    parser.add_argument('--env_obs_space_name', type=str, default=["pos", "angle"])  # ["pos", "angle", "vx", "v angle"]
    parser.add_argument('--render', type=bool, default=False,
                        help='visualize the run or not')  # Only False when not used
    # study settings
    parser.add_argument('--study_name', type=str, default='sum', help='beta sum discount normalize baseline nstep')
    parser.add_argument('--experiment', type= str, default = 'comparison', help = 'cem, pg, comparison')
    parser.add_argument('--reinforce', type=bool, default = False, help='wether you want to study a reinforce algo or another pg based algo')
    parser.add_argument('--critic_update_method', type=str, default="dataset",
                        help='critic update method: batch or dataset')
    parser.add_argument('--policy_type', type=str, default="beta",
                        help='policy type: bernoulli, normal, squashedGaussian, discrete, beta')
    parser.add_argument('--team_name', type=str, default='default_team', help='team name')
    parser.add_argument('--deterministic_eval', type=bool, default=True,
                        help='deterministic policy evaluation?')  # Only True when not used
    # study parameters
    parser.add_argument('--nb_repet', type=int, default=1, help='number of repetitions to get statistics')
    parser.add_argument('--nb_cycles', type=int, default=40, help='number of training cycles')
    parser.add_argument('--nb_trajs', type=int, default=20, help='number of trajectories in a MC batch')
    parser.add_argument('--nb_trajs_cem', type=int, default=None, help='number of trajectories in a MC batch for cem')
    parser.add_argument('--nb_trajs_pg', type=int, default=20, help='number of trajectories in a MC batch for pg')
    parser.add_argument('--nb_batches', type=int, default=20, help='number of updates of the network using datasets')
    # algo settings
    parser.add_argument('--gradients', type=str, nargs='+', default=['sum'], help='other: baseline, beta')
    parser.add_argument('--critic_estim_method', type=str, default="td",
                        help='critic estimation method: mc, td or nstep')
    # learning parameters
    parser.add_argument('--start_from_policy', type=bool, default=False, help='give a starting policy in /Models. Must be of same type as --policy_type')
    parser.add_argument('--start_from_same_policy', type=bool, default=True, help='force pg and cem to start exploration from same point in space')

    parser.add_argument('--gamma', type=float, default=1, help='discount factor')
    parser.add_argument('--lr_actor', type=float, default=0.0001, help='learning rate of the actor')
    parser.add_argument('--lr_critic', type=float, default=0.01, help='learning rate of the critic')
    parser.add_argument('--beta', type=float, default=0.1, help='temperature in AWR-like learning')
    parser.add_argument('--nstep', type=int, default=5, help='n in n-step return')
    parser.add_argument('--batch_size', type=int, default=64, help='size of a minibatch')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='shuffle replay samples or not')  # Only False when not used
    parser.add_argument('--max_episode_steps', type=int, default=200, help='duration of an episode (step limit)')
    parser.add_argument('--nb_workers', type=int, default=2, help='number of cpus to collect samples')
    parser.add_argument('--sigma', type=float, default=1.0, help='noise for cem covariance matrix')
    parser.add_argument('--elites_frac', type=float, default=0.2,
                        help='proportion of the population to keep at each iter for cem')
    parser.add_argument('--population', type=int, default=15, help='population size for cem')

    # evaluation settings
    parser.add_argument('--eval_freq', default=1, type=int)  # frequency for evaluation.
    parser.add_argument('--save_freq', default=1, type=int)  # frequency for saving evaluation.
    parser.add_argument('--nb_evals', default=5,
                        type=int)  # number of steps for the evaluation. Depends on environment.

    args = parser.parse_args()
    return args
