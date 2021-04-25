import argparse

# the following functions are used to build file names for saving data and displaying results


def make_study_string(params):
    return params.env_name + '_' + params.study_name + '_' + params.critic_update_method \
           + '_' + params.critic_estim_method+ '_eval_' + str(params.deterministic_eval)


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
    parser.add_argument('--render', type=bool, default=False, help='visualize the run or not') # Only False when not used
    # study settings
    parser.add_argument('--study_name', type=str, default='pg', help='study name: pg, regress, nstep, cem, evo_pg')
    parser.add_argument('--critic_update_method', type=str, default="dataset", help='critic update method: batch or dataset')
    parser.add_argument('--policy_type', type=str, default="normal", help='policy type: bernoulli, normal, squashedGaussian, discrete')
    parser.add_argument('--team_name', type=str, default='default_team', help='team name')
    parser.add_argument('--deterministic_eval', type=bool, default=True, help='deterministic policy evaluation?') # Only False when not used
    # study parameters
    parser.add_argument('--nb_repet', type=int, default=1, help='number of repetitions to get statistics')
    parser.add_argument('--nb_cycles', type=int, default=40, help='number of training cycles')
    parser.add_argument('--nb_trajs', type=int, default=20, help='number of trajectories in a MC batch')
    parser.add_argument('--nb_trajs_cem', type=int, default=5, help='number of trajectories in a MC batch for cem')
    parser.add_argument('--nb_trajs_pg', type=int, default=20, help='number of trajectories in a MC batch for pg')
    parser.add_argument('--nb_batches', type=int, default=20, help='number of updates of the network using datasets')
    # algo settings
    parser.add_argument('--gradients', type=str, nargs='+', default=['sum'], help='other: baseline, beta')
    parser.add_argument('--critic_estim_method', type=str, default="td", help='critic estimation method: mc, td or nstep')
    parser.add_argument('--fix_layers', type=str, default=False, help='only for normal')
    # learning parameters

    parser.add_argument('--gamma', type=float, default=1, help='discount factor')
    parser.add_argument('--lr_actor', type=float, default=0.0001, help='learning rate of the actor')
    parser.add_argument('--lr_critic', type=float, default=0.01, help='learning rate of the critic')
    parser.add_argument('--beta', type=float, default=0.1, help='temperature in AWR-like learning')
    parser.add_argument('--nstep', type=int, default=5, help='n in n-step return')
    parser.add_argument('--batch_size', type=int, default=64, help='size of a minibatch')
    parser.add_argument('--nb_workers', type=int, default=2, help='number of cpus to collect samples')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle replay samples or not') # Only False when not used
    parser.add_argument('--max_episode_steps', type=int, default=200, help='duration of an episode (step limit)')
    parser.add_argument('--sigma', type=float, default=1.0, help='noise')
    parser.add_argument('--elites_frac',type=float, default=0.2, help='proportion of the population to keep at each iter')
    parser.add_argument('--bests_frac',type=float, default=1, help='proportion of the population to keep at each iter')
    parser.add_argument('--population',type=int, default=15, help='population')

    # plots settings
    parser.add_argument('--night_mode',type=bool, default=False, help='dont show the plots')
    parser.add_argument('--plot_mode',type=str, default="all", help='to launch several studies at the same time')
    parser.add_argument('--nb_intervals',type=int, default=50, help='number of intervals for plot of segment')
    # vignettes
    parser.add_argument('--nb_lines', default=5, type=int)# number of directions generated,good value : precise 100, fast 60, ultrafast 50
    parser.add_argument('--minalpha', default=0.0, type=float)# start value for alpha, good value : 0.0
    parser.add_argument('--nb_evals', default=5, type=int)# number of steps for the evaluation. Depends on environment.
    parser.add_argument('--maxalpha', default=0.01, type=float)# end value for alpha, good value : large 100, around model 10
    parser.add_argument('--stepalpha', default=0.0001, type=float)# step for alpha in the loop, good value : precise 0.5 or 1, less precise 2 or 3
    #	2D plot parameters
    parser.add_argument('--pixelWidth', default=10, type=int)# width of each pixel in 2D Vignette
    parser.add_argument('--pixelHeight', default=10, type=int)# height of each pixel in 2D Vignette
    #	3D plot parameters
    parser.add_argument('--x_diff', default=2., type=float)# the space between each point along the x-axis
    parser.add_argument('--y_diff', default=2., type=float)# the space between each point along the y-axis

    # File management
	#	Input parameters
    parser.add_argument('--inputDir', default="Models", type=str)# name of the directory containing the models to load
    parser.add_argument('--basename', default="rl_model_", type=str)# file prefix for the loaded model
    parser.add_argument('--min_iter', default=1, type=int)# iteration (file suffix) of the first model
    parser.add_argument('--max_iter', default=10, type=int)# iteration (file suffix) of the last model
    parser.add_argument('--step_iter', default=1, type=int)# iteration step between two consecutive models
    # 		Input policies parameters
    parser.add_argument('--policiesPath', default='None', type=str) # path to a list of policies to be included in Vignette
    #	Output parameters
    parser.add_argument('--saveInFile', default=True, type=bool)# true if want to save the savedVignette
    parser.add_argument('--save2D', default=True, type=bool)# true if want to save the 2D Vignette
    parser.add_argument('--save3D', default=True, type=bool)# true if want to save the 3D Vignette
    parser.add_argument('--directoryFile', default="SavedVignette", type=str)# name of the directory that will contain the vignettes
    parser.add_argument('--directory2D', default="Vignette_output", type=str)# name of the directory that will contain the 2D vignette
    parser.add_argument('--directory3D', default="Vignette_output", type=str)# name of the directory that will contain the 3D vignette
    # Tools parameters	#	Drawing parameters
    parser.add_argument('--maxValue', default=360, type=int)# max score value for colormap used (dependent of benchmark used)
    parser.add_argument('--line_height', default=3, type=int) # The height in number of pixel for each result
	#	Dot product parameters
    parser.add_argument('--dotWidth', default=150, type=int)# max width of the dot product (added on the side)
    parser.add_argument('--dotText', default=True, type=str)# true if want to show value of the dot product
    parser.add_argument('--xMargin', default=10, type=int) # xMargin for the side panel

	# File management
	#	Output parameters
    parser.add_argument('--saveFile', default=True, type=bool) # True if want to save the Gradient as SavedGradient
    parser.add_argument('--saveImage', default=True, type=bool) # True if want to save the Image of the Gradient
    parser.add_argument('--directoryFileGrad', default="SavedGradient", type=str) # name of the directory where SavedGradient is saved
    parser.add_argument('--directoryImage', default="Gradient_output", type=str) # name of the output directory that will contain the image
    parser.add_argument('--saved_file_name', default = 'new_grad_', type = str)

    '''
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str, default='experiments/saved_models/', help='the path to save the models')
    parser.add_argument('--save-files', type=str, default='experiments/saved_files/', help='the path to save the files')
    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replaced')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')


    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=100, help='the number of tests')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int, default=1500, help='the demo length')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=20, help='the rollouts per mpi')
    parser.add_argument('--use-curriculum', action='store_true', default=False, help='use the curriculum or not')
    parser.add_argument('--multiple-target-sizes', action='store_true', default=False, help='train the agent to reach targets of different sizes')
    parser.add_argument('--include-movement-cost', action='store_false', default=True, help='consider or not the cost of a movement')
    '''

    args = parser.parse_args()
    return args
