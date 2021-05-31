import os
from chrono import Chrono
from simu import make_simu_from_params
from policies import BernoulliPolicy, NormalPolicy, SquashedGaussianPolicy, DiscretePolicy, PolicyWrapper, BetaPolicy
from critics import VNetwork, QNetworkContinuous
from arguments import get_args
from visu.visu_critics import plot_critic
from visu.visu_policies import plot_policy
from visu.visu_results import plot_results
import gym


def create_data_folders() -> None:
    """
    Create folders where to save output files if they are not already there
    :return: nothing
    """
    if not os.path.exists("data/save"):
        os.mkdir("./data")
        os.mkdir("./data/save")
    if not os.path.exists("data/critics"):
        os.mkdir("./data/critics")
    if not os.path.exists('data/policies/'):
        os.mkdir('./data/policies')
    if not os.path.exists('data/results/'):
        os.mkdir('./data/results')
    # if not os.path.exists('./data/results/gradients_angles/'):
    #     os.mkdir('./data/results/gradient_angles/')


def set_files(study_name, env_name):
    """
    Create files to save the policy loss and the critic loss
    :param study_name: the name of the study
    :param env_name: the name of the environment
    :return:
    """
    policy_loss_name = "data/save/policy_loss_" + study_name + '_' + env_name + ".txt"
    policy_loss_file = open(policy_loss_name, "w")
    critic_loss_name = "data/save/critic_loss_" + study_name + '_' + env_name + ".txt"
    critic_loss_file = open(critic_loss_name, "w")
    return policy_loss_file, critic_loss_file


def study_cem(params, starting_pol=None) -> None:
    """
    Start a study of CEM algorithms
    :param params: the parameters of the study
    :param starting_pol: initial policy
    :return: nothing
    """
    assert params.policy_type in ['squashedGaussian', 'normal', 'beta'], 'unsupported policy type'
    chrono = Chrono()
    # cuda = torch.device('cuda')
    study = params.gradients
    if params.nb_trajs_cem is not None:
        params.nb_trajs = params.nb_trajs_cem
    simu = make_simu_from_params(params)
    for i in range(1):  # len(study) Only sum here
        simu.env.set_file_name('cem' + study[i] + '_' + simu.env_name)
        print("study : ", study[i])
        for j in range(params.nb_repet):
            simu.env.reinit()
            if params.policy_type == "squashedGaussian":
                policy = SquashedGaussianPolicy(simu.obs_size, 32, 64, 1)
            elif params.policy_type == "normal":
                policy = NormalPolicy(simu.obs_size, 32, 64, 1)
            elif params.policy_type == "beta":
                policy = BetaPolicy(simu.obs_size, 32, 64, 1)
            if starting_pol is not None:
                policy.set_weights(starting_pol[j])
            pw = PolicyWrapper(policy, j, params.policy_type, simu.env_name, params.team_name, params.max_episode_steps)
            # plot_policy(policy, simu.env, True, simu.env_name, study[i], '_ante_', j, plot=False)
            simu.train_cem(pw, params, policy)
            # plot_policy(policy, simu.env, True, simu.env_name, study[i], '_post_', j, plot=False)
    chrono.stop()


def study_pg(params, starting_pol=None) -> None:
    """
    Start a study of the policy gradient algorithms
    :param params: the parameters of the study
    :param starting_pol: initial policy
    :return: nothing
    """
    assert params.policy_type in ['bernoulli', 'normal', 'squashedGaussian', 'discrete', 'beta'], \
        'unsupported policy type'
    chrono = Chrono()
    # cuda = torch.device('cuda')
    study = params.gradients
    if params.nb_trajs_pg is not None:
        params.nb_trajs = params.nb_trajs_pg
    simu = make_simu_from_params(params)
    for i in range(len(study)):
        simu.env.set_file_name('pg'+study[i] + '_' + simu.env_name, params.nb_cycles)
        policy_loss_file, critic_loss_file = set_files(study[i], simu.env_name)
        print("study : ", study[i])
        for j in range(params.nb_repet):
            simu.env.reinit()
            if params.policy_type == "bernoulli":
                policy = BernoulliPolicy(simu.obs_size, 24, 36, 1, params.lr_actor)
            elif params.policy_type == "discrete":
                if isinstance(simu.env.action_space, gym.spaces.box.Box):
                    nb_actions = int(simu.env.action_space.high[0] - simu.env.action_space.low[0] + 1)
                    print("Error : environment action space is not discrete :" + str(simu.env.action_space))
                else:
                    nb_actions = simu.env.action_space.n
                policy = DiscretePolicy(simu.obs_size, 24, 36, nb_actions, params.lr_actor)
            elif params.policy_type == "normal":
                policy = NormalPolicy(simu.obs_size, 32, 64, 1, params.lr_actor)
            elif params.policy_type == "squashedGaussian":
                policy = SquashedGaussianPolicy(simu.obs_size, 32, 64, 1, params.lr_actor)
            elif params.policy_type == "beta":
                policy = BetaPolicy(simu.obs_size, 32, 64, 1, params.lr_actor)
            if starting_pol is not None:
                policy.set_weights(starting_pol[j])
            pw = PolicyWrapper(policy, j, params.policy_type, simu.env_name, params.team_name, params.max_episode_steps)
            plot_policy(policy, simu.env, True, simu.env_name, study[i], '_ante_', j, plot=False)

            if not simu.discrete:
                act_size = simu.env.action_space.shape[0]
                critic = QNetworkContinuous(simu.obs_size + act_size, 24, 36, 1, params.lr_critic)
            else:
                critic = VNetwork(simu.obs_size, 24, 36, 1, params.lr_critic)
            # plot_critic(simu, critic, policy, study[i], '_ante_', j)

            simu.train_pg(pw, params, policy, policy_loss_file)
            plot_policy(policy, simu.env, True, simu.env_name, study[i], '_post_', j, plot=False)
        plot_critic(simu, critic, policy, study[i], '_post_', j)
        critic.save_model('data/critics/' + params.env_name + '#' + params.team_name + '#' + study[i] + str(j) + '.pt')
    chrono.stop()


def get_same_starting_policies(params):
    simu = make_simu_from_params(params)
    policies = []
    for i in range(params.nb_repet):
        if params.policy_type == 'normal':
            policies.append(NormalPolicy(simu.obs_size, 32, 64, 1, params.lr_actor).get_weights())
        elif params.policy_type == 'squashedGaussian':
            policies.append(SquashedGaussianPolicy(simu.obs_size, 32, 64, 1, params.lr_actor).get_weights())
        elif params.policy_type == 'beta':
            policies.append(BetaPolicy(simu.obs_size, 32, 64, 1, params.lr_actor).get_weights())
    return policies


if __name__ == '__main__':
    args = get_args()
    starting_pol = None
    print(args)
    create_data_folders()
    if args.study_name == 'cem':
        study_cem(args)
        plot_results(args)
    elif args.study_name == 'pg':
        study_pg(args)
        plot_results(args)
    elif args.study_name == 'comparison':
        if args.start_from_same_policy:
            starting_pol = get_same_starting_policies(args)

        print('PG')
        study_pg(args, starting_pol)

        print('CEM')
        study_cem(args, starting_pol)

        plot_results(args)
    else:
        study_pg(args)
        plot_results(args)
