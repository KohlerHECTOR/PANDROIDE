import os
import pandas as pd
import matplotlib.pyplot as plt
from arguments import make_full_string, get_args
import seaborn as sns
sns.set()

# This file contains variance functions to plot results
# Only a few of them are used and up-to-date
# The others are left here as examples of how to design new plotting functions


def plot_data(filename, label):
    """
    Generic plot function to return a curve from a file with an index and a number per line
    importantly, several datasets can be stored into the same file
    and the curve will contain a variance information based on the repetition
    Retrieving the variance information is based on pandas
    :param filename: the file containing the data
    :param label: the label to be shown in the plot (a string)
    :return: a curve with some variance and a label, embedded in plt.
    """
    data = pd.read_csv(filename, sep=' ')
    data = pd.read_csv(filename, sep=' ', names=list(range(data.shape[1])))
    x1 = list(data.groupby([0]).quantile(0.75)[1])
    x2 = list(data.groupby([0]).quantile(0.25)[1])
    if (len(set(list(data.iloc[:, 0]))) == len(list(data.iloc[:, 0]))):
        x_index = list(data.iloc[:, 0])
    else:
        x_index = list(range(len(x1)))
    x_mean = list(data.groupby([0]).mean()[1])
    x_std = list(data.groupby([0]).std()[1])
    plt.plot(x_index, x_mean, label=label)
    plt.fill_between(x_index, x1, x2, alpha=0.25)
    return x_mean, x_std


def plot_from_file(filename, label) -> None:
    """
    Generic plot function to return a curve from a file with just one number per line
    :param filename: the file containing the data
    :param label: the label to be shown in the plot (a string)
    :return: a curve with a label, embedded in plt.
    """
    with open(filename, 'r') as file:
        data = [float(x) for x in file]
    # print(data)
    plt.plot(data, label=label)


def exploit_duration_full(params) -> None:
    path = os.getcwd() + "/data/save"
    study = params.gradients
    for i in range(len(study)):
        plot_data(path + "/duration_" + study[i] + '_' + params.env_name + '.txt', "duration " + study[i])

    plt.xlabel("Episodes")
    plt.ylabel("Duration")
    plt.legend(loc="lower right")
    plt.title(params.env_name)
    plt.savefig(path + '/../results/durations_' + make_full_string(params) + 'pg.pdf')
    plt.show()


def exploit_reward_full(params) -> None:
    path = os.getcwd() + "/data/save"
    study = params.gradients
    for i in range(len(study)):
        plot_data(path + "/reward_" + params.study_name+study[i] + '_' + params.env_name + '.txt', params.study_name +'_'+"reward " + study[i])

    plt.title(params.env_name)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend(loc="best")
    plt.savefig(path + '/../results/rewards_' + make_full_string(params) + '.pdf')
    plt.show()

def exploit_cov_full(params) -> None:
    path = os.getcwd() + "/data/save"
    study = params.gradients
    for i in range(len(study)):
        plot_data(path + "/covariance_" + '.txt', "cem covariance norm ")

    plt.title(params.env_name)
    plt.xlabel("Episodes")
    plt.ylabel("Covariance Norm")
    plt.legend(loc="best")
    plt.savefig(path + '/../results/covariance_' + make_full_string(params) + '.pdf')
    plt.clf()

def exploit_angles_full(params) -> None:
    path = os.getcwd() + "/data/save"
    study = params.gradients
    for i in range(len(study)):
        for cycle in range(params.nb_cycles):
            plot_data(path + "/gradient_angles_" +"#"+ str(cycle) +'.txt', params.study_name +'_'+"gradient_angles " )

            plt.title(params.env_name)
            plt.xlabel("traj_in_batch")
            plt.ylabel("angle")
            plt.legend(loc="best")
            plt.savefig(path + '/../results/gradient_angles/gradient_angles_'+'#'+str(cycle) + make_full_string(params) + '.pdf')
            plt.clf()

def exploit_reward_full_comparison(params) -> None:
    path = os.getcwd() + "/data/save"
    study = params.gradients
    for i in range(len(study)):
        plot_data(path + "/reward_" + 'pg'+study[i] + '_' + params.env_name + '.txt', "reward pg")
        plot_data(path + "/reward_" + 'cem'+study[i] + '_' + params.env_name + '.txt', "reward cem")
        # plot_data(path + "/reward_" + 'evo_pg'+study[i] + '_' + params.env_name + '.txt', "reward evo_pg")

    plt.title(params.env_name)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend(loc="lower right")
    plt.savefig(path + '/../results/rewards_' + make_full_string(params) + '.pdf')
    plt.show()

def exploit_angles_global_full(params) -> None:
    path = os.getcwd() + "/data/save"
    study = params.gradients
    for i in range(len(study)):
        plot_data(path + "/angle_" + params.study_name+study[i] + '_' + params.env_name + '.txt', "angles " )

    plt.title(params.env_name)
    plt.xlabel("Episodes")
    plt.ylabel("Angle")
    plt.legend(loc="best")
    plt.savefig(path + '/../results/angles_' + make_full_string(params) + '.pdf')
    plt.clf()

def exploit_angles_global_full_comparison(params) -> None:
    path = os.getcwd() + "/data/save"
    study = params.gradients
    for i in range(len(study)):
        plot_data(path + "/angle_" + 'pg'+study[i] + '_' + params.env_name + '.txt', "angles pg")
        plot_data(path + "/angle_" + 'cem'+study[i] + '_' + params.env_name + '.txt', "angles cem")
        # plot_data(path + "/reward_" + 'evo_pg'+study[i] + '_' + params.env_name + '.txt', "reward evo_pg")

    plt.title(params.env_name)
    plt.xlabel("Episodes")
    plt.ylabel("angles")
    plt.legend(loc="lower right")
    plt.savefig(path + '/../results/angles_' + make_full_string(params) + '.pdf')
    plt.clf()

def exploit_distance(params) -> None:
    path = os.getcwd() + "/data/save"
    study = params.gradients
    for i in range(len(study)):
        plot_data(path + "/distance_" + 'pg'+study[i] + '_' + params.env_name + '.txt', "distance pg")
        plot_data(path + "/distance_" + 'cem'+study[i] + '_' + params.env_name + '.txt', "distance cem")
        # plot_data(path + "/reward_" + 'evo_pg'+study[i] + '_' + params.env_name + '.txt', "reward evo_pg")

    plt.title(params.env_name)
    plt.xlabel("Episodes")
    plt.ylabel("distance")
    plt.legend(loc="lower right")
    plt.savefig(path + '/../results/distances_' + make_full_string(params) + '.pdf')
    plt.clf()


def exploit_critic_loss_full(params) -> None:
    path = os.getcwd() + "/data/save"
    study = params.gradients
    for i in range(len(study)):
        plot_data(path + "/critic_loss_" + study[i] + '_' + params.env_name + '.txt', "critic loss " + study[i])

    plt.xlabel("Cycles")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.title(params.env_name)
    plt.savefig(path + '/../results/critic_loss_' + make_full_string(params) + 'pg.pdf')
    plt.show()


def exploit_policy_loss_full(params) -> None:
    path = os.getcwd() + "/data/save"
    study = params.gradients
    for i in range(len(study)):
        plot_data(path + "/policy_loss_" + study[i] + '_' + params.env_name + '.txt', "policy loss " + study[i])

    plt.xlabel("Cycles")
    plt.ylabel("Loss")
    plt.legend(loc="lower right")
    plt.title(params.env_name)
    plt.savefig(path + '/../results/policy_loss_' + make_full_string(params) + 'pg.pdf')
    plt.show()


def exploit_nstep(params) -> None:
    path = os.getcwd() + "/data/save"
    steps = [1, 5, 10, 15, 20]

    for j in ['policy_loss', 'critic_loss', 'reward', 'duration']:
        mean_list = []
        std_list = []
        for i in steps:
            mean, std = plot_data(path + '/' + j + '_nstep_' + str(i) + '_'
                                  + params.env_name + '.txt', j + '_nstep_' + str(i))
            mean_list.append(mean[-1])
            std_list.append(std[-1])

        plt.title(params.env_name)
        plt.xlabel("Episodes")
        plt.ylabel(j)
        plt.legend(loc="lower right")
        plt.savefig(path + '/../results/' + j + '_nstep_' + make_full_string(params) + '.pdf')
        plt.show()

        plt.plot(steps, mean_list, label="bias")
        plt.plot(steps, std_list, label="variance")
        plt.title(params.env_name)
        plt.xlabel("N in N-step")
        plt.ylabel('variance, bias')
        plt.legend(loc="lower right")
        plt.savefig(path + '/../results/bias_variance_' + j + '_' + make_full_string(params) + '.pdf')
        plt.show()


# to be refreshed
def plot_beta_results(params) -> None:
    path = os.getcwd() + "/data/save"
    for beta in [0.1, 0.5, 1.0, 5.0, 10.0]:
        plot_data(path + "/reward_" + str(beta) + '_' + params.env_name + '.txt', "reward " + str(beta))

    plt.title(params.env_name)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend(loc="lower right")
    plt.savefig(path + '/../results/rewards_' + make_full_string(params) + '.pdf')
    plt.show()


def check_nstep(params) -> None:
    path = os.getcwd() + "/data/save"
    study1 = 'batchTD'
    for j in ['loss', 'reward', 'duration']:
        for i in [1]:
            file_name = path + "/" + j + '_nstep_' + str(i) + '_' + params.env_name + '.txt'
            mean, std = plot_data(file_name, j + '_nstep_' + str(i))
            print('n:', i, ' mean :', mean[-1], ' std:', std[-1])
        plot_data(path + "/" + j + '_' + study1 + '_' + params.env_name + '.txt', 'loss ' + study1)

        plt.title(params.env_name)
        plt.xlabel("Episodes")
        plt.ylabel(j)
        plt.legend(loc="lower right")
        plt.savefig(path + '/../results/' + j + '_nstep_check.pdf')
        plt.show()


def exploit_nstep_diff(params) -> None:
    path = os.getcwd() + "/data/save"
    steps = [1, 5, 10, 20]
    mean_list = []
    std_list = []
    for i in steps:
        mean, std = plot_data(path + '/diff_' + str(i) + '_' + params.env_name + '.txt', 'nstep_' + str(i))
        # print('n:', i, ' mean :', mean[-1], ' std:', std[-1])
        mean_list.append(mean[-1])
        std_list.append(std[-1])

    plt.title(params.env_name)
    plt.xlabel("Episodes")
    plt.ylabel("diff")
    plt.legend(loc="lower right")
    plt.savefig(path + '/../results/diff_nstep_' + make_full_string(params) + '.pdf')
    plt.show()

    plt.plot(steps, mean_list, label="bias")
    plt.plot(steps, std_list, label="variance")
    plt.title(params.env_name)
    plt.xlabel("N in N-step")
    plt.ylabel('variance, bias')
    plt.legend(loc="lower right")  # , bbox_to_anchor=(1, 0.5)
    plt.savefig(path + '/../results/bias_variance_' + make_full_string(params) + '.pdf')
    plt.show()

def plot_results(params) -> None:
    """
    Plot the results from a study previously saved in files in "./data/save"
    :param params: parameters of the study
    :return: nothing
    """
    assert params.study_name in ['pg', 'regress', 'nstep','cem','comparison'], 'unsupported study name'
    if params.study_name == "pg":
        # exploit_duration_full(params)
        exploit_reward_full(params)
        exploit_angles_global_full(params)
        # exploit_angles_full(params)
        exploit_policy_loss_full(params)
        # exploit_critic_loss_full(params)
    elif params.study_name == "nstep":
        exploit_nstep(params)
    elif params.study_name == 'cem':
        exploit_reward_full(params)
        exploit_angles_global_full(params)
        exploit_cov_full(params)
    elif params.study_name == 'comparison':
        exploit_reward_full_comparison(params)
        exploit_angles_global_full_comparison(params)
        exploit_cov_full(params)
        exploit_distance(params)
        # exploit_angles_full(params)
        exploit_policy_loss_full(params)
    elif params.study_name == 'evo_pg':
        exploit_reward_full(params)


def plot_results_cem(params) -> None:
    """
    Plot the results from a study previously saved in files in "./data/save"
    :param params: parameters of the study
    :return: nothing
    """
    exploit_reward_full(params)


if __name__ == '__main__':
    args = get_args()
    print(args)
    plot_results(args)
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# from arguments import make_full_string, get_args
# import shutil
# import seaborn as sns
# sns.set()
#
# # This file contains variance functions to plot results
# # Only a few of them are used and up-to-date
# # The others are left here as examples of how to design new plotting functions
#
#
# def plot_data(filename, label, night_mode=False):
#     """
#     Generic plot function to return a curve from a file with an index and a number per line
#     importantly, several datasets can be stored into the same file
#     and the curve will contain a variance information based on the repetition
#     Retrieving the variance information is based on pandas
#     :param filename: the file containing the data
#     :param label: the label to be shown in the plot (a string)
#     :return: a curve with some variance and a label, embedded in plt.
#     """
#     data = pd.read_csv(filename, sep=' ')
#     data = pd.read_csv(filename, sep=' ', names=list(range(data.shape[1])))
#     x1 = list(data.groupby([0]).quantile(0.75)[1])
#     x2 = list(data.groupby([0]).quantile(0.25)[1])
#     x_mean = list(data.groupby([0]).mean()[1])
#     x_std = list(data.groupby([0]).std()[1])
#     if not(night_mode):
#         plt.plot(x_mean, label=label)
#         plt.fill_between(list(range(len(x1))), x1, x2, alpha=0.25)
#     return x_mean, x_std
#
#
# def plot_from_file(filename, label) -> None:
#     """
#     Generic plot function to return a curve from a file with just one number per line
#     :param filename: the file containing the data
#     :param label: the label to be shown in the plot (a string)
#     :return: a curve with a label, embedded in plt.
#     """
#     with open(filename, 'r') as file:
#         data = [float(x) for x in file]
#     # print(data)
#     plt.plot(data, label=label)
#
#
# def exploit_duration_full_cem_vs_pg(params, env, study) -> None:
#     path = os.getcwd() + "/data/save"
#     study = params.studyname
#     plot_data(path + "/duration_" + study + '_' + env + '.txt', "duration " + study)
#     plot_data(path + "/duration_cem_" + study + '_' + env + '.txt', "duration " + study)
#
#     plt.xlabel("Episodes")
#     plt.ylabel("Duration")
#     plt.legend(loc="lower right")
#     plt.title(params.env_name)
#     plt.savefig(path + '/../results/durations_' + make_full_string(params) + 'pg.pdf')
#     plt.show()
#
# def exploit_total_reward_cem_vs_pg(params) -> None:
#     path = os.getcwd() + "/data/save"
#     plot_data(path + "/saved_reward_" + "pg" + '_' + params.env_name + '.txt', "pg")
#     plot_data(path + "/saved_reward_fixed_layers_" + "pg" + '_' + params.env_name + '.txt', "pg fixed")
#     plot_data(path + "/saved_reward_" + "cem" + '_' + params.env_name + '.txt', "cem")
#     plot_data(path + "/saved_reward_fixed_layers_" + "cem" + '_' + params.env_name + '.txt', "cem fixed")
#
#     plt.xlabel("Episodes")
#     plt.ylabel("Rewards")
#     plt.legend(loc="lower right")
#     plt.title(params.env_name + " " + params.policy_type)
#     plt.savefig(path + '/../results/durations_' + make_full_string(params) + 'pgvspgfixedvscemvscemfixed.pdf')
#     plt.show()
#
# def exploit_duration_full_cem(params) -> None:
#     path = os.getcwd() + "/data/save"
#     study = params.gradients
#     for i in range(1):
#         plot_data(path + "/duration_" + study[i] + '_' + params.env_name + '.txt', "duration " + study[i])
#
#     plt.xlabel("Episodes")
#     plt.ylabel("Duration")
#     plt.legend(loc="lower right")
#     plt.title(params.env_name)
#     plt.savefig(path + '/../results/durations_cem_' + make_full_string(params) + 'cem.pdf')
#     plt.show()
#
# def exploit_duration_full(params) -> None:
#     path = os.getcwd() + "/data/save"
#     study = params.gradients
#     for i in range(len(study)):
#         plot_data(path + "/duration_" + study[i] + '_' + params.env_name + '.txt', "duration " + study[i])
#
#     plt.xlabel("Episodes")
#     plt.ylabel("Duration")
#     plt.legend(loc="lower right")
#     plt.title(params.env_name)
#     plt.savefig(path + '/../results/durations_' + make_full_string(params) + 'cem.pdf')
#     fig = plt.show()
#
# def exploit_reward_full(params) -> None:
#     if params.fix_layers:
#         fixed = True
#     else:
#         fixed = False
#     path = os.getcwd() + "/data/save"
#     study = params.gradients
#     if fixed:
#         for i in range(len(study)):
#             plot_data(path + "/reward_" + study[i] + '_' + params.env_name + '.txt', "pg fixed", params.night_mode)
#             shutil.copy(path + "/reward_" + study[i] + '_' + params.env_name + '.txt', path + "/saved_reward_fixed_layers_" + "pg" + '_' + params.env_name + '.txt')
#     else:
#         for i in range(len(study)):
#             plot_data(path + "/reward_" + study[i] + '_' + params.env_name + '.txt', "pg", params.night_mode)
#             shutil.copy(path + "/reward_" + study[i] + '_' + params.env_name + '.txt', path + "/saved_reward_" + "pg" + '_' + params.env_name + '.txt')
#     if not(params.night_mode):
#         plt.xlabel("Episodes")
#         plt.ylabel("Rewards")
#         plt.legend(loc="lower right")
#         plt.title(params.env_name + " " + params.policy_type)
#         plt.show()
#
# def exploit_reward_full_cem(params) -> None:
#     path = os.getcwd() + "/data/save"
#     study = params.gradients
#     #for i in range(1):
#     plot_data(path + "/reward_" + "cem" + '_' + params.env_name + '.txt', "reward " + "cem")
#
# def exploit_total_reward_cem(params) -> None:
#     if params.fix_layers:
#         fixed = True
#     else:
#         fixed = False
#     path = os.getcwd() + "/data/save"
#     study = params.gradients
#     #for i in range(1):
#     if fixed:
#         plot_data(path + "/total_reward_" + "cem" + '_' + params.env_name + '.txt', "cem fixed", params.night_mode)
#         shutil.copy(path + "/total_reward_" + "cem" + '_' + params.env_name + '.txt', path + "/saved_reward_fixed_layers_" + "cem" + '_' + params.env_name + '.txt')
#     else:
#         plot_data(path + "/total_reward_" + "cem" + '_' + params.env_name + '.txt', "cem", params.night_mode)
#         shutil.copy(path + "/total_reward_" + "cem" + '_' + params.env_name + '.txt', path + "/saved_reward_" + "cem" + '_' + params.env_name + '.txt')
#     if not(params.night_mode):
#         plt.title(params.env_name + " " + params.policy_type)
#         plt.xlabel("Episodes")
#         plt.ylabel("Rewards")
#         plt.legend(loc="lower right")
#         plt.savefig(path + '/../results/total_rewards_cem_' + make_full_string(params) + '.pdf')
#         plt.show()
#
# def exploit_critic_loss_full(params) -> None:
#     path = os.getcwd() + "/data/save"
#     study = params.gradients
#     for i in range(len(study)):
#         plot_data(path + "/critic_loss_" + study[i] + '_' + params.env_name + '.txt', "critic loss " + study[i])
#
#     plt.xlabel("Cycles")
#     plt.ylabel("Loss")
#     plt.legend(loc="upper right")
#     plt.title(params.env_name)
#     plt.savefig(path + '/../results/critic_loss_' + make_full_string(params) + 'pg.pdf')
#     plt.show()
#
#
# def exploit_policy_loss_full(params) -> None:
#     path = os.getcwd() + "/data/save"
#     study = params.gradients
#     for i in range(len(study)):
#         plot_data(path + "/policy_loss_" + study[i] + '_' + params.env_name + '.txt', "policy loss " + study[i])
#
#     plt.xlabel("Cycles")
#     plt.ylabel("Loss")
#     plt.legend(loc="lower right")
#     plt.title(params.env_name)
#     plt.savefig(path + '/../results/policy_loss_' + make_full_string(params) + 'pg.pdf')
#     plt.show()
#
# def exploit_nstep(params) -> None:
#     path = os.getcwd() + "/data/save"
#     steps = [1, 5, 10, 15, 20]
#
#     for j in ['policy_loss', 'critic_loss', 'reward', 'duration']:
#         mean_list = []
#         std_list = []
#         for i in steps:
#             mean, std = plot_data(path + '/' + j + '_nstep_' + str(i) + '_'
#                                   + params.env_name + '.txt', j + '_nstep_' + str(i))
#             mean_list.append(mean[-1])
#             std_list.append(std[-1])
#
#         plt.title(params.env_name)
#         plt.xlabel("Episodes")
#         plt.ylabel(j)
#         plt.legend(loc="lower right")
#         plt.savefig(path + '/../results/' + j + '_nstep_' + make_full_string(params) + '.pdf')
#         plt.show()
#
#         plt.plot(steps, mean_list, label="bias")
#         plt.plot(steps, std_list, label="variance")
#         plt.title(params.env_name)
#         plt.xlabel("N in N-step")
#         plt.ylabel('variance, bias')
#         plt.legend(loc="lower right")
#         plt.savefig(path + '/../results/bias_variance_' + j + '_' + make_full_string(params) + '.pdf')
#         plt.show()
#
#
# # to be refreshed
# def plot_beta_results(params) -> None:
#     path = os.getcwd() + "/data/save"
#     for beta in [0.1, 0.5, 1.0, 5.0, 10.0]:
#         plot_data(path + "/reward_" + str(beta) + '_' + params.env_name + '.txt', "reward " + str(beta))
#
#     plt.title(params.env_name)
#     plt.xlabel("Episodes")
#     plt.ylabel("Reward")
#     plt.legend(loc="lower right")
#     plt.savefig(path + '/../results/rewards_' + make_full_string(params) + '.pdf')
#     plt.show()
#
#
# def check_nstep(params) -> None:
#     path = os.getcwd() + "/data/save"
#     study1 = 'batchTD'
#     for j in ['loss', 'reward', 'duration']:
#         for i in [1]:
#             file_name = path + "/" + j + '_nstep_' + str(i) + '_' + params.env_name + '.txt'
#             mean, std = plot_data(file_name, j + '_nstep_' + str(i))
#             print('n:', i, ' mean :', mean[-1], ' std:', std[-1])
#         plot_data(path + "/" + j + '_' + study1 + '_' + params.env_name + '.txt', 'loss ' + study1)
#
#         plt.title(params.env_name)
#         plt.xlabel("Episodes")
#         plt.ylabel(j)
#         plt.legend(loc="lower right")
#         plt.savefig(path + '/../results/' + j + '_nstep_check.pdf')
#         plt.show()
#
#
# def exploit_nstep_diff(params) -> None:
#     path = os.getcwd() + "/data/save"
#     steps = [1, 5, 10, 20]
#     mean_list = []
#     std_list = []
#     for i in steps:
#         mean, std = plot_data(path + '/diff_' + str(i) + '_' + params.env_name + '.txt', 'nstep_' + str(i))
#         # print('n:', i, ' mean :', mean[-1], ' std:', std[-1])
#         mean_list.append(mean[-1])
#         std_list.append(std[-1])
#
#     plt.title(params.env_name)
#     plt.xlabel("Episodes")
#     plt.ylabel("diff")
#     plt.legend(loc="lower right")
#     plt.savefig(path + '/../results/diff_nstep_' + make_full_string(params) + '.pdf')
#     plt.show()
#
#     plt.plot(steps, mean_list, label="bias")
#     plt.plot(steps, std_list, label="variance")
#     plt.title(params.env_name)
#     plt.xlabel("N in N-step")
#     plt.ylabel('variance, bias')
#     plt.legend(loc="lower right")  # , bbox_to_anchor=(1, 0.5)
#     plt.savefig(path + '/../results/bias_variance_' + make_full_string(params) + '.pdf')
#     plt.show()
#
# def plot_results(params) -> None:
#     """
#     Plot the results from a study previously saved in files in "./data/save"
#     :param params: parameters of the study
#     :return: nothing
#     """
#     assert params.study_name in ['pg', 'regress', 'nstep'], 'unsupported study name'
#     if params.study_name == "pg":
#         exploit_duration_full(params)
#         exploit_reward_full(params)
#         exploit_policy_loss_full(params)
#         exploit_critic_loss_full(params)
#     elif params.study_name == "nstep":
#         exploit_nstep(params)
#
# if __name__ == '__main__':
#     args = get_args()
#     print(args)
#     plot_results(args)
