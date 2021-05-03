import os
import gym


class PerfWriter(gym.Wrapper):
    """
    This wrapper is used to save the performance and episode duration into a file in a transparent way
    Two flags are used to decide if one wants to save respectively performance and episode duration
    They can be set from outside anytime

    The set_file_name() function has to be called on the environment before starting to save data, specifying
    the file name
    Files are saved in "./data/save/"

    """
    def __init__(self, env):
        super(PerfWriter, self).__init__(env)

        self.duration = 0
        self.num_episode = 0
        self.total_reward = 0
        self.reward_flag = True
        self.duration_flag = True
        self.duration_file = None
        self.reward_file = None
        self.gradient_angles_file_names = None
        self.covariance_file = None
        self.angle_file = None
        self.distance_file = None

        self.directory = os.getcwd() + '/data/save/'
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def reinit(self):
        self.duration = 0
        self.num_episode = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.duration += 1
        self.total_reward += reward
        if done:
            if self.reward_flag:
                self.reward_file.write(str(self.num_episode) + ' ' + str(self.total_reward) + '\n')
            if self.duration_flag:
                self.duration_file.write(str(self.num_episode) + ' ' + str(self.duration) + '\n')
        return observation, reward, done, info

    def write_reward(self,cycle,reward):
        self.reward_file.write(str(cycle) + ' ' + str(reward) + '\n')

    def write_angles(self,cycle,angle):
        self.angle_file.write(str(cycle) + ' ' + str(angle) + '\n')
    def write_distances(self,cycle,distance):
        self.distance_file.write(str(cycle) + ' ' + str(distance) + '\n')


    def write_gradients(self, gradient_angles,cycle):
        for i in range(len(gradient_angles)):
            self.gradient_angles_file_names[cycle].write(str(i)+' '+str(gradient_angles[i])+ '\n')

    def write_cov(self, cycle, cov_norm):
        self.covariance_file.write(str(cycle)+' '+str(cov_norm)+'\n')


    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.duration = 0
        if self.reward_flag or self.duration_flag:
            self.num_episode += 1
        self.total_reward = 0
        return observation

    def close(self):
        if self.reward_file:
            self.reward_file.close()
        if self.duration_file:
            self.duration_file.close()
        if self.gradient_angles_file:
            self.gradient_angles_file.close()

    def set_reward_flag(self, val):
        self.reward_flag = val

    def set_duration_flag(self, val):
        self.duration_flag = val

    def set_file_name(self, name,nb_cycles = None):
        duration_name = self.directory + "duration_" + name + ".txt"
        self.duration_file = open(duration_name, "w")
        reward_name = self.directory + "reward_" + name + ".txt"
        self.reward_file = open(reward_name, "w")
        angles_name = self.directory + "angle_"+name +".txt"
        self.angle_file = open(angles_name, "w")
        if nb_cycles!=None:
            gradient_angles_names = [self.directory +"gradient_angles_" + "#" + str(cycle) +".txt" for cycle in range(nb_cycles)]
            self.gradient_angles_file_names = [open(gradient_angles_name, "w") for gradient_angles_name in gradient_angles_names]

        else:
            covariance_name = self.directory + "covariance_"  + ".txt"
            self.covariance_file = open(covariance_name, "w")
        distances_name = self.directory + "distance_"+name +".txt"
        self.distance_file = open(distances_name, "w")
