import numpy as np
import gym
import math
from matplotlib import pyplot as plt

def eval(env):
    average_tot_score = []
    for j in range(900):
        state = env.reset()
        total_reward = 0
        for t in range(200):
            angle = math.degrees(math.acos(state[0]))
            velocity = state[2]
            action = expert_policy(angle, velocity)
            next_state, reward, done, _ = env.step([action])
            total_reward += reward
            state = next_state

            if done:
                print(total_reward)
                average_tot_score.append(total_reward)
                break
    return average_tot_score


def expert_policy(angle, velocity):
    if (velocity < (-0.053*angle)):
        if ((velocity <= 0) and (angle <= -50)):
            return -2
        else:
            return 2
    if (velocity >= (-0.053*angle)):
        if ((velocity >= 0) and (angle >= 50)):
            return 2
        else:
            return -2

env = gym.make("Pendulum-v0")
res=eval(env)
print(np.mean(np.array(res)))
plt.plot(res)
plt.show()
