import gym
env = gym.make('LunarLanderContinuous-v2')
env.reset()
for _ in range(1000):

    env.render()
    s, r, done, info = env.step(env.action_space.sample()) # take a random
    if done == True:
        break


import gym
env = gym.make('LunarLanderContinuous-v2')
env.reset()

import numpy as np
a = np.array([0,0])
for _ in range(100):
    s, r, done, info = env.step(a) # take a random
    env.render()
    print(r)
    print(s)
    if done == True:
        break


