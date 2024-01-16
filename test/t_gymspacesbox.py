import gym
import numpy as np

a = np.array([[1, 2, 3],[4, 5, 6]])
b = a*2

space = gym.spaces.Box(a, b)

print(a.shape)