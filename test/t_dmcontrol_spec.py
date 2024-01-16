from dm_control import suite
import gym
import matplotlib.pyplot as plt
import numpy as np

env = suite.load('walker', 'walk')

spec = env.action_spec()
print(spec.shape)
# print(spec.minimum, spec.maximum)



time_step = env.reset()

figure = env.physics.render(*(256, 256))

action = np.random.rand(6)
time_step = env.step(action)

print(time_step.discount)



plt.imshow(figure)
plt.axis('off')
plt.show()

