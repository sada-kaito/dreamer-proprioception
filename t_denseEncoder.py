import tensorflow as tf
from tensorflow.keras import layers as kl
import numpy as np
import dreamer_proprioception
import models_proprioception
import wrappers_proprioception

class DenseEncoder(tf.keras.Model):

    def __init__(self, depth=32, act=tf.nn.relu):
        super().__init__()
        self.h1 = kl.Dense(20, activation=act)
        self.h2 = kl.Dense(20, activation=act)

    def call(self, obs):
        # print(obs)
        obs = {k: v for k, v in obs.items() if k not in ['image','reward','action']}
        for k, v in obs.items():
            if len(v.shape)==2:
                obs[k] = tf.expand_dims(v, axis=-1)
        obs = tf.concat(list(obs.values()), axis=-1)
        x = self.h1(obs)
        x = self.h2(x)
        return x

config = dreamer_proprioception.define_config()
datadir = config.logdir / 'episodes'
dataset = iter(dreamer_proprioception.load_dataset(datadir, config))
data = next(dataset)
# print(data.keys())
# print(data['orientation'])
# print(data['velocity'])
# print(data['image'])
encoder = DenseEncoder()
# encoder = models_repro.ConvEncoder()
embed = encoder(data)
# print(embed)

env = wrappers_proprioception.DeepMindControl('pendulum', 'swingup')
env = wrappers_proprioception.DeepMindControl('pendulum', 'swingup')
# env = wrappers_repro.DeepMindControl('cartpole', 'swingup')
# env = wrappers_repro.DeepMindControl('walker', 'run')
# env = wrappers_repro.DeepMindControl('walker', 'walk')

state = env.observation_space
# print(state)


# print(state_space)
state_space = [v for k, v in state.items() if k not in ['image','reward']]
# print(state_space)
state_dim = sum([v.shape[0] for v in state_space])
decoder = models_proprioception.DenseDecoder(state_dim)
# decoder = models_repro.ConvDecoder()

dynamics = models_proprioception.RSSM()


post, prior = dynamics.observe(embed, data['action'])
feat = dynamics.get_feat(post)
image_pred = decoder(feat)
obs = [v for k, v in data.items() if k not in ['image','reward','action']]
# print(obs)
# print(tf.concat(obs, axis=-1))
# print(image_pred)














