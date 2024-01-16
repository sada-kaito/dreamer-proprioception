import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as kl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import global_policy

class RSSM(tf.keras.Model):
    
    def __init__(self, stoch=30, deter=200, hidden=200, act=tf.nn.elu):
        super().__init__()
        
        self.rnn_cell = kl.GRUCell(deter)
        self.obs1 = kl.Dense(hidden,activation=act)
        self.obs_mean = kl.Dense(stoch, activation=None)
        self.obs_std = kl.Dense(stoch, activation=None)
        self.img1 = kl.Dense(hidden, activation=act)
        self.img2 = kl.Dense(hidden, activation=act)
        self.img_mean = kl.Dense(stoch, activation=None)
        self.img_std = kl.Dense(stoch, activation=None)
        
    def obs_step(self, prev_state, prev_action, embed):
        prior = self.img_step(prev_state, prev_action)
        x = tf.concat(prior['deter'], embed, -1)
        x = self.obs1(x)
        mean = self.obs_mean(x)
        std = self.obs_std(x)
        std = tf.nn.softplus(std)
        stoch = self.getdist({'mean': mean, 'std': std}).sample()
        post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
        return post, prior
        
    def img_step(self, prev_state, prev_action):
        x = tf.concat([prev_state['stoch'], prev_action], -1)
        x = self.img1(x)
        x, deter = self.rnn_cell(x, [prev_state['deter']])
        x = self.img2(x)
        mean = self.img_mean(x)
        std = self.img_std(x)
        std = tf.nn.softplus(std) + 0.1
        stoch = self.get_dist({'mean': mean, 'std': std}).sample()
        prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}
        return prior

    def get_dist(self, state):
        return tfd.MultivariateNormalDiag(state['mean'], state['std'])


class ConvEncoder(tf.keras.Model):
    
    def __init__(self, depth=32, act=tf.nn.relu):
        super().__init__()
        self.h1 = kl.Conv2D(1 * depth, 4, activation=act, strides=2)
        self.h2 = kl.Conv2D(2 * depth, 4, activation=act, strides=2)
        self.h3 = kl.Conv2D(4 * depth, 4, activation=act, strides=2)
        self.h4 = kl.Conv2D(8 * depth, 4, activation=act, strides=2)
        
    def call(self, x):
        x = self.h1(x) #入力shape: (1, 64, 64, 1) 出力shape: (1, 31, 31, 32)
        x = self.h2(x) #出力shape: (1, 14, 14, 64)
        x = self.h3(x) #出力shape: (1, 6, 6, 96)
        x = self.h4(x) #出力shape: (1, 2, 2, 128)
        shape = tf.concat([tf.shape(x)[:-3], [32 * depth]], 0)
        return tf.reshape(x, shape) # (batch_size, 2, 2, 128)->(batch_size, 1024)
        


# RSSMtest
state = {'stoch': tf.ones([1, 30]), 'deter': tf.ones([1, 200])}
action = tf.zeros([1, 30])
agent = RSSM()
prior = agent.img_step(state, action)
dist = agent.get_dist(prior)
# print(dist.sample())
# print(prior['mean'])

# ConvEncodertest
figure = tf.random.uniform((3, 64, 64, 3), minval=0, maxval=1)
# Conv2Dへの入力shape(batch_size, height, width, channels)
# print(figure.shape)  # 出力: (1, 64, 64)
# print(figure.dtype)  # 出力: <dtype: 'float32'>
# fig_numpy = figure.numpy()
# print(fig_numpy)
encoder = ConvEncoder()
x = encoder(figure)
# print(x.shape)
depth = 32

shape = tf.concat([tf.shape(figure)[:-3], [32 * depth]], 0)
# print(shape)