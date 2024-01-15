import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as kl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import global_policy

import tools_repro

class RSSM(tf.keras.Model):
    
    def __init__(self, stoch=30, deter=200, hidden=200, act=tf.nn.elu):
        super().__init__()
        
        self.stoch_size = 30
        self.deter_size = 200
        self.rnn_cell = kl.GRUCell(deter)
        self.obs1 = kl.Dense(hidden,activation=act)
        self.obs_mean = kl.Dense(stoch, activation=None)
        self.obs_std = kl.Dense(stoch, activation=None)
        self.img1 = kl.Dense(hidden, activation=act)
        self.img2 = kl.Dense(hidden, activation=act)
        self.img_mean = kl.Dense(stoch, activation=None)
        self.img_std = kl.Dense(stoch, activation=None)
        
    @tf.function
    def obs_step(self, prev_state, prev_action, embed):
        prior = self.img_step(prev_state, prev_action)
        x = tf.concat([prior['deter'], embed], -1)
        x = self.obs1(x)
        mean = self.obs_mean(x)
        std = self.obs_std(x)
        std = tf.nn.softplus(std)
        stoch = self.get_dist({'mean': mean, 'std': std}).sample()
        post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
        return post, prior
        
    @tf.function
    def img_step(self, prev_state, prev_action):
        x = tf.concat([prev_state['stoch'], prev_action], -1)
        x = self.img1(x)
        x, deter = self.rnn_cell(x, [prev_state['deter']])
        deter = deter[0]
        x = self.img2(x)
        mean = self.img_mean(x)
        std = self.img_std(x)
        std = tf.nn.softplus(std) + 0.1
        stoch = self.get_dist({'mean': mean, 'std': std}).sample()
        prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}
        return prior

    def get_dist(self, state):
        return tfd.MultivariateNormalDiag(state['mean'], state['std'])
    
    def get_feat(self, state):
        return tf.concat([state['stoch'], state['deter']], -1)
    
    def initial(self, batch_size):
        dtype = global_policy().compute_dtype
        return dict(
            mean=tf.zeros([batch_size, self.stoch_size], dtype),
            std=tf.zeros([batch_size, self.stoch_size], dtype),
            stoch=tf.zeros([batch_size, self.stoch_size], dtype),
            deter=self.rnn_cell.get_initial_state(None, batch_size, dtype))

    @tf.function
    def observe(self, embed, action, state=None):
        if state is None:
            state = self.initial(tf.shape(action)[0])
        last = (state, state)
        embed = tf.transpose(embed, [1, 0, 2])
        action = tf.transpose(action, [1, 0, 2])
        outputs = [[] for _ in tf.nest.flatten((state, state))]
        indices = range(len(tf.nest.flatten((action, embed))[0]))
        for index in indices:
            inp = tf.nest.map_structure(lambda x:x[index], (action, embed))
            last = self.obs_step(last[0], *inp)
            for o, l in zip(outputs, tf.nest.flatten(last)):
                o.append(l)
        outputs = [tf.stack(x, 0) for x in outputs]
        post, prior = tf.nest.pack_sequence_as((state, state), outputs)
        post = {k: tf.transpose(v, [1, 0, 2]) for k, v in post.items()}
        prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
        return post, prior
    
    @tf.function
    def imagine(self, action, state=None):
        if state is None:
            state = self.initial(tf.shape(action)[0])
        assert isinstance(state, dict), state
        action = tf.transpose(action, [1, 0, 2])
        last = state
        outputs = [[] for _ in tf.nest.flatten(state)]
        indices = range(len(tf.nest.flatten(action)[0]))
        for index in indices:
            last = self.img_step(last, action[index])
            for o, l in zip(outputs, tf.nest.flatten(last)):
                o.append(l)
        outputs = [tf.stack(x, 0) for x in outputs]
        prior = tf.nest.pack_sequence_as(state, outputs)
        prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
        return prior


class DenseEncoder(tf.keras.Model):
    
    def __init__(self, depth=32, act=tf.nn.relu):
        super().__init__()
        self._depth = depth
        self.h1_en = kl.Conv2D(1 * depth, 4, activation=act, strides=2)
        self.h2_en = kl.Conv2D(2 * depth, 4, activation=act, strides=2)
        self.h3_en = kl.Conv2D(4 * depth, 4, activation=act, strides=2)
        self.h4_en = kl.Conv2D(8 * depth, 4, activation=act, strides=2)
        
    def call(self, obs):
        # (batch_size,batch_length,64,64,3)>(batch_size*batchlength,64,64,3)
        x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
        x = self.h1_en(x) #入力shape: (1, 64, 64, 1) 出力shape: (1, 31, 31, 32)
        x = self.h2_en(x) #出力shape: (1, 14, 14, 64)
        x = self.h3_en(x) #出力shape: (1, 6, 6, 96)
        x = self.h4_en(x) #出力shape: (1, 2, 2, 128)
        shape = tf.concat([tf.shape(obs['image'])[:-3], [32 * self._depth]], 0)
        return tf.reshape(x, shape) # (batch_size,2,2,128)->(batch_size,1024)
        

class DenseDecoder(tf.keras.Model):

    def __init__(self, depth=32, act=tf.nn.relu, shape=(64, 64, 3)):
        super().__init__()
        self._depth = depth
        self._shape = shape
        self.h1_de = kl.Dense(32 * depth, activation = None)
        self.h2_de = kl.Conv2DTranspose(4 * depth, 5, activation=act, strides=2)
        self.h3_de = kl.Conv2DTranspose(2 * depth, 5, activation=act, strides=2)
        self.h4_de = kl.Conv2DTranspose(1 * depth, 6, activation=act, strides=2)
        self.h5_de = kl.Conv2DTranspose(shape[-1], 6, activation=None,strides=2)

    def call(self, features):
        x = self.h1_de(features)
        x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
        # Conv2D系は4次元の入力を想定しているため
        # reshape(batch_len,batch_size,1024)->(len*size,1,1,1024)
        x = self.h2_de(x) # 出力shape: (batch, 5, 5, 128)
        x = self.h3_de(x) # 出力shape: (batch, 13, 13, 64)
        x = self.h4_de(x) # 出力shape: (batch, 30, 30, 32)
        x = self.h5_de(x) # 出力shape: (batch, 64, 64, 3)
        shape = tf.concat([tf.shape(features)[:-1], self._shape], 0)
        mean = tf.reshape(x, shape) # 出力shape: (batch, 64, 64, 3)
        return tfd.Independent(tfd.Normal(mean, scale=1), len(self._shape))


class ActorNetwork(tf.keras.Model):
    
    def __init__(self, size, act=tf.nn.elu,
                 min_std=1e-4, init_std=5, mean_scale=5):
        super().__init__()
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.h1_act = kl.Dense(400, activation=act)
        self.h2_act = kl.Dense(400, activation=act)
        self.h3_act = kl.Dense(400, activation=act)
        self.h4_act = kl.Dense(400, activation=act)
        self.act_mean = kl.Dense(size, activation=None)
        self.act_std = kl.Dense(size, activation=None)
        
    def call(self, features):
        raw_init_std = np.log(np.exp(self.init_std) - 1)
        x = self.h1_act(features)
        x = self.h2_act(x)
        x = self.h3_act(x)
        x = self.h4_act(x)
        mean = self.act_mean(x)
        mean = self.mean_scale * tf.tanh(mean / self.mean_scale)
        std = self.act_std(x)
        std = tf.nn.softplus(std + raw_init_std) + self.min_std
        dist = tfd.Normal(mean, std)
        dist = tfd.TransformedDistribution(dist, tools_repro.TanhBijector())
        dist = tfd.Independent(dist, reinterpreted_batch_ndims=1)
        dist = tools_repro.SampleDist(dist)
        return dist


class ValueNetwork(tf.keras.Model):
    
    def __init__(self, shape=(), act=tf.nn.elu):
        super().__init__()
        self.shape = shape
        self.act = act
        self.h1_v = kl.Dense(400, activation=act)
        self.h2_v = kl.Dense(400, activation=act)
        self.h3_v = kl.Dense(400, activation=act)
        self.v = kl.Dense(1, activation=None)
        
    def call(self, features):
        x = self.h1_v(features)
        x = self.h2_v(x)
        x = self.h3_v(x)
        x = self.v(x)
        mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self.shape], 0))
        return tfd.Independent(tfd.Normal(mean, 1), len(self.shape))


class RewardDecoder(tf.keras.Model):
    
    def __init__(self, shape=(), act=tf.nn.elu):
        super().__init__()
        self.shape = shape
        self.act = act
        self.h1_r = kl.Dense(400, activation=act)
        self.h2_r = kl.Dense(400, activation=act)
        self.r = kl.Dense(1, activation=None)

    def call(self, features):
        x = self.h1_r(features)
        x = self.h2_r(x)
        x = self.r(x)
        mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self.shape], 0))
        return tfd.Independent(tfd.Normal(mean, 1), len(self.shape))




