import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as kl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import global_policy

import tools_proprioception

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
    
    def __init__(self, act=tf.nn.relu):
        super().__init__()
        self.h1 = kl.Dense(200, activation=act)
        self.h2 = kl.Dense(200, activation=act)

    def call(self, obs):
        obs = {k: v for k, v in obs.items() if k not in ['image','reward','action']}
        for k, v in obs.items():
            if len(v.shape)==2:
                obs[k] = tf.expand_dims(v, axis=-1)
            elif len(v.shape)==0:
                obs[k] = tf.expand_dims(v, axis=-1)
        print(obs)
        obs = tf.concat(list(obs.values()), axis=-1)
        if len(obs.shape)==1:
            obs = tf.expand_dims(obs, axis=0)
        x = self.h1(obs)
        x = self.h2(x)
        return x


class DenseDecoder(tf.keras.Model):

    def __init__(self, shape, act=tf.nn.relu):
        super().__init__()
        self._shape = [shape]
        self.h1_de = kl.Dense(200, activation=act)
        self.h2_de = kl.Dense(shape, activation=None)

    def call(self, features):
        x = self.h1_de(features)
        mean = self.h2_de(x)
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
        dist = tfd.TransformedDistribution(dist, tools_proprioception.TanhBijector())
        dist = tfd.Independent(dist, reinterpreted_batch_ndims=1)
        dist = tools_proprioception.SampleDist(dist)
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




