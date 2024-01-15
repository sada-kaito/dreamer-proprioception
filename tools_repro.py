
import datetime
import io
import pathlib
import pickle
import re
import uuid

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from tensorflow_probability import distributions as tfd


class AttrDict(dict):

  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__

class TanhBijector(tfp.bijectors.Bijector):
    
    def __init__(self, validate_args=False, name='tanh'):
        super().__init__(
            forward_min_event_ndims=0,
            validate_args=validate_args, name=name)
        
    def _forward(self, x):
        return tf.nn.tanh(x)
    
    def _inverse(self, y):
        dtype = y.dtype
        y = tf.cast(y, tf.float32)
        y = tf.where(tf.less_equal(tf.abs(y), 1),
                     tf.clip_by_value(y, -0.99999997, 0.99999997), y)
        y = tf.atanh(y)
        y = tf.cast(y, dtype)
        return y
    
    def _forward_log_det_jacobian(self, x):
        log2 = tf.math.log(tf.constant(2.0, dtype=x.dtype))
        return 2.0 * (log2 - x - tf.nn.softplus(-2.0 * x))
    

class Adam(tf.Module):

    def __init__(self, modules, lr, clip=None):
        self.modules = modules
        self.clip = clip
        self.opt = tf.optimizers.Adam(lr)
        self.opt = LossScaleOptimizer(self.opt, dynamic=True)
        self._variables = None

    def __call__(self, tape, loss):
        if self._variables is None:
            variables = [module.variables for module in self.modules]
            self._variables = tf.nest.flatten(variables)
        assert len(loss.shape) == 0, loss.shape
        with tape:
            loss = self.opt.get_scaled_loss(loss)
        grads = tape.gradient(loss, self._variables)
        grads = self.opt.get_unscaled_gradients(grads)
        norm = tf.linalg.global_norm(grads)
        if self.clip:
            grads, _ = tf.clip_by_global_norm(grads, self.clip, norm)
        self.opt.apply_gradients(zip(grads, self._variables))
        return norm


def save_episodes(directory, episodes):
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    for episode in episodes:
        identifer = str(uuid.uuid4().hex)[:8]
        length = len(episode['reward'])
        file = directory / f'{timestamp}-{identifer}-{length}.npz'
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with file.open('wb') as f2:
                f2.write(f1.read())


def load_episodes(directory, train_steps, batch_len=None, seed=0):
    directory = pathlib.Path(directory).expanduser()
    random = np.random.default_rng(seed)
    cache = {}
    while True:
        for filename in directory.glob('*.npz'):
            if filename not in cache:
                try:
                    with filename.open('rb') as f:
                        episode = np.load(f)
                        episode = {k: episode[k] for k in episode.keys()}
                except Exception as e:
                    print(f'Could not load episode: {e}')
                    continue
                cache[filename] = episode
        keys = list(cache.keys())
        for index in random.choice(len(keys), train_steps):
            episode = cache[keys[index]]
            if batch_len:
                total = len(next(iter(episode.values())))
                available = total - batch_len
                index = random.integers(0, available)
                episode = {k: v[index: index + batch_len] for k, v in episode.items()}
            yield episode


class SampleDist:
    
    def __init__(self, dist, samples=100):
        self.dist = dist
        self.samples = samples
        
    def __getattr__(self, name):
        return getattr(self.dist, name)

    def mean(self):
        samples = self.dist.sample(self.samples)
        return tf.reduce_mean(samples, 0)
    
    def mode(self):
        sample = self.dist.sample(self.samples)
        logprob = self.dist.log_prob(sample)
        return tf.gather(sample, tf.argmax(logprob))
    
    def entropy(self):
        sample = self.dist.sample(self.samples)
        logprob = self.dist.log_prob(sample)
        return -tf.reduce_mean(logprob, 0)



# def lambda_return(reward, value, bootstrap, lambda_, axis):
#     assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)

def lambda_return(
        reward, value, gamma, bootstrap, lambda_, horizon, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    # reward shape and value shape: (horizon, batch_len*batch_size)
    assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    # reward:(r1,...,r14), value:(v1,...,v14), bootstrap:v15
    next_values = tf.concat([value[1:], bootstrap[None, :]], 0)
    # next_values:(v2,...,v15)
    inputs = reward + gamma * next_values * (1 - lambda_)
    # inputs: (r1 + gamma*v2*(1-lambda), ...)
    v_lambda = bootstrap
    outputs = [[] for _ in tf.nest.flatten(bootstrap)]
    indices = range(horizon - 1)
    indices = reversed(indices)
    for index in indices:
        v_lambda = inputs[index] + gamma[index] * lambda_ * v_lambda
        for o, l in zip(outputs, tf.nest.flatten(v_lambda)):
            o.append(l)
    outputs = [list(reversed(x)) for x in outputs]
    outputs = [tf.stack(x, 0) for x in outputs]
    returns = tf.nest.pack_sequence_as(bootstrap, outputs)
    return returns


def count_episodes(directory):
  filenames = directory.glob('*.npz')
  lengths = [int(n.stem.rsplit('-', 1)[-1]) - 1 for n in filenames]
  episodes, steps = len(lengths), sum(lengths)
  return episodes, steps


def simulate(agent, envs, steps=0, episodes=0, state=None):
  # Initialize or unpack simulation state.
  if state is None:
    step, episode = 0, 0
    done = np.ones(1, bool)
    length = np.zeros(len(envs), np.int32)
    obs = [None] * 1
    agent_state = None
  else:
    step, episode, done, length, obs, agent_state = state
  while (steps and step < steps) or (episodes and episode < episodes):
    # Reset envs if necessary.
    if done.any():
      indices = [index for index, d in enumerate(done) if d]
      promises = [envs[i].reset(blocking=False) for i in indices]
      for index, promise in zip(indices, promises):
        obs[index] = promise()
    # Step agents.
    obs = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
    action, agent_state = agent(obs, done, agent_state)
    action = np.array(action)
    # Step envs.
    promises = [e.step(a, blocking=False) for e, a in zip(envs, action)]
    obs, _, done = zip(*[p()[:3] for p in promises])
    obs = list(obs)
    done = np.stack(done)
    episode += int(done.sum())
    length += 1
    step += (done * length).sum()
    length *= (1 - done)
  # Return new state to allow resuming the simulation.
  return (step - steps, episode - episodes, done, length, obs, agent_state)




















