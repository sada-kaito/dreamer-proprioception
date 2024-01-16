import collections
import functools
import pathlib
import json
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.mixed_precision import global_policy
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.mixed_precision import Policy
from tensorflow_probability import distributions as tfd

import models_proprioception
import tools_proprioception
import wrappers_proprioception


tf.get_logger().setLevel('ERROR')

def define_config():
    config = tools_proprioception.AttrDict()
    # General
    config.logdir = pathlib.Path('.')
    config.load_model = False
    config.seed = 0
    config.steps = 5e6
    config.eval_every = 1e4
    config.log_scalars = True
    config.precision = 16
    
    # Environment
    config.domain = 'pendulum'
    config.task = 'swingup'
    config.action_count = 2
    config.time_limit = 1000
    config.prefill = 5000
    config.clip_rewards = 'none'
    
    # Model
    config.free_nats = 3.0
    config.kl_scale = 1.0
    
    # Training
    config.batch_size = 50
    config.batch_length = 50
    config.train_steps = 100
    config.model_lr = 6e-4
    config.value_lr = 8e-5
    config.actor_lr = 8e-5
    config.grad_clip = 100.0
    config.dataset_balance = False
    
    # Behavior
    config.discount = 0.99
    config.disclam = 0.95
    config.horizon = 15
    
    return config


class Dreamer(tf.keras.Model):
    
    def __init__(self, config, datadir, act, state, writer):
        super().__init__()
        self.c = config
        self.act_dim = act.shape[0]
        state_space = [v for k, v in state.items() if k not in ['image','reward']]
        self.state_dim = sum([v.shape[0] for v in state_space])
        self._metrics = collections.defaultdict(tf.metrics.Mean)
        self.float = global_policy().compute_dtype
        self._writer = writer
        self.dataset = iter(load_dataset(datadir, self.c))
        self.build_model()
    
    def build_model(self):
        self.encoder = models_proprioception.DenseEncoder()
        self.decoder = models_proprioception.DenseDecoder(self.state_dim)
        self.dynamics = models_proprioception.RSSM()
        self.reward = models_proprioception.RewardDecoder()
        self.value = models_proprioception.ValueNetwork()
        self.actor = models_proprioception.ActorNetwork(self.act_dim)
        model_modules = [self.encoder, self.dynamics, self.decoder, self.reward]
        Optimizer = functools.partial(tools_proprioception.Adam, clip=self.c.grad_clip)
        self.model_opt = Optimizer(model_modules, self.c.model_lr)
        self.value_opt = Optimizer([self.value], self.c.value_lr)
        self.actor_opt = Optimizer([self.actor], self.c.actor_lr)
        self._train(next(self.dataset))
        
    def train(self):
        for  train_step in range(self.c.train_steps):
            self._train(next(self.dataset))
            
    @tf.function
    def _train(self, data):
        with tf.GradientTape() as model_tape:
            embed = self.encoder(data)
            post, prior = self.dynamics.observe(embed, data['action'])
            feat = self.dynamics.get_feat(post)
            image_pred = self.decoder(feat)
            reward_pred = self.reward(feat)
            likes = tools_proprioception.AttrDict()
            obs = {k: v for k, v in data.items() if k not in ['image','reward','action']}
            for k, v in obs.items():
                if len(v.shape)==2:
                    obs[k] = tf.expand_dims(v, axis=-1)
            obs = tf.concat(list(obs.values()), axis=-1)
            likes.obs = tf.reduce_mean(image_pred.log_prob(obs))
            likes.reward = tf.reduce_mean(reward_pred.log_prob(data['reward']))
            prior_dist = self.dynamics.get_dist(prior)
            post_dist = self.dynamics.get_dist(post)
            div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            div = tf.maximum(div, self.c.free_nats)
            model_loss = self.c.kl_scale * div - sum(likes.values())
            
        with tf.GradientTape() as actor_tape:
            imag_feat = self._imagine_ahead(post)
            reward = self.reward(imag_feat).mode()
            gamma = self.c.discount * tf.ones_like(reward)
            value = self.value(imag_feat).mode()
            returns = tools_proprioception.lambda_return(
                reward[:-1], value[:-1], gamma[:-1], bootstrap=value[-1], 
                lambda_=self.c.disclam, horizon=self.c.horizon, axis=0)
            discount = tf.concat([tf.ones_like(gamma[:1]), gamma[:-2]], 0)
            discount = tf.stop_gradient(tf.math.cumprod(discount, 0))
            actor_loss = -tf.reduce_mean(discount * returns)
            
        with tf.GradientTape() as value_tape:
            value_pred = self.value(imag_feat)[:-1]
            target = tf.stop_gradient(returns)
            value_loss = -tf.reduce_mean(discount*value_pred.log_prob(target))
        
        model_norm = self.model_opt(model_tape, model_loss)
        actor_norm = self.actor_opt(actor_tape, actor_loss)
        value_norm = self.value_opt(value_tape, value_loss)
        
        if self.c.log_scalars:
            self._scalar_summaries(
                feat, prior_dist, post_dist, likes, div,
                model_loss, actor_loss, value_loss, 
                model_norm, actor_norm, value_norm)

    @tf.function
    def policy(self, obs, state, training):
        if state is None:
            latent = self.dynamics.initial(batch_size=1)
            action = tf.zeros((1, self.act_dim), self.float)
        else:
            latent, action = state
        embed = self.encoder(preprocess(obs, self.c))
        embed = tf.reshape(embed, (1,200))
        latent, _ = self.dynamics.obs_step(latent, action, embed)
        feat = self.dynamics.get_feat(latent)
        if training:
            action = self.actor(feat).sample()
        else:
            action = self.actor(feat).mode()
        action = tf.reshape(action, (1,1))
        state = (latent, action)
        return action, state


    def _imagine_ahead(self, post):
        flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in post.items()}
        last = start
        outputs = [[] for _ in tf.nest.flatten(start)]
        for index in range(self.c.horizon):
            action = self.actor(
                tf.stop_gradient(self.dynamics.get_feat(last))).sample()
            last = self.dynamics.img_step(last, action)
            for o, l in zip(outputs, tf.nest.flatten(last)):
                o.append(l)
        outputs = [tf.stack(x, 0) for x in outputs]
        outputs = tf.nest.pack_sequence_as(start, outputs)
        imag_feat = self.dynamics.get_feat(outputs)
        return imag_feat
    
    def _scalar_summaries(
        self, feat, prior_dist, post_dist, likes, div, model_loss,
        actor_loss, value_loss, model_norm, actor_norm, value_norm):
        self._metrics['model_grad_norm'].update_state(model_norm)
        self._metrics['actor_grad_norm'].update_state(actor_norm)
        self._metrics['value_grad_norm'].update_state(value_norm)
        self._metrics['prior_entropy'].update_state(prior_dist.entropy())
        self._metrics['post_entropy'].update_state(post_dist.entropy())
        self._metrics['obs_loss'].update_state(-likes['obs'])
        self._metrics['reward_loss'].update_state(-likes['reward'])
        self._metrics['kl_div'].update_state(div)
        self._metrics['model_loss'].update_state(model_loss)
        self._metrics['value_loss'].update_state(value_loss)
        self._metrics['actor_loss'].update_state(actor_loss)
        self._metrics['action_ent'].update_state(self.actor(feat).entropy())
        
    def write_summaries(self, step):
       metrics = [(k, float(v.result())) for k, v in self._metrics.items()]
       [m.reset_states() for m in self._metrics.values()]
       with (self.c.logdir / 'metrics.jsonl').open('a') as f:
         f.write(json.dumps({'step': step, **dict(metrics)}) + '\n')
       [tf.summary.scalar('agent/' + k, m) for k, m in metrics]
       print(f'[{step}]', ' / '.join(f'{k} {v:.1f}' for k, v in metrics))
       self._writer.flush()
       
    def save(self):
        os.makedirs('weights', exist_ok=True)
        self.encoder.save_weights('weights/encoder_weights',save_format='tf')
        self.decoder.save_weights('weights/decoder_weights',save_format='tf')
        self.dynamics.save_weights('weights/dynamics_weights',save_format='tf')
        self.reward.save_weights('weights/reward_weights',save_format='tf')
        self.value.save_weights('weights/value_weights',save_format='tf')
        self.actor.save_weights('weights/actor_weights',save_format='tf')
        
    def load(self):
        self.encoder.load_weights('weights/encoder_weights')
        self.decoder.load_weights('weights/decoder_weights')
        self.dynamics.load_weights('weights/dynamics_weights')
        self.reward.load_weights('weights/reward_weights')
        self.value.load_weights('weights/value_weights')
        self.actor.load_weights('weights/actor_weights')



def preprocess(obs, config):
    dtype = global_policy().compute_dtype
    obs = obs.copy()
    with tf.device('cpu:0'):
        obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
        clip_rewards = dict(none=lambda x: x, tanh=tf.tanh)[config.clip_rewards]
        obs['reward'] = clip_rewards(obs['reward'])
    return obs

def load_dataset(directory, config):
    episode = next(tools_proprioception.load_episodes(directory, 1))
    types = {k: v.dtype for k, v in episode.items()}
    shapes = {k: (None,) + v.shape[1:] for k, v in episode.items()}
    generator = lambda: tools_proprioception.load_episodes(
        directory, config.train_steps, config.batch_length)
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.map(functools.partial(preprocess, config=config))
    dataset = dataset.prefetch(10)
    return dataset

def count_steps(datadir, config):
  return tools_proprioception.count_episodes(datadir)[1] * config.action_count


def summarize_episode(episode, config, datadir, writer, prefix):
    episodes, steps = tools_proprioception.count_episodes(datadir)
    length = (len(episode['reward']) - 1) * config.action_count
    ret = episode['reward'].sum()
    print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
    metrics = [(f'{prefix}/return', float(episode['reward'].sum()))]
    step = count_steps(datadir, config)
    with (config.logdir / 'metrics.jsonl').open('a') as f:
        f.write(json.dumps(dict([('step', step)] + metrics))+ '\n')
    with writer.as_default():
        tf.summary.experimental.set_step(step)
        [tf.summary.scalar('sim/' + k, v) for k, v in metrics]

def make_env(config, writer, prefix, datadir, store):
    env = wrappers_proprioception.DeepMindControl(config.domain, config.task, camera='side')
    env = wrappers_proprioception.ActionRepeat(env, config.action_count)
    env = wrappers_proprioception.NormalizeActions(env)
    env = wrappers_proprioception.TimeLimit(env, config.time_limit / config.action_count)
    callbacks = []
    if store:
        callbacks.append(lambda ep: tools_proprioception.save_episodes(datadir, [ep]))
    callbacks.append(
        lambda ep: summarize_episode(ep, config, datadir, writer, prefix))
    env = wrappers_proprioception.Collect(env, callbacks, config.precision)
    env = wrappers_proprioception.RewardObs(env)
    return env
    

set_global_policy(Policy('mixed_float16'))

config = define_config()

def main(config):
    if config.precision == 16:
        set_global_policy(Policy('mixed_float16'))
    config.steps = int(config.steps)
    config.logdir.mkdir(parents=True, exist_ok=True)
    
    datadir = config.logdir / 'episodes'
    
    writer = tf.summary.create_file_writer(
        str(config.logdir), max_queue=1000, flush_millis=20000)
    writer.set_as_default()
    train_env = make_env(config, writer, 'train', datadir, store=True)
    test_env = make_env(config, writer, 'test', datadir, store=False)
    act_space = train_env.action_space # Box(-1.0, 1.0, (6,), float32)
    state_space = train_env.observation_space
    
    # prefill dataset with random episodes.
    step = count_steps(datadir, config)
    prefill = max(0, config.prefill - step)
    print(f'prefill dataset with {prefill} steps.')
    while prefill > count_steps(datadir, config):
        train_env.reset()
        done = False
        while not done:
            obs, reward, done = train_env.step(act_space.sample())
    
    # Train and evaluate the agent
    step = count_steps(datadir, config)
    print(f'Simulating agent for {config.steps-step} steps.')
    agent = Dreamer(config, datadir, act_space, state_space, writer)
    
    if config.load_model:
        print('load')
        agent.load()
        
    state = None
    done = None
    
    while step < config.steps:
        start_time = time.time()
        obs = test_env.reset()
        done = False
        state = None
        if step % config.eval_every == 0:
            while not done:
                action, state = agent.policy(obs, state, training=False)
                action = np.array(action)
                obs, _, done = test_env.step(action[0])

        obs = train_env.reset()
        done = False
        state = None
        while not done:
            action, state = agent.policy(obs, state, training=True)
            action = np.array(action)
            obs, _, done = train_env.step(action[0])
        
        writer.flush()
        agent.train()

        step = count_steps(datadir, config)
        agent.write_summaries(step)
        agent.save()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"プログラムの実行時間: {elapsed_time} 秒")
        

if __name__ == "__main__":
    main(config)
    pass





















