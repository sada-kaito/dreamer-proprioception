import gym
import numpy as np

class DeepMindControl:
    
    def __init__(self, domain, task, size=(64, 64), camera=None):
        if isinstance(domain, str):
            from dm_control import suite
            self.env = suite.load(domain, task)
        else:
            assert task is None
            self.env = domain()
        self.size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self.camera = camera

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self.env.observation_spec().items():
            spaces[key] = gym.spaces.Box(
                -np.inf, np.inf, value.shape, dtype=np.float32)
        spaces['image'] = gym.spaces.Box(
            0, 255, self.size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)
    
    @property
    def action_space(self):
        spec = self.env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
    
    def step(self, action):
        time_step = self.env.step(action)
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        reward = time_step.reward or 0
        done = time_step.last()
        return obs, reward, done
    
    def reset(self):
        time_step = self.env.reset()
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        return obs
    
    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self.env.physics.render(*self.size, camera_id=self.camera)       
    

class ActionRepeat:
    
    def __init__(self, env, action_count):
        self.env = env
        self.action_count = action_count
        
    def __getattr__(self, name):
        return getattr(self.env, name)
        
    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self.action_count and not done:
            obs, reward, done = self.env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done


class NormalizeActions:
    
    def __init__(self, env):
        self.env = env
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low),
            np.isfinite(env.action_space.high))
        # どちらかが有限でなかった場合，その要素を±1に収める
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def action_space(self):
        # マスクの要素が真(もともと有限の値)であったら±1を入れ，
        # 他はそのまま,結局全部の要素が±1になっている？
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = ((action + 1) / 2) * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)


class TimeLimit:
    
    def __init__(self, env, duration):
        self.env = env
        self.duration = duration #　time_limit/action_repeat
        self._step = None
        
    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def step(self, action):
        assert self._step is not None, 'Must reset environment'
        obs, reward, done = self.env.step(action)
        self._step += 1
        if self._step >= self.duration:
            done = True
            self._step = None
        return obs, reward, done
    
    def reset(self):
        self._step = 0
        return self.env.reset()

# step毎に得られるobs,rewardを保存しておく．1エピソード毎に別ファイルに保存される．
class Collect:
    
    def __init__(self, env, callbacks=None, precision=32):
        self.env = env
        self.callbacks = callbacks
        self.precision = precision
        self.episode = None
        
    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def step(self, action):
        obs, reward, done = self.env.step(action)
        obs = {k: self._convert(v) for k, v in obs.items()}
        transition = obs.copy()
        transition['action'] = action
        transition['reward'] = reward
        self.episode.append(transition)
        if done:
            episode = {k: [t[k] for t in self.episode] for k in self.episode[0]}
            episode = {k: self._convert(v) for k, v in episode.items()}
            for callback in self.callbacks:
                callback(episode)
        return obs, reward, done
    
    def reset(self):
        obs = self.env.reset()
        transition = obs.copy()
        transition['action'] = np.zeros(self.env.action_space.shape)
        transition['reward'] = 0.0
        self.episode = [transition]
        return obs
        
        
    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self.precision]
        elif np.issubdtype(value.dtype, np.signedinteger):
            dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self.precision]
        elif np.issubsctype(value.dtype, np.uint8):
            dtype = np.uint8
        else:
            raise NotImplementedError(value.dtype)
        return value.astype(dtype)

# obsにreward要素を追加するメソッド
class RewardObs:
    
    def __init__(self, env):
        self.env = env
        
    def __getattr__(self, name):
        return getattr(self.env, name)
    
    @property
    def observation_space(self):
        spaces = self.env.observation_space.spaces
        assert 'reward' not in spaces
        spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
        return gym.spaces.Dict(spaces)
    
    def step(self, action):
        obs, reward, done = self.env.step(action)
        obs['reward'] = reward
        return obs, reward, done
    
    def reset(self):
        obs = self.env.reset()
        obs['reward'] = 0.0
        return obs
        


















