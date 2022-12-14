import numpy as np
import random
import gym
from .wrappers import Recorder, Monitor, Stack, FakeMultiReshape
from .subproc_vec_env import SubprocVecEnv
from .car_image_utils import to_grayscale, zero_center, crop


class Preprocess(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.env_step = 0
        self.rew_sum = 0

    def reset(self):
        obs, info = self.env.reset()
        obs = self.preprocess(obs)
        self.env_step = 0
        return obs, info

    def step(self, action):
        obs, rew, done, timeout, info = self.env.step(action)
        obs = self.preprocess(obs)
        # rew = np.clip(rew, a_min=None, a_max=1.0)  # clip
        self.env_step += 1
        if self.env_step >= 1000:
            done = True
        # or early terminate depends on past reward records
        return obs, rew, done, timeout, info

    @staticmethod
    def crop3(img, bottom=12, left=6, right=6):
        height, width, channel = img.shape
        return img[0: height - bottom, left: width - right, :]

    def render(self):
        state = self.env.render()
        state = self.crop3(state)
        return state

    @staticmethod
    def preprocess(state):
        preprocessed_state = to_grayscale(state)
        preprocessed_state = zero_center(preprocessed_state)
        preprocessed_state = crop(preprocessed_state)
        preprocessed_state = np.expand_dims(preprocessed_state, axis=-1)
        return preprocessed_state


def env_maker(env_name, ienv, env_seed, args):
    def __make_env():
        if args.env_type == 'discrete':
            env = gym.make(env_name, render_mode="rgb_array", continuous=False)
        elif args.env_type == 'continuous':
            env = gym.make(env_name, render_mode="rgb_array")  # rgb_array state_pixels
        else:
            raise NotImplementedError
        env = Preprocess(env)
        # env.seed(ienv + env_seed)
        random.seed(ienv + env_seed)
        np.random.seed(ienv + env_seed)
        env = Recorder(env, ienv, args)
        if args.render:
            env = Monitor(env, ienv, args)
        return env
    return __make_env


def get_environment(args):
    env = [env_maker(args.env_name, ienv, args.env_seed, args) for ienv in range(args.env_nums)]
    env = SubprocVecEnv(env)
    env = Stack(env, args)
    env.metadata['n_agents'] = 1
    env.metadata['observation_space'] = [env.observation_space]
    env.metadata['action_space'] = [env.action_space]
    print(env.observation_space.shape, env.action_space)
    env = FakeMultiReshape(env)
    return env
