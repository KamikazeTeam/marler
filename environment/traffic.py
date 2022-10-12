import numpy as np
import random
import gym
from .wrappers import Monitor, Stack
from .subproc_vec_env import SubprocVecEnv
# import games
# from easydict import EasyDict


# class Preprocess(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#
#     def reset(self):
#         obs, info = self.env.reset()
#         return obs, info
#
#     def step(self, action):
#         obs, rew, done, timeout, info = self.env.step(action)
#         return obs, rew, done, timeout, info


class RecorderMulti(gym.Wrapper):
    def __init__(self, env, ienv, args):
        super().__init__(env)

    def reset(self):
        obs, info = self.env.reset()
        print(obs.shape)
        exit()
        return obs, info

    def step(self, action):
        obs, rew, done, timeout, info = self.env.step(action)
        return obs, rew, done, timeout, info


def env_maker(env_name, ienv, env_seed, args):
    def __make_env():
        import games
        env = gym.make(env_name, args=args.env_args, render=args.render)
        # env = Preprocess(env)
        # env.seed(ienv + env_seed)
        random.seed(ienv + env_seed)
        np.random.seed(ienv + env_seed)
        env = RecorderMulti(env, ienv, args)  #
        if args.render:
            env = Monitor(env, ienv, args)
        return env
    return __make_env


def get_environment(args):
    env = [env_maker(args.env_name, ienv, args.env_seed, args) for ienv in range(args.env_nums)]
    env = SubprocVecEnv(env)
    env = Stack(env, args)
    print(env.observation_space.shape, env.action_space)
    return env
