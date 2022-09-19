import numpy as np
import random
import gym
import json
from .wrappers import Recorder, Monitor, Stack
from .subproc_vec_env import SubprocVecEnv


def env_maker(env_name, ienv, env_seed, args):
    def __make_env():
        if args.env_type == 'discrete':
            env = gym.make(env_name, render_mode="state_pixels", continuous=False)  #
        else:
            env = gym.make(env_name, render_mode="state_pixels")
        # env.seed(ienv + env_seed)
        random.seed(ienv + env_seed)
        np.random.seed(ienv + env_seed)
        env = Recorder(env, ienv, args)
        if args.render:
            env = Monitor(env, ienv, args)
        return env
    return __make_env


def get_environment(args):
    with open('./games/gameinfo.json', 'w') as f:
        print(json.dumps(vars(args)), file=f)
    env = [env_maker(args.env_name, ienv, args.env_seed, args) for ienv in range(args.env_nums)]
    env = SubprocVecEnv(env)
    env = Stack(env, args)
    print(env.observation_space.shape, env.action_space)
    return env
