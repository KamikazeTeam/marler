import numpy as np
import random
import gym
import json
from .wrappers import Recorder, Monitor, Stack
from .wrappers import WrapDeepmindRender
from .subproc_vec_env import SubprocVecEnv
from .atari_wrappers import NoopResetEnv, MaxAndSkipEnv, \
    EpisodicLifeEnv, FireResetEnv, WarpFrame, ClipRewardEnv


def make_atari(env_id):
    env = gym.make(env_id, render_mode='rgb_array')
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env


def wrap_deepmind(env, episode_life=True, clip_rewards=True):  # , frame_stack=False, scale=False):
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    return env


def env_maker(env_name, ienv, env_seed, args):
    def __make_env():
        if args.env_type == 'atari':
            env = make_atari(env_name)
        else:
            env = gym.make(env_name, render_mode='rgb_array')
        env.seed(ienv + env_seed)
        random.seed(ienv + env_seed)
        np.random.seed(ienv + env_seed)
        env = Recorder(env, ienv, args)
        if args.env_type == 'atari':
            if args.render:
                env = Monitor(env, ienv, args, 'org_')
            env = wrap_deepmind(env)
            env = WrapDeepmindRender(env)
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
