import numpy as np
import random
import gym
import json
from .wrappers import Recorder, Monitor
from .wrappers import WrapDeepmindRender
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, \
    EpisodicLifeEnv, FireResetEnv, WarpFrame, ClipRewardEnv


def make_atari(env_id):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env


def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    return env


class Stack(gym.Wrapper):
    def __init__(self, env, args):
        gym.Wrapper.__init__(self, env=env)
        self.args = args
        self.zero_stack = np.zeros([self.args.env_nums] +
                                   [self.args.stack_num] +
                                   list(env.observation_space.shape),
                                   dtype=env.observation_space.dtype)
        self.obs_stack = self.zero_stack.copy()

    def obs_stack_update(self, _new_obs, old_obs_stack):
        updated_obs_stack = np.roll(old_obs_stack, shift=-1, axis=1)
        updated_obs_stack[:, -1, :] = _new_obs  # [:]###
        return updated_obs_stack

    def reset(self):
        obs = self.env.reset()
        self.obs_stack = self.zero_stack.copy()
        self.obs_stack = self.obs_stack_update(obs, self.obs_stack)
        obs = self.obs_stack
        return obs

    def step(self, act):
        obs, rew, done, info = self.env.step(act)
        self.zero(done)
        self.obs_stack = self.obs_stack_update(obs, self.obs_stack)
        obs = self.obs_stack
        return obs, rew, done, info

    def zero(self, done):
        for i, done_i in enumerate(done):
            if done_i:
                self.obs_stack[i] *= 0  # [:-1]*=0


def env_maker(env_name, ienv, env_seed, args):
    def __make_env():
        if args.env_type == 'atari':
            env = make_atari(env_name)
        else:
            env = gym.make(env_name)
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
    with open('./games/gameinfo.json', 'w') as f:
        print(json.dumps(vars(args)), file=f)
    env = [env_maker(args.env_name, ienv, args.env_seed, args) for ienv in range(args.env_nums)]
    env = SubprocVecEnv(env)
    env = Stack(env, args)
    print(env.observation_space.shape, env.action_space)
    return env
