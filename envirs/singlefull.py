import numpy as np
import gym, easydict, cv2, random, scipy, json, os
from envirs.wrappers import Recorder, Monitor
from envirs.wrappers import wrap_deepmind_render
from stable_baselines3.common.atari_wrappers import NoopResetEnv,MaxAndSkipEnv,EpisodicLifeEnv,FireResetEnv,WarpFrame,ClipRewardEnv
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
def env_maker(env_name, ienv, env_seed, args):
    def __make_env():
        if args.env_type=='atari':  env = make_atari(env_name)
        else:                       env = gym.make(env_name)
        env.seed(ienv+env_seed)
        random.seed(ienv+env_seed)
        np.random.seed(ienv+env_seed)
        env = Recorder(env, ienv, args)
        if args.env_type=='atari':
            if args.render: env = Monitor(env, ienv, args, 'org_')
            env = wrap_deepmind(env)
            env = wrap_deepmind_render(env)
        if args.render: env = Monitor(env, ienv, args)
        if hasattr(env,'attr'): env.spec._kwargs['attr']=env.attr
        return env
    return __make_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
def fEnv(args):
    with open('./myenv/envinfo.json', 'w') as fenvinfo:
        print(json.dumps(vars(args)),file=fenvinfo)
    env = [env_maker(args.env_name, ienv, args.env_seed, args) for ienv in range(args.env_num)]
    env = SubprocVecEnv(env)
    return env
