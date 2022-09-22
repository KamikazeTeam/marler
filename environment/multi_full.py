import numpy as np
import random
import gym
import json
from .wrappers import Recorder, Monitor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import easydict
import cv2
import importlib
import pettingzoo as zoo
import supersuit as ss


class ShowState(zoo.utils.BaseParallelWraper):
    def __init__(self, env, showstateN):
        super().__init__(env)
        self.debug = True
        self.max_step = showstateN
        self.g_step = 0

    def printstate(self, state):
        print(self.g_step, 'state shape', self.state().shape)
        if len(state.shape) == 3:
            for i in range(state.shape[-1]):
                print(state[:, :, i])
        else:
            print(state)

    def printobs(self, obs):
        print(self.g_step, 'obs length', len(obs.items()))
        for key, value in obs.items():
            print('key', key, 'obs shape', value.shape)
            if len(value.shape) == 3:
                for i in range(value.shape[-1]):
                    print(value[:, :, i].transpose(1, 0))
                # break
            else:
                print(value)

    def reset(self):
        res = self.env.reset()
        self.agents = self.env.agents
        if self.debug: self.printstate(self.state())
        if self.debug: self.printobs(res)
        return res

    def step(self, actions):
        res = self.env.step(actions)
        self.agents = self.env.agents
        if self.debug: print('actions', actions)
        if self.debug: print('rewards', res[1])
        self.g_step += 1
        if self.debug: self.printstate(self.state())
        if self.debug: self.printobs(res[0])
        if self.g_step >= self.max_step:
            self.debug = False
            exit()
        return res


class Comm(zoo.utils.BaseParallelWraper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        res = self.env.reset()
        self.agents = self.env.agents
        return res

    def step(self, actions):
        res = self.env.step(actions)
        self.agents = self.env.agents
        return res


def env_maker(env_name, ienv, env_seed, args):
    def __make_env():
        try:
            mod_preprocess = importlib.import_module('envirs.preprocess.' + args.env_type + '-' + args.env_name)
            env_kwargs, env_config = mod_preprocess.env_setting(args)
            Recorder_before = mod_preprocess.Recorder_before
            Recorder_after = mod_preprocess.Recorder_after
        except:
            print(
                'No module named envirs.preprocess.' + args.env_type + '-' + args.env_name + ', using default preprocess (None).')
            env_kwargs = easydict.EasyDict()
            env_config = easydict.EasyDict()
            Recorder_before = Do_Nothing
            Recorder_after = Do_Nothing
        vars(args).update(env_kwargs)
        vars(args).update(env_config)
        mod = importlib.import_module('zoo.' + args.env_type + '.' + args.env_name)
        env = mod.parallel_env(**env_kwargs)
        env.seed(ienv + env_seed)
        random.seed(ienv + env_seed)
        np.random.seed(ienv + env_seed)
        # env.env_kwargs = env_kwargs
        # env.env_config = env_config
        env = Comm(env)
        if 'showstateN' in env_config and env_config.showstateN: env = ShowState(env, env_config.showstateN)
        if 'paddingobs' in env_config and env_config.paddingobs: env = ss.pad_observations_v0(env)
        if 'paddingact' in env_config and env_config.paddingact: env = ss.pad_action_space_v0(env)
        # from zoo.utils.conversions import from_parallel, to_parallel
        # env = from_parallel(env)
        env = ss.black_death_v2(
            env)  # when an agent die, other teammate's attack_penalty and opponent's attack_opponent_reward are increased...
        # after an agent die, there are extremely large reward to killer in the next step and large penalty to teammate in the next step...
        # after an agent die, there are bugs in the next step...
        # env = to_parallel(env)
        if 'num_frames' in env_config and env_config.num_frames: env = ss.frame_stack_v1(env, env_config.num_frames)
        # is it must before pettingzoo_env_to_vec_env_v1?
        env = ss.pettingzoo_env_to_vec_env_v1(env)

        if len(env.observation_space.shape) == 1:
            highvalue = np.max(env.observation_space.high)
            lowvalue = np.min(env.observation_space.low)
            ones = np.ones(env.observation_space.shape)
            ones = np.expand_dims(ones, -1)
            ones = np.expand_dims(ones, -1)
            high = highvalue * ones
            low = lowvalue * ones
            env.observation_space = gym.spaces.Box(high=np.float32(high), low=np.float32(low))
        env = Recorder_before(env, args)
        env = Recorder(env, ienv, args)
        env = Recorder_after(env, args)
        if args.render:
            if 'screen_size' in env_config:
                env.screen_size = env_config.screen_size
            env = Monitor_before(env, args)
            env = Monitor(env, ienv, args)
        if hasattr(env, 'attr'): env.spec._kwargs['attr'] = env.attr
        return env

    return __make_env


def get_environment(args):
    with open('./games/gameinfo.json', 'w') as f:
        print(json.dumps(vars(args)), file=f)
    env = [env_maker(args.env_name, ienv, args.env_seed, args) for ienv in range(args.env_nums)]
    env = SubprocVecEnv(env)
    # env = Stack(env, args)
    print(env.observation_space.shape, env.action_space)
    return env


class Monitor_before(gym.Wrapper):
    def __init__(self, env, args):
        gym.Wrapper.__init__(self, env=env)
        self.zoom_in = args.zoom_in
        self.r_step = 0
        self.fontdict = {'fontFace': cv2.FONT_HERSHEY_SIMPLEX, 'fontScale': 0.5, 'color': (0, 255, 0), 'thickness': 1,
                         'lineType': cv2.LINE_4}

    def reset(self):
        obs = self.env.reset()
        self.act, self.rew, self.obs = [0], [0], obs
        self.info = {'render': {}}
        return obs

    def step(self, act):
        obs, rew, done, info = self.env.step(act)
        self.r_step += 1
        self.act, self.rew, self.obs = act, rew, obs
        self.info.update(info)
        return obs, rew, done, info

    def render_text(self, frame, text):
        cv2.putText(frame, text=text, org=(0, self.startline), **self.fontdict)
        self.startline += self.splitline

    def render(self, mode):
        frame = self.env.render(mode)
        frame = self.env.render(mode)
        frame = self.env.render(mode)
        frame = cv2.resize(frame, (int(frame.shape[1] * self.zoom_in), int(frame.shape[0] * self.zoom_in)),
                           interpolation=cv2.INTER_AREA)
        self.startline, self.splitline = 20, 20
        self.render_text(frame, text='Frame:' + str(self.r_step))
        self.render_text(frame, text='Act C:' + '|'.join([str(format(act, '02d')) for act in self.act]))
        self.render_text(frame, text='Rew C:' + '|'.join([str(format(rew, '+.4f')) for rew in self.rew]))
        self.render_text(frame, text='TeamC:' + '|'.join([str(format(rew, '+.4f')) for rew in self.env.rewards]))
        # for i,obs in enumerate(self.obs):
        #     self.render_text(frame,text='obs:'+'|'.join([str(np.round(ob,2)) for ob in obs]))
        for key, value in self.info['render'].items():
            self.render_text(frame, text=key + ':' + value)
        return frame


class Do_Nothing(gym.Wrapper):
    def __init__(self, env, args):
        gym.Wrapper.__init__(self, env=env)

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, act):
        obs, rew, done, info = self.env.step(act)
        return obs, rew, done, info
