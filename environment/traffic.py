import numpy as np
import random
import gym
from .wrappers import Monitor, Stack
from .subproc_vec_env import SubprocVecEnv
import os


class Preprocess(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        obs, info = self.env.reset()
        obs = obs.reshape((obs.shape[0], obs.shape[1],
                           self.env.metadata['n_agents'], obs.shape[2]//self.env.metadata['n_agents'], *obs.shape[3:]))
        obs = np.transpose(obs, axes=[0, 1, 2, 4, 5, 3])
        obs = np.transpose(obs, axes=[2, 0, 1, 3, 4, 5])
        return obs, info

    def step(self, action):
        action = np.transpose(action, axes=[1, 0])
        obs, rew, done, timeout, info = self.env.step(action)
        rew = np.transpose(rew, axes=[1, 0])
        obs = obs.reshape((obs.shape[0], obs.shape[1],
                           self.env.metadata['n_agents'], obs.shape[2]//self.env.metadata['n_agents'], *obs.shape[3:]))
        obs = np.transpose(obs, axes=[0, 1, 2, 4, 5, 3])
        obs = np.transpose(obs, axes=[2, 0, 1, 3, 4, 5])
        return obs, rew, done, timeout, info


class RecorderMulti(gym.Wrapper):
    def __init__(self, env, ienv, args):
        gym.Wrapper.__init__(self, env=env)
        folder_name = args.exp_dir + 'rewards/'
        os.makedirs(folder_name, exist_ok=True)
        prefix = ''
        if args.test_steps:
            prefix = 'test_' + prefix
        if args.render:
            prefix = 'render_' + prefix
        self.f_rewards = open(folder_name + prefix + str(ienv), 'a')
        self.rewards = np.array([0.0 for _ in range(self.env.metadata['n_agents'])])
        self.g_step = 0
        self.g_step_plus = args.env_nums

    def __del__(self):
        print('', file=self.f_rewards, flush=True)
        self.f_rewards.close()

    def reset(self):
        self.rewards = np.array([0.0 for _ in range(self.env.metadata['n_agents'])])
        return self.env.reset()

    def step(self, act):
        obs, rew, done, timeout, info = self.env.step(act)
        self.g_step += self.g_step_plus
        self.rewards += rew
        if done:
            info = {'score': self.rewards, 'g_step': self.g_step, **info}
            rewards_string = '_'.join([str(int(reward)) for reward in self.rewards])
            print(self.g_step, ',', rewards_string, end='|', file=self.f_rewards, flush=True)
            self.rewards = np.array([0.0 for _ in range(self.env.metadata['n_agents'])])
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
    env = Preprocess(env)
    return env
