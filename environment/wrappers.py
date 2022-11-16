import numpy as np
import gym
import cv2
import os


class FakeMultiReshape(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        obs, info = self.env.reset()
        return [obs], info

    def step(self, action):
        action = action[0]
        obs, rew, done, timeout, info = self.env.step(action)
        return [obs], [rew], done, timeout, info


class Stack(gym.Wrapper):
    def __init__(self, env, args):
        gym.Wrapper.__init__(self, env=env)
        self.args = args
        self.zero_stack = np.zeros([self.args.env_nums] +
                                   [self.args.stack_num] +
                                   list(env.observation_space.shape),
                                   dtype=env.observation_space.dtype)
        self.obs_stack = self.zero_stack.copy()

    @staticmethod
    def obs_stack_update(_new_obs, old_obs_stack):
        updated_obs_stack = np.roll(old_obs_stack, shift=-1, axis=1)
        updated_obs_stack[:, -1, :] = _new_obs  # [:]###
        return updated_obs_stack

    def reset(self):
        obs, info = self.env.reset()
        self.obs_stack = self.zero_stack.copy()
        self.obs_stack = self.obs_stack_update(obs, self.obs_stack)
        obs = self.obs_stack
        return obs, info

    def step(self, act):
        obs, rew, done, timeout, info = self.env.step(act)
        self.zero(done)
        self.obs_stack = self.obs_stack_update(obs, self.obs_stack)
        obs = self.obs_stack
        return obs, rew, done, timeout, info

    def zero(self, done):
        for i, done_i in enumerate(done):
            if done_i:
                self.obs_stack[i] *= 0  # [:-1]*=0

    def render(self):
        frame = self.env.render(mode='rgb_array')
        # else render window will be open (by SubprocVecEnv setting?)
        return frame


class Recorder(gym.Wrapper):
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
        self.rewards = 0
        self.g_step = 0
        self.g_step_plus = args.env_nums

    def __del__(self):
        print('', file=self.f_rewards, flush=True)
        self.f_rewards.close()

    def reset(self):
        self.rewards = 0
        return self.env.reset()

    def step(self, act):
        obs, rew, done, timeout, info = self.env.step(act)
        self.g_step += self.g_step_plus
        self.rewards += rew
        if done:
            info = {'score': int(self.rewards), 'g_step': self.g_step, **info}
            print(self.g_step, ',', int(self.rewards), end='|', file=self.f_rewards, flush=True)
            self.rewards = 0
        return obs, rew, done, timeout, info


class Monitor(gym.Wrapper):
    def __init__(self, env, ienv, args, prefix=''):
        gym.Wrapper.__init__(self, env=env)
        video_name = args.exp_dir + prefix + str(args.env_seed) + '_' + str(ienv) + '.mp4'
        fps, fourcc = args.fps, cv2.VideoWriter_fourcc(*'mp4v')  # 'M','J','P','G')
        if hasattr(env, 'screen_size'):
            screen_size = env.screen_size
        else:
            screen_size = self.env.observation_space.shape[:2]
        if hasattr(env, 'screen_nums'):
            screen_nums = env.screen_nums
        else:
            screen_nums = [1, 1]
        # if hasattr(env, 'zoom_in'):     zoom_in     = args.zoom_in*env.zoom_in
        # else:                           zoom_in     = args.zoom_in
        if args.height != -1 and args.width != -1:
            height, width = args.height, args.width
        else:
            height, width = int(screen_size[0] * args.zoom_in * screen_nums[0]),\
                            int(screen_size[1] * args.zoom_in * screen_nums[1])
        if args.env_type == 'atari' and prefix == 'org_':
            height, width = 210, 160
        self.vWriter = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
        self.debug = True
        self.debug_one_screen_size, self.debug_height, self.debug_width = screen_size, height, width

    def __del__(self):
        self.vWriter.release()

    def render(self):
        frame = self.env.render()
        if self.debug:
            print('debug_one_screen_size:', self.debug_one_screen_size)
            print('debug_video_size_h_w : ', self.debug_height, '', self.debug_width)
            print('debug     frame.shape:', frame.shape)
            self.debug = False
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.vWriter.write(frame)
        return frame


class WrapDeepmindRender(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env=env)

    def render(self):
        frame = self.env.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)  # width,height
        frame = np.expand_dims(frame, -1)
        frame = np.tile(frame, 3)
        return frame
