import numpy as np
import gym
import cv2
import os


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
        obs = self.env.reset()
        self.rewards = 0
        return obs

    def step(self, act):
        obs, rew, done, info = self.env.step(act)
        self.g_step += self.g_step_plus
        self.rewards += rew
        if done:
            print(self.g_step, ',', int(self.rewards), end='|', file=self.f_rewards, flush=True)
            self.rewards = 0
        # info = {'last_score':int(self.reward),'last_length':int(self.length),**info}
        return obs, rew, done, info


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
        if len(screen_size) == 2:
            height, width = int(screen_size[0] * args.zoom_in * screen_nums[0]), int(
                screen_size[1] * args.zoom_in * screen_nums[1])
        else:
            height, width = args.height, args.width
        if args.env_type == 'atari' and prefix == 'org_':
            height, width = 210, 160
        self.vWriter = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
        self.debug = True
        self.debug_one_screen_size, self.debug_height, self.debug_width = screen_size, height, width

    def __del__(self):
        self.vWriter.release()

    def render(self, mode):
        frame = self.env.render(mode)  # mode='rgb_array'
        if self.debug:
            print('debug_one_screen_size:', self.debug_one_screen_size)
            print('debugvideoheightwidth: ', self.debug_height, '', self.debug_width)
            print('debug     frame.shape:', frame.shape)
            self.debug = False
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.vWriter.write(frame)
        return frame

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, act):
        obs, rew, done, info = self.env.step(act)
        return obs, rew, done, info


class WrapDeepmindRender(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env=env)

    def render(self, mode):  # def render(self, *args, **kwargs):
        frame = self.env.render(mode)  # mode='rgb_array'
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)  # width,height
        frame = np.expand_dims(frame, -1)
        frame = np.tile(frame, 3)
        return frame

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, act):
        obs, rew, done, info = self.env.step(act)
        return obs, rew, done, info