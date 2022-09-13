import numpy as np
import gym, easydict, random

def env_setting(args):
    env_kwargs = easydict.EasyDict()
    # env_kwargs.num_food    = 2
    # env_kwargs.num_forests = 2
    # env_kwargs.num_good       = 1
    # env_kwargs.num_adversaries= 3
    # env_kwargs.num_obstacles  = 2
    env_kwargs.local_ratio = 0.0#5
    env_kwargs.N = 2
    # env_kwargs.continuous_actions       = True#False
    env_kwargs.max_cycles               = 30#max_cycles
    # env_kwargs.map_size                 = map_size
    # env_kwargs.minimap_mode             = False#True
    # env_kwargs.extra_features           = False
    # env_kwargs.step_reward              = 0#step_reward#-0.02
    # env_kwargs.dead_penalty             = 0#-0.9#dead_penalty
    # env_kwargs.attack_penalty           = 0#-0.1#attack_penalty
    # env_kwargs.attack_opponent_reward   = 0#1.2#attack_opponent_reward#1
    # env_kwargs.kill_reward              = 9.2#kill_reward
    env_config = easydict.EasyDict()
    env_kwargs.map_size,env_config.screen_size = 7,(56,71)
    #7,56,71#17,136,151#12,96,111;16,128,143;17,136,151;20,160,175;30,240,255;45,750,750;
    env_config.showstateN, env_config.paddingobs, env_config.paddingact, env_config.num_frames = 0, False, False, 1
    return env_kwargs, env_config
class Recorder_before(gym.Wrapper): # for magent battle env
    def __init__(self, env, args):
        gym.Wrapper.__init__(self, env=env)
        self.kill_reward = args.kill_reward
        self.dead_penalty= args.dead_penalty
    def reset(self):
        obs = self.env.reset()
        obs = self.env.reset() # additional reset to add reset times to 3 to avoid generate same map every two times...
        return obs
    def step(self, act):
        obs, rew, done, info = self.env.step(act)
        rew = [reward if reward<=self.kill_reward and reward>=self.dead_penalty else 0.0 for reward in rew] # cut bug reward!!!
        return obs, rew, done, info
class Recorder_after(gym.Wrapper):
    def __init__(self, env, args):
        gym.Wrapper.__init__(self, env=env)
    def reset(self):
        obs = self.env.reset()
        return obs
    def step(self, act):
        obs, rew, done, info = self.env.step(act)
        return obs, rew, done[0], info[0]



# def env_setting(args):
#     env_kwargs = easydict.EasyDict()
#     # env_kwargs.num_food    = 2
#     # env_kwargs.num_forests = 2
#     # env_kwargs.num_good       = 1
#     # env_kwargs.num_adversaries= 3
#     # env_kwargs.num_obstacles  = 2
#     env_kwargs.local_ratio = 0.0#5
#     env_kwargs.N = 2
#     # env_kwargs.continuous_actions       = True#False
#     env_kwargs.max_cycles               = 30#max_cycles
#     # env_kwargs.map_size                 = map_size
#     # env_kwargs.minimap_mode             = False#True
#     # env_kwargs.extra_features           = False
#     # env_kwargs.step_reward              = 0#step_reward#-0.02
#     # env_kwargs.dead_penalty             = 0#-0.9#dead_penalty
#     # env_kwargs.attack_penalty           = 0#-0.1#attack_penalty
#     # env_kwargs.attack_opponent_reward   = 0#1.2#attack_opponent_reward#1
#     # env_kwargs.kill_reward              = 9.2#kill_reward
#     env_config = easydict.EasyDict()
#     env_kwargs.map_size,env_config.screen_size = 7,(56,71)
#     #7,56,71#17,136,151#12,96,111;16,128,143;17,136,151;20,160,175;30,240,255;45,750,750;
#     env_config.showstateN, env_config.paddingobs, env_config.paddingact, env_config.num_frames = 0, False, False, 1
#     return env_kwargs, env_config
# class Recorder_before(gym.Wrapper): # for magent battle env
#     def __init__(self, env):
#         gym.Wrapper.__init__(self, env=env)
#     def reset(self):
#         obs = self.env.reset()
#         obs = self.env.reset() # additional reset to add reset times to 3 to avoid generate same map every two times...
#         return obs
#     def step(self, act):
#         obs, rew, done, info = self.env.step(act)
#         rew = [reward if reward<=self.env.env_kwargs.kill_reward and reward>=self.env.env_kwargs.dead_penalty else 0.0 for reward in rew] # cut bug reward!!!
#         return obs, rew, done, info
# class Recorder_after(gym.Wrapper):
#     def __init__(self, env):
#         gym.Wrapper.__init__(self, env=env)
#     def reset(self):
#         obs = self.env.reset()
#         return obs
#     def step(self, act):
#         obs, rew, done, info = self.env.step(act)
#         return obs, rew, done[0], info[0]


