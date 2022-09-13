import numpy as np
import gym, easydict, random

def env_setting(args):
    env_kwargs = easydict.EasyDict()
    env_kwargs.local_ratio = 0.0#5
    env_kwargs.N = int(args.teamagts.split(',')[0])
    # env_kwargs.continuous_actions = True#False
    env_kwargs.max_cycles = 30
    env_config = easydict.EasyDict()
    env_config.screen_size=(700,700)
    env_config.showstateN = 0
    env_config.paddingobs = False
    env_config.paddingact = False
    env_config.num_frames = 0
    return env_kwargs, env_config
class Recorder_before(gym.Wrapper):
    def __init__(self, env, args):
        gym.Wrapper.__init__(self, env=env)
        self.N = args.N
        self.offset = 4
    def obs_modify(self,obs):
        return obs
        tmp = obs.copy()
        for i,obsi in enumerate(obs):
            obs[i][-2:]=i
        return obs
    def rew_modify(self,rew,obs):
        return rew
        for i,obsi in enumerate(obs):
            dist  = np.sqrt(obsi[self.offset+2*i]*obsi[self.offset+2*i]+obsi[self.offset+2*i+1]*obsi[self.offset+2*i+1])
            rew[i]=-dist
        return rew
    def info_modify(self,obs):
        info_render = {}
        for i,obsi in enumerate(obs):
            info_render['obs'+str(i)] = '|'.join([str(format(ob,'+.2f')) for ob in obsi])
        for i,obsi in enumerate(obs):
            disttojs = []
            distring = ''
            for j in range(self.N):
                disttoj = np.sqrt(obsi[self.offset+j]*obsi[self.offset+j]+obsi[self.offset+j+1]*obsi[self.offset+j+1])
                disttojs.append(disttoj)
                distring+= '|'+str(format(disttoj,'+.4f'))
            distsum = np.sum(disttojs)
            distring= str(format(distsum,'+.4f'))+distring
            info_render['dist'+str(i)] = distring
        return info_render
    def reset(self):
        obs = self.env.reset()
        obs = self.obs_modify(obs)
        return obs
    def step(self, act):
        obs, rew, done, info = self.env.step(act)
        obs = self.obs_modify(obs)
        rew = self.rew_modify(rew,obs)
        info[0]['render']= self.info_modify(obs)
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
# class Recorder_before(gym.Wrapper):
#     def __init__(self, env):
#         gym.Wrapper.__init__(self, env=env)
#     # def obs_modify2(self,obs):
#     #     tmp = obs.copy()
#     #     for i,obsi in enumerate(obs):
#     #         if i==0:
#     #             obs[i][-4:]= 0
#     #         else:
#     #             obs[i][-4:]= 1
#     #             obs[i][4:6]=tmp[i][6:8]
#     #             obs[i][6:8]=tmp[i][4:6]
#     #     return obs
#     def obs_modify(self,obs):
#         return obs
#         tmp = obs.copy()
#         for i,obsi in enumerate(obs):
#             if i==0:
#                 obs[i][-4:]= 0
#                 # obs[i][4:6]= np.sqrt(tmp[i][4]*tmp[i][4]+tmp[i][5]*tmp[i][5])
#                 #obs[i][6:8]= 0
#             else:
#                 obs[i][-4:]= 1
#                 #obs[i][4:6]= 0
#                 # obs[i][6:8]= np.sqrt(tmp[i][6]*tmp[i][6]+tmp[i][7]*tmp[i][7])
#         return obs
#     # def rew_modify2(self,rew,obs):
#     #     for i,obsi in enumerate(obs):
#     #         dist1 = np.sqrt(obsi[4]*obsi[4]+obsi[5]*obsi[5])
#     #         rew[i]=-dist1
#     #     return rew
#     def rew_modify(self,rew,obs):
#         return rew
#         for i,obsi in enumerate(obs):
#             dist1 = np.sqrt(obsi[4]*obsi[4]+obsi[5]*obsi[5])
#             dist2 = np.sqrt(obsi[6]*obsi[6]+obsi[7]*obsi[7]) # dist  = dist1+dist2
#             if i==0: rew[i]= -dist1
#             else:    rew[i]= -dist2
#         return rew
#     def info_modify(self,obs):
#         info_render = {}
#         for i,obsi in enumerate(obs):
#             info_render['obs'+str(i)] = '|'.join([str(np.round(ob,2)) for ob in obsi])
#         for i,obsi in enumerate(obs):
#             distto1 = np.sqrt(obsi[4]*obsi[4]+obsi[5]*obsi[5])
#             distto2 = np.sqrt(obsi[6]*obsi[6]+obsi[7]*obsi[7])
#             distsum = distto1+distto2
#             info_render['dist'+str(i)]= str(np.round(distsum,8))+'|'+str(np.round(distto1,8))+'|'+str(np.round(distto2,8))
#         return info_render
#     def reset(self):
#         obs = self.env.reset()
#         obs = self.obs_modify(obs)
#         return obs
#     def step(self, act):
#         obs, rew, done, info = self.env.step(act)
#         obs = self.obs_modify(obs)
#         rew = self.rew_modify(rew,obs)
#         info[0]['render']= self.info_modify(obs)
#         return obs, rew, done, info


