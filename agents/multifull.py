import numpy as np
import agents
from agents.wrappers import Memo,env_to_mem_obs_Stack
from algos import getAlgo
class MyAgent(agents.Agent):
    def __init__(self,args,env):
        agents.Agent.__init__(self)
        self.args = args
        self.attr.teamagts= [int(num) for num in args.teamagts.split(',')]
        self.envs = env
        self.attr.initobs = env.reset()
        print(args.env_name,env.observation_space.shape,env.action_space)
        #print(env.observation_space)
        self.algo = getAlgo(env.observation_space,env.action_space,args)
        #self.attr.teamagts  = [num*self.args.env_num for num in self.attr.teamagts]
        print('self.attr.teamagts ',self.attr.teamagts)
        self.attr.teamslice = [0]+[np.sum(self.attr.teamagts[0:i+1]) for i in range(len(self.attr.teamagts))]
        print('self.attr.teamslice',self.attr.teamslice)
        self.attr.teamnum   = 0
        print('self.attr.teamnum  ',self.attr.teamnum)
        self.attr.teamstart = self.attr.teamslice[self.attr.teamnum]
        print('self.attr.teamstart',self.attr.teamstart)
        self.attr.teamend   = self.attr.teamslice[self.attr.teamnum+1]
        print('self.attr.teamend  ',self.attr.teamend)
    def memoexps(self, new_obs, rew, done, info):
        self.algo.memoexps(new_obs, rew, done, info)
    def getaction(self, obs, explore):
        act, act_info = self.algo.get_action(obs,explore)
        return act, act_info
    def update(self, crt_step, max_step, info_in):
        self.algo.update(crt_step=crt_step, max_step=max_step, info_in=info_in)
    def save(self,name):
        self.algo.save(name)
    def load(self):
        self.algo.load()

def fAgent(args,env):
    agt = MyAgent(args,env)
    agt = agt_to_mem_act_Shape(agt)
    agt = mem_to_agt_update_Shape(agt)
    agt = mem_to_agt_obs_Shape(agt)
    agt = Memo(agt)
    agt = mem_to_env_act_Shape(agt)
    agt = env_to_mem_update_Shape(agt)
    agt = env_to_mem_obs_Shape(agt)
    agt = env_to_mem_obs_Stack(agt)
    return agt

class agt_to_mem_act_Shape(agents.Wrapper):
    def __init__(self,agt):
        agents.Wrapper.__init__(self,agt)
    def combine_value_teams(self,orgvalues,newvalues,teamnum):
        values = []
        try:
            for j in range(len(self.attr.teamagts)):
                if j==teamnum:
                    for k in range(self.attr.teamagts[j]):
                        for i in range(self.args.env_num):
                            values.append(newvalues[k*self.args.env_num+i])
                else:
                    for k in range(self.attr.teamagts[j]):
                        for i in range(self.args.env_num):
                            values.append(orgvalues[(self.attr.teamslice[j]+k)*self.args.env_num+i])
        except:
            print(j,k,i)
            print(self.args.env_num,self.attr.teamslice)
            print(self.attr.teamslice[j]+k)
            print(k)
            print(len(orgvalues),len(newvalues))
            exit()
        values = np.array(values)
        return values
    def shapeact(self,act,act_info): # only support discrete action currently...
        act_rnd = np.random.randint(0,self.envs.action_space.n,self.attr.teamagts[self.attr.teamnum]*self.args.env_num)
        act_all = self.combine_value_teams(act,act_rnd,1) # let rnd act to team 1
        act = act_all.reshape(-1,self.args.env_num)
        act = act.transpose(1,0)
        return act,act_info
    def getaction(self, obs, explore):
        act, act_info = self.agt.getaction(obs,explore)
        act, act_info = self.shapeact(act, act_info)
        return act, act_info
class mem_to_env_act_Shape(agents.Wrapper):
    def __init__(self,agt):
        agents.Wrapper.__init__(self,agt)
    def shapeact(self,act,act_info):
        return act,act_info
    def getaction(self, obs, explore):
        act, act_info = self.agt.getaction(obs,explore)
        act, act_info = self.shapeact(act, act_info)
        return act, act_info
class env_to_mem_update_Shape(agents.Wrapper):
    def __init__(self,agt):
        agents.Wrapper.__init__(self,agt)
    def update(self, crt_step, max_step, info_in):
        self.agt.update(crt_step=crt_step, max_step=max_step, info_in=info_in)
class mem_to_agt_update_Shape(agents.Wrapper):
    def __init__(self,agt):
        agents.Wrapper.__init__(self,agt)
        self.debug = False
    def update(self, crt_step, max_step, info_in):
        if self.debug:
            for key,value in info_in.items():
                print(key,value.shape)
        info_in['mb_obs'] = info_in['mb_obs'][:,:,:,self.attr.teamstart:self.attr.teamend,:,:,:]
        info_in['mb_act'] = info_in['mb_act'][:,:,self.attr.teamstart:self.attr.teamend]
        info_in['mb_new_obs'] = info_in['mb_new_obs'][:,:,:,self.attr.teamstart:self.attr.teamend,:,:,:]
        info_in['mb_rew'] = info_in['mb_rew'][:,:,self.attr.teamstart:self.attr.teamend]
        if self.debug:
            for key,value in info_in.items():
                print(key,value.shape)
        info_in['mb_obs']     = info_in['mb_obs'].transpose(0,1,3,2,4,5,6) # org is roll batch stack numOfAgts width height channal
        info_in['mb_obs']     = info_in['mb_obs'].reshape(info_in['mb_obs'].shape[0],-1,*info_in['mb_obs'].shape[3:])
        info_in['mb_act']  = info_in['mb_act'].reshape(info_in['mb_act'].shape[0],-1) # org is roll batch numOfAgts
        info_in['mb_new_obs'] = info_in['mb_new_obs'].transpose(0,1,3,2,4,5,6)
        info_in['mb_new_obs'] = info_in['mb_new_obs'].reshape(info_in['mb_new_obs'].shape[0],-1,*info_in['mb_new_obs'].shape[3:])
        info_in['mb_done'] = np.expand_dims(info_in['mb_done'],-1) # org is roll batch
        info_in['mb_done'] = np.tile(info_in['mb_done'],info_in['mb_rew'].shape[-1])
        info_in['mb_rew']  = info_in['mb_rew'].reshape(info_in['mb_rew'].shape[0],-1) # must reshape after done, because done used its shape...
        info_in['mb_done'] = info_in['mb_done'].reshape(info_in['mb_done'].shape[0],-1)
        info_in['mb_info'] = info_in['mb_info'].reshape(info_in['mb_info'].shape[0],-1) # info and act info's shape are wrong, to do...
        if self.debug:
            for key,value in info_in.items():
                print(key,value.shape)
            exit()
        self.agt.update(crt_step=crt_step, max_step=max_step, info_in=info_in)
class env_to_mem_obs_Shape(agents.Wrapper):
    def __init__(self,agt):
        agents.Wrapper.__init__(self,agt)
    def shapeobs(self,obs):
        obs = obs*255.0/2.0 # to do
        if len(obs.shape)==4: # batch stack agt envobs
            obs = np.expand_dims(obs,-1)
            obs = np.expand_dims(obs,-1)
        return obs
    def memoexps(self, new_obs, rew, done, info):
        new_obs = self.shapeobs(new_obs)
        self.agt.memoexps(new_obs, rew, done, info)
    def getaction(self, obs, explore):
        obs = self.shapeobs(obs)
        act, act_info = self.agt.getaction(obs,explore)
        return act, act_info
class mem_to_agt_obs_Shape(agents.Wrapper):
    def __init__(self,agt):
        agents.Wrapper.__init__(self,agt)
    def shapeobs(self,obs):
        obs = obs.transpose(2,0,1,3,4,5) # go to agts batch stack width height channel
        obs = obs.reshape(-1,*obs.shape[2:])
        return obs
    def memoexps(self, new_obs, rew, done, info):
        new_obs = self.shapeobs(new_obs)
        self.agt.memoexps(new_obs, rew, done, info)
    def getaction(self, obs, explore):
        obs = self.shapeobs(obs)
        act, act_info = self.agt.getaction(obs,explore)
        return act, act_info

