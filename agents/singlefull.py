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
    agt = Memo(agt)
    agt = env_to_mem_obs_Stack(agt)
    return agt
