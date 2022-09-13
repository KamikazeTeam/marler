import numpy as np
import agents
from collections import deque
# obs new_obs rew done info need to place after Memo, act act_info and update need to place before Memo
class Memo(agents.Wrapper):
    def __init__(self,agt):
        agents.Wrapper.__init__(self,agt)
        memo_size = agt.args.memo_size
        agt.attr.memo_obs       = deque(maxlen=memo_size)
        agt.attr.memo_act       = deque(maxlen=memo_size)
        agt.attr.memo_new_obs   = deque(maxlen=memo_size)
        agt.attr.memo_rew       = deque(maxlen=memo_size)
        agt.attr.memo_done      = deque(maxlen=memo_size)
        agt.attr.memo_info      = deque(maxlen=memo_size)
        agt.attr.memo_act_info  = deque(maxlen=memo_size)
    def memoexps(self, new_obs, rew, done, info):
        self.agt.attr.memo_new_obs.append(new_obs)
        self.agt.attr.memo_rew.append(rew)
        self.agt.attr.memo_done.append(done)
        self.agt.attr.memo_info.append(info)
        self.agt.memoexps(new_obs, rew, done, info)
    def getaction(self, obs, explore):
        self.agt.attr.memo_obs.append(obs)
        act, act_info = self.agt.getaction(obs,explore)
        self.agt.attr.memo_act.append(act)
        self.agt.attr.memo_act_info.append(act_info)
        return act, act_info
    def update(self, crt_step, max_step, info_in={}):
        info_in =  {'mb_obs':       np.array(self.agt.attr.memo_obs),
                    'mb_act':       np.array(self.agt.attr.memo_act),
                    'mb_new_obs':   np.array(self.agt.attr.memo_new_obs),
                    'mb_rew':       np.array(self.agt.attr.memo_rew),
                    'mb_done':      np.array(self.agt.attr.memo_done),
                    'mb_info':      np.array(self.agt.attr.memo_info),
                    'mb_act_info':  np.array(self.agt.attr.memo_act_info),
                    **info_in}
        self.agt.update(crt_step=crt_step, max_step=max_step, info_in=info_in)
class env_to_mem_obs_Stack(agents.Wrapper):
    def __init__(self,agt):
        agents.Wrapper.__init__(self,agt)
        #self.attr.obs_stack = np.zeros([agt.attr.args.env_num]+[agt.attr.args.stack_num]+list(agt.attr.stack_shape), dtype=agt.attr.stack_dtype)
        self.attr.obs_stack = np.zeros([agt.attr.initobs.shape[0]]+[agt.args.stack_num]+list(agt.attr.initobs.shape[1:]), dtype=agt.attr.initobs.dtype)
    def obs_stack_update(self, new_obs, old_obs_stack):
        updated_obs_stack = np.roll(old_obs_stack, shift=-1, axis=1)
        updated_obs_stack[:,-1,:] = new_obs#[:]###
        return updated_obs_stack
    def memoexps(self, new_obs, rew, done, info):
        new_stack = self.obs_stack_update(new_obs, self.attr.obs_stack)
        for i,donei in enumerate(done):
            if donei: self.attr.obs_stack[i]*=0#[:-1]*=0
        self.agt.memoexps(new_stack, rew, done, info)
    def getaction(self, obs, explore):
        self.attr.obs_stack = self.obs_stack_update(obs, self.attr.obs_stack)
        obs_stack_copy = np.copy(self.attr.obs_stack)###must copy!
        act, act_info = self.agt.getaction(obs_stack_copy,explore)
        return act, act_info
