import numpy as np
from collections import deque
import easydict


# obs new_obs rew done info need to place after Memo, act act_info and update need to place before Memo
class Storage:
    def __init__(self, memo_size):
        self.memo = easydict.EasyDict()
        self.memo.obs = deque(maxlen=memo_size)
        self.memo.act = deque(maxlen=memo_size)
        self.memo.act_info = deque(maxlen=memo_size)
        self.memo.new_obs = deque(maxlen=memo_size)
        self.memo.rew = deque(maxlen=memo_size)
        self.memo.done = deque(maxlen=memo_size)
        self.memo.info = deque(maxlen=memo_size)

    def append_data(self, obs, act, act_info, new_obs, rew, done, info):
        self.memo.obs.append(obs)
        self.memo.act.append(act)
        self.memo.act_info.append(act_info)
        self.memo.new_obs.append(new_obs)
        self.memo.rew.append(rew)
        self.memo.done.append(done)
        self.memo.info.append(info)

    def get_data(self):
        data = {'mb_obs': np.array(self.memo['obs']),
                'mb_act': np.array(self.memo['act']),
                'mb_act_info': np.array(self.memo['act_info']),
                'mb_new_obs': np.array(self.memo['new_obs']),
                'mb_rew': np.array(self.memo['rew']),
                'mb_done': np.array(self.memo['done']),
                'mb_info': np.array(self.memo['info'])}
        return data


def get_storage(memo_size):
    storage = Storage(memo_size)
    return storage
