def add_arguments(parser):
    parser.add_argument('--agt-mode', default='imagine', help='agent mode to use: imagine | reconimg (imagine)')#agent
    parser.add_argument('--agt-agtparas', default='', help='agent parameters: number of players,number of types,number of units')
    parser.add_argument('--agt-agtonoff', default='', help='agent switches')
    parser.add_argument('--agt-learnrun', default='', help='learn flag switches: 0,-1,-1,-1')
    #parser.add_argument('--drawinfo', default='0,0,0,0', help='draw layer info frequency')
    parser.add_argument('--stack-num', type=int, default=1, help='number of observation stacks (default: 1)')
    parser.add_argument('--memo-size', type=int, default=1, help='size of memory (default: 5)')
    parser.add_argument('--drawinfo', default='0,0,0,0', help='draw layer info frequency')
def add_strings(args):
    args.exp_dir=args.exp_dir+':'+args.agt_mode+'_'+str(args.agt_agtparas)+'_'+str(args.agt_agtonoff)+'_'+str(args.agt_learnrun) \
                                               +'_'+str(args.stack_num)+'_'+str(args.memo_size)
import importlib,easydict
def getAgent(args,env):
    mod = importlib.import_module('agents.'+args.agt_mode)
    return mod.fAgent(args,env)

class Agent(object):
    meta = {'render.modes': []}
    spec = None
    def __init__(self):
        self.attr, self.args, self.envs, self.algo = easydict.EasyDict(), None, None, None
    @property
    def unwrapped(self):
        return self
    def __str__(self):
        if self.spec is None: return '<{} instance>'.format(type(self).__name__)
        else:                 return '<{}<{}>>'.format(type(self).__name__, self.spec.id)
    def memoexps(self, new_obs, rew, done, info):
        raise NotImplementedError
    def getaction(self, obs, explore):
        raise NotImplementedError
    def update(self, crt_step, max_step, info_in):
        raise NotImplementedError
    def save(self,name):
        raise NotImplementedError
    def load(self):
        raise NotImplementedError
class Wrapper(Agent):
    agt = None
    def __init__(self, agt):
        self.agt = agt
        self.attr, self.args, self.envs, self.algo = self.agt.attr, self.agt.args, self.agt.envs, self.agt.algo
        self.meta = self.agt.meta
    @property
    def spec(self):
        return self.agt.spec
    @property
    def unwrapped(self):
        return self.agt.unwrapped
    @classmethod
    def class_name(cls):
        return cls.__name__
    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.agt)
    def __repr__(self):
        return str(self)
    def memoexps(self, new_obs, rew, done, info, **kwargs):
        return self.agt.memoexps(new_obs, rew, done, info, **kwargs)
    def getaction(self, obs, explore, **kwargs):
        return self.agt.getaction(obs, explore, **kwargs)
    def update(self, crt_step, max_step, info_in, **kwargs):
        return self.agt.update(crt_step, max_step, info_in, **kwargs)
    def save(self, name, **kwargs):
        return self.agt.save(name, **kwargs)
    def load(self, **kwargs):
        return self.agt.load(**kwargs)
