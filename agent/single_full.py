import agent
from agent.wrappers import Memo, EnvToMemObsStack
import importlib


class MyAgent(agent.Agent):
    def __init__(self, args, env):
        agent.Agent.__init__(self)
        self.args = args
        self.envs = env
        self.attr.init_obs = env.reset()
        print(args.env_name, env.observation_space.shape, env.action_space)
        # print(env.observation_space)
        self.algo = importlib.import_module('algos.' + args.alg_mode).get_algo(
            env.observation_space, env.action_space, args)

    def memoexps(self, new_obs, rew, done, info):
        self.algo.memoexps(new_obs, rew, done, info)

    def getaction(self, obs, explore):
        act, act_info = self.algo.get_action(obs, explore)
        return act, act_info

    def update(self, crt_step, max_step, info_in):
        self.algo.update(crt_step=crt_step, max_step=max_step, info_in=info_in)

    def save(self, name):
        self.algo.save(name)

    def load(self):
        self.algo.load()


def get_agent(args, env):
    agt = MyAgent(args, env)
    agt = Memo(agt)
    agt = EnvToMemObsStack(agt)
    return agt
