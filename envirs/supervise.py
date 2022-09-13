import numpy as np
import gym,random,json
def fEnv(args):
    with open('./myenv/envinfo.json', 'w') as fenvinfo:
        print(json.dumps(vars(args)),file=fenvinfo)
    env = gym.make(args.env_name)
    env.seed(args.env_seed)
    random.seed(args.env_seed)
    np.random.seed(args.env_seed)
    if hasattr(env,'attr'): env.spec._kwargs['attr']=env.attr
    return env
