import argparse
import numpy as np
import os
import random
import json
import tqdm
import envirs
import agents
import algos


@profile  # perf_counter used, not process_time,count real time,not cpu time
def train(args, env, agt):
    with open(args.exp_dir[:-1] + '_args', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    try:
        if args.to_load: agt.load()
        obs = env.reset()
        args.max_train_steps = int(args.max_steps // args.env_num // args.roll_num)
        iterator = tqdm.tqdm(range(args.max_train_steps))
        for t in iterator:
            for n in range(args.roll_num):
                act, act_info = agt.getaction(obs, explore=True)
                new_obs, rew, done, info = env.step(act)  # must create a new_obs each step
                agt.memoexps(new_obs, rew, done, info)  # must not to change new_obs
                obs = new_obs
            agt.update(t, args.max_train_steps, info_in={})
        agt.save(str(args.env_seed) + '_' + str(t))
    except KeyboardInterrupt:
        agt.save(str(args.env_seed) + '_' + str(t))  # if pass, files will not have enough time to close...


def test(args, env, agt):
    agt.load()
    obs = env.reset()
    args.max_test_steps = int(args.test_steps // args.env_num)
    iterator = tqdm.tqdm(range(args.max_test_steps))
    for t in iterator:
        if args.render: env.render(mode='rgb_array')
        act, act_info = agt.getaction(obs, explore=False)
        new_obs, rew, done, info = env.step(act)
        agt.memoexps(new_obs, rew, done, info)
        obs = new_obs


def mainloop(args):
    random.seed(args.env_seed)
    np.random.seed(args.env_seed)
    args.exp_dir = 'results/' + args.env_name + '_' + str(args.env_num) + '_' + str(args.roll_num)
    envirs.add_strings(args)
    agents.add_strings(args)
    algos.add_strings(args)
    args.exp_dir = args.exp_dir + '/'
    print('exp_dir length: ', len(args.exp_dir))  # check whether folder name over length limits
    os.makedirs(args.exp_dir, exist_ok=True)
    args.max_steps = int(float(args.max_stepsM) * 1e6)
    args.lr = float(args.lr_M) * 1e-6
    if args.test_steps and args.render: args.env_num = 1
    env = envirs.getEnvir(args)
    agt = agents.getAgent(args, env)
    print(args.env_seed, ':', args.fin_seed)
    if args.test_steps:
        test(args, env, agt)
    else:
        train(args, env, agt)
    env.close()


def main():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--env-seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--env-type', default='', help='atari or sc2 flag (default: )')
    parser.add_argument('--env-name', default='', help='environment name (default: BreakoutNoFrameskip-v4)')
    parser.add_argument('--env-nums', type=int, default=1, help='how many training CPU processes to use (default: 12)')
    parser.add_argument('--roll-num', type=int, default=1, help='number of forward steps in A2C (default: 1)')
    parser.add_argument('--max-stepsM', default='10', help='number of environment steps to train (default: 10M)')
    parser.add_argument('--to-load', action='store_true', default=False, help='load previous agent flag')
    envirs.add_arguments(parser)
    agents.add_arguments(parser)
    algos.add_arguments(parser)
    parser.add_argument('--test-steps', type=int, default=0, help='test steps (default: 0)')
    parser.add_argument('--render', action='store_true', default=False, help='render flag')
    parser.add_argument('--zoom-in', type=int, default=1, help='zoom-in size for render (default: 1)')
    parser.add_argument('--fps', type=int, default=60, help='fps for render (default: 60)')
    parser.add_argument('--width', type=int, default=600, help='width for render (default: 600)')
    parser.add_argument('--height', type=int, default=400, help='height for render (default: 400)')
    args = parser.parse_args()
    mainloop(args)


if __name__ == '__main__':
    main()
