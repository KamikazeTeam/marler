import argparse
import numpy as np
import random
import os
import json
import tqdm
import importlib


# @profile  # perf_counter used, not process_time,count real time,not cpu time
def train(args, env, agt):
    with open(args.exp_dir[:-1] + '_args', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    args.max_train_steps = int(args.max_steps // args.env_nums // args.roll_num)
    t, iterator = 0, tqdm.tqdm(range(args.max_train_steps))
    try:
        if args.to_load:
            agt.load()
        obs = env.reset()
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
    args.max_test_steps = int(args.test_steps // args.env_nums)
    iterator = tqdm.tqdm(range(args.max_test_steps))
    for _ in iterator:
        if args.render:
            env.render(mode='rgb_array')
        act, act_info = agt.getaction(obs, explore=False)
        new_obs, rew, done, info = env.step(act)
        agt.memoexps(new_obs, rew, done, info)
        obs = new_obs


def main():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--env-seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--env-mode', default='', help='env mode to use: full | part | skip (full)')
    parser.add_argument('--env-type', default='', help='atari or sc2 flag (default: )')
    parser.add_argument('--env-name', default='', help='environment name (default: BreakoutNoFrameskip-v4)')
    parser.add_argument('--env-nums', type=int, default=1, help='how many training CPU processes to use (default: 12)')
    parser.add_argument('--roll-num', type=int, default=1, help='number of forward steps in A2C (default: 1)')
    parser.add_argument('--max-stepsM', default='10', help='number of environment steps to train (default: 10M)')
    parser.add_argument('--agt-mode', default='', help='agent mode to use: imagine | recon (imagine)')  # agt
    parser.add_argument('--stack-num', type=int, default=1, help='number of observation stacks (default: 1)')
    parser.add_argument('--memo-size', type=int, default=1, help='size of memory (default: 5)')
    parser.add_argument('--alg-mode', default='PTa2c1', help='algo to use: TFa2c1 | PTa2c1 | PTppo')  # algo
    parser.add_argument('--lr-M', default='700', help='learning rate (default: 700e-6)')
    parser.add_argument('--decay', default='linear', help='decay to use: linear | exp | cos | cos-cos | cos-dec')
    parser.add_argument('--decay-paras', default='0.01,,', help='decay parameters')  # 0.01,55556,0.8
    parser.add_argument('--opt', default='RMSprop', help='optimizer to use: RMSprop | Adam | ()')
    parser.add_argument('--opt-eps', type=float, default=1e-5, help='optimizer epsilon (default: 1e-5)')
    parser.add_argument('--opt-alpha', type=float, default=0.99, help='optimizer alpha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--loss-value-weight', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument('--loss-entropy-weight', type=float, default=0.01, help='entropy coefficient (default: 0.01)')
    parser.add_argument('--grad-max-norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument('--approx-func', default='cnn2d', help='approximate function: cnn2d | cnn3d | res3d')
    parser.add_argument('--approx-func-paras', default='8,8,4,4,32,1^4,4,2,2,64,1^3,3,1,1,64,1=512=64',
                        help='approximate function parameters')  # approximate function
    parser.add_argument('--to-load', action='store_true', default=False, help='render flag')
    parser.add_argument('--test-steps', type=int, default=0, help='test steps (default: 0)')  # test
    parser.add_argument('--render', action='store_true', default=False, help='render flag')
    parser.add_argument('--zoom-in', type=int, default=1, help='zoom-in size for render (default: 1)')
    parser.add_argument('--fps', type=int, default=60, help='fps for render (default: 60)')
    parser.add_argument('--width', type=int, default=600, help='width for render (default: 600)')
    parser.add_argument('--height', type=int, default=400, help='height for render (default: 400)')
    args = parser.parse_args()
    args.exp_dir = 'results/' + args.env_name + '_' + args.env_mode \
                   + '_' + str(args.env_nums) + '_' + str(args.roll_num) \
                   + ':' + args.agt_mode + '_' + str(args.stack_num) + '_' + str(args.memo_size) \
                   + ':' + args.alg_mode + '_' + str(args.lr_M) + '_' + args.decay + '_' + args.decay_paras \
                   + '_' + args.opt \
                   + ':' + args.approx_func + '_' + args.approx_func_paras \
                   + '/'
    print('exp_dir length: ', len(args.exp_dir))  # check whether folder name over length limits
    os.makedirs(args.exp_dir, exist_ok=True)
    args.max_steps = int(float(args.max_stepsM) * 1e6)
    args.lr = float(args.lr_M) * 1e-6
    if args.test_steps and args.render:
        args.env_nums = 1
    random.seed(args.env_seed)
    np.random.seed(args.env_seed)
    env = importlib.import_module('environment.' + args.env_mode).get_environment(args)
    agt = importlib.import_module('agent.' + args.agt_mode).get_agent(args, env)
    if args.test_steps:
        test(args, env, agt)
    else:
        train(args, env, agt)
    env.close()


if __name__ == '__main__':
    main()
