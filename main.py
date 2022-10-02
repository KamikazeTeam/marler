import argparse
import numpy as np
import random
import os
import tqdm
import importlib
import torch
import wandb


# @profile  # perf_counter used, not process_time,count real time,not cpu time
def train(args, env, mdl, stg, alg):
    if args.wandb:
        wandb.init(project=args.env_name)
        wandb.config.update(args)
    args.max_train_steps = int(args.max_steps // args.env_nums // args.roll_num)
    t, iterator = 0, tqdm.tqdm(range(args.max_train_steps))
    try:
        if args.to_load:
            mdl.load()
        obs, info = env.reset()
        for t in iterator:
            for n in range(args.roll_num):
                act, act_info = mdl.get_action(obs, explore=True)
                new_obs, rew, done, timeout, info = env.step(act)  # must create a new_obs each step
                stg.append_data(obs, act, act_info, new_obs, rew, done, info)
                obs = np.copy(new_obs)  # must copy!
                for i, info_i in enumerate(info):
                    if 'score' in info_i:
                        if args.wandb:
                            wandb.log({"g_step": info_i['g_step'],  # it is possible to have multi-score in same g_step
                                       # "score/"+str(i): info_i['score'],
                                       "scores": info_i['score']})
            data = stg.get_data()
            value_loss, action_loss, dist_entropy = alg.update(t, args.max_train_steps, data, mdl)
            if args.wandb:
                wandb.log({"update_step": (t+1)*args.roll_num*args.env_nums,  # g_step not only increase, need fix
                           "value_loss": value_loss, "action_loss": action_loss, "dist_entropy": dist_entropy})
            # wandb.watch(mdl)
        mdl.save(str(args.env_seed) + '_' + str(t))
    except KeyboardInterrupt:
        mdl.save(str(args.env_seed) + '_' + str(t))  # if pass, files will not have enough time to close...


def test(args, env, mdl):
    mdl.load()
    obs, info = env.reset()
    args.max_test_steps = int(args.test_steps // args.env_nums)
    iterator = tqdm.tqdm(range(args.max_test_steps))
    for _ in iterator:
        if args.render:
            env.render()
        act, act_info = mdl.get_action(obs, explore=False)
        new_obs, rew, done, timeout, info = env.step(act)
        obs = np.copy(new_obs)


def main():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--env-seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--env-mode', default='', help='env mode to use: full | part | skip (full)')
    parser.add_argument('--env-type', default='', help='atari or sc2 flag (default: )')
    parser.add_argument('--env-name', default='', help='environment name (default: BreakoutNoFrameskip-v4)')
    parser.add_argument('--env-nums', type=int, default=1, help='how many training CPU processes to use (default: 12)')
    parser.add_argument('--stack-num', type=int, default=1, help='number of observation stacks (default: 1)')
    parser.add_argument('--roll-num', type=int, default=1, help='number of forward steps in A2C (default: 1)')
    parser.add_argument('--max-stepsM', default='10', help='number of environment steps to train (default: 10M)')
    parser.add_argument('--stg-mode', default='', help='storage to use: deque | ')  # storage
    parser.add_argument('--memo-size', type=int, default=1, help='size of memory (default: 5)')
    parser.add_argument('--alg-mode', default='PTa2c1', help='algo to use: TFa2c1 | PTa2c1 | PTppo')  # algorithm
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--loss-value-weight', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument('--loss-entropy-weight', type=float, default=0.01, help='entropy coefficient (default: 0.01)')
    parser.add_argument('--mdl-mode', default='neural', help='model to use: neural | ')  # model
    parser.add_argument('--lr-M', default='700', help='learning rate (default: 700e-6)')
    parser.add_argument('--decay', default='linear', help='decay to use: linear | exp | cos | cos-cos | cos-dec')
    parser.add_argument('--decay-paras', default='0.01,,', help='decay parameters')  # 0.01,55556,0.8
    parser.add_argument('--opt', default='RMSprop', help='optimizer to use: RMSprop | Adam | ()')
    parser.add_argument('--opt-eps', type=float, default=1e-5, help='optimizer epsilon (default: 1e-5)')
    parser.add_argument('--opt-alpha', type=float, default=0.99, help='optimizer alpha (default: 0.99)')
    parser.add_argument('--grad-max-norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument('--approx-func', default='cnn2d', help='approximate function: cnn2d | cnn3d | res3d')
    parser.add_argument('--approx-func-paras', default='8,8,4,4,32,1^4,4,2,2,64,1^3,3,1,1,64,1=512=64',
                        help='approximate function parameters')  # approximate function
    parser.add_argument('--to-load', action='store_true', default=False, help='render flag')
    parser.add_argument('--test-steps', type=int, default=0, help='test steps (default: 0)')  # test
    parser.add_argument('--render', action='store_true', default=False, help='render flag')
    parser.add_argument('--zoom-in', type=int, default=1, help='zoom-in size for render (default: 1)')
    parser.add_argument('--fps', type=int, default=60, help='fps for render (default: 60)')
    parser.add_argument('--width', type=int, default=-1, help='width for render (default: 600)')
    parser.add_argument('--height', type=int, default=-1, help='height for render (default: 400)')
    parser.add_argument('--wandb', action='store_false', default=True, help='wandb flag')

    args = parser.parse_args()
    args.exp_dir = 'results/' + args.env_name + '_' + args.env_mode \
                   + '_' + str(args.env_nums) + '_' + str(args.stack_num) + '_' + str(args.roll_num) \
                   + ':' + args.stg_mode + '_' + str(args.memo_size) \
                   + ':' + args.mdl_mode + '_' + str(args.lr_M) + '_' + args.decay + '_' + args.decay_paras \
                   + '_' + args.opt \
                   + ':' + args.alg_mode \
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
    torch.manual_seed(args.env_seed)
    torch.cuda.manual_seed_all(args.env_seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    env = importlib.import_module('environment.' + args.env_mode).get_environment(args)
    stg = importlib.import_module('storage.' + args.stg_mode).get_storage(args.memo_size)
    mdl = importlib.import_module('model.' + args.mdl_mode).get_model(args, env.observation_space, env.action_space)
    alg = importlib.import_module('algorithm.' + args.alg_mode).get_algorithm(args)
    if args.test_steps:
        test(args, env, mdl)
    else:
        train(args, env, mdl, stg, alg)
    env.close()


if __name__ == '__main__':
    main()
