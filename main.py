import optuna,joblib
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import random,os,copy,time
import argparse,json,pprint,tqdm
import envirs,agents,algos
def optunascoring():
    return 'exinfos'#'last_score'
def optunasetting(args,trial):
    list_envparas = args.envparas.split('_')
    #args.envparas = list_envparas[0]+'_'+args.g_end+'_'+args.g_rew+'_'+list_envparas[3]+'_'+list_envparas[4]+'_'+list_envparas[5]
    #args.optuna_nloops   = 2
    #args.optuna_tlimit   = 600

    #5,5,3_13,987,200_1,-1,-10,100_0_28,1,1_3,4,10,1,1
    args.g_size = list_envparas[0]#'9,9,3,4'#'5,5,3,5'
    #g_end = trial.suggest_int('g_end',100,300)
    args.g_end  = list_envparas[1]#'10,1500'#+str(g_end*10)
    #print(args.g_end)

    lr_M  = trial.suggest_int('lr_M', 2, 100)
    args.lr_M   = lr_M*100
    #args.lr   = round(args.lr,2)
    print('optuna lr_M: ',args.lr_M)
    #args.lr   = trial.suggest_categorical('lr', [0.0004, 0.0007, 0.001, 0.0013, 0.0016])
    #return

    #g_rew.append(str(trial.suggest_categorical('rewrng',['0','1','-1'])))
    #g_rew.append(str(trial.suggest_categorical('rewrng',['0','1','-1'])))
    g_rews, rnum = [], 2
    g_rew = trial.suggest_int('timego', int(-1.00*pow(10,rnum)),  int(0.00*pow(10,rnum)))
    g_rews.append(str(round(g_rew/pow(10,rnum),rnum)))
    g_rew = trial.suggest_int('atkhit', int( 0.00*pow(10,rnum)),  int(1.00*pow(10,rnum)))
    g_rews.append(str(round(g_rew/pow(10,rnum),rnum)))
    g_rew2 = 0#trial.suggest_int('hitted', int( 0.00*pow(10,rnum)),  int(1.00*pow(10,rnum)))
    g_rew = -g_rew*g_rew2/pow(10,rnum)
    g_rews.append(str(round(g_rew/pow(10,rnum),rnum)))
    g_rew = trial.suggest_int('draw',   int(-1.00*pow(10,rnum)),  int(0.00*pow(10,rnum)))
    g_rews.append(str(round(g_rew/pow(10,rnum),rnum)))
    g_rew = trial.suggest_int('lost',   int(-5.00*pow(10,rnum)), int(-1.00*pow(10,rnum)))
    g_rews.append(str(round(g_rew/pow(10,rnum),rnum)))
    g_rew2 = int(50.00*pow(10,rnum))#trial.suggest_int('winn',   int(10.00*pow(10,rnum)), int(50.00*pow(10,rnum)))
    g_rew = -g_rew*g_rew2/pow(10,rnum)
    g_rews.append(str(round(g_rew/pow(10,rnum),rnum-1)))
    args.g_rew  = ','.join(g_rews)
    #args.g_rew  = '0,1,-1,0,-2,10'
    print(args.g_rew)
    #args.g_envp = '0'
    #args.g_npcp = '1,4,35,1,1'
    #args.g_agtp = '3,4,10,1,1'
    args.envparas = args.g_size+'_'+args.g_end+'_'+args.g_rew#+'_'+args.g_envp+'_'+args.g_npcp+'_'+args.g_agtp
    return

    #args.stack_num = trial.suggest_int('stack_num', 2, 4)
    args.res  = trial.suggest_categorical('res',['3,3,3,2,2,1,1,1,1,64,2^64,1,1^128,1,2',
                                                 '3,3,3,2,2,1,2,2,1,64,2^64,1,1^128,1,2'])
    args.cnn  = trial.suggest_categorical('cnn',['3,3,3,2,2,1,0,0,1,128,1^2,2,3,1,1,1,0,0,1,256,1',
                                                 '3,3,3,2,2,1,1,1,1,128,1^2,2,3,1,1,1,1,1,1,256,1',
                                                 '3,3,3,2,2,2,1,1,1,128,1^2,2,3,1,1,1,1,1,1,256,1'])
    args.mlp  = trial.suggest_categorical('mlp',['512','256','128'])
    #args.memoplace = trial.suggest_categorical('memoplace', ['agtcpu', 'algocpu', 'algogpu'])
    if args.aprxfunc == 'cnn3d': args.apfparas = args.cnn+'='+args.mlp
    if args.aprxfunc == 'res3d': args.apfparas = args.res+'='+args.mlp
    return
def optunaloop(args):
    try:
        lines = open('results/optunacurve','r').read().splitlines()
        elements = lines[0].split(',')
        scores = [float(element) for element in elements[:-1]]
    except:
        print('read results/optunacurve error, create new scores list...')
        scores = []
    def objective(trial):
        optunasetting(args,trial)
        args.plotscore = True
        optunascores = []
        for i in range(args.optuna_nloops):
            args.env_seed = i
            mainloop(args)
            print('optunascore: ',args.optunascore)
            optunascores.append(args.optunascore)
        score = np.mean(optunascores)
        scores.append(score)
        return score
    try:
        study = joblib.load('results/study.pkl')
        print('Best trial until now:')
        print(' Value: ', study.best_trial.value)
        print(' Params: ')
        for key, value in study.best_trial.params.items():
            print(f'    {key}: {value}')
    except:
        study = optuna.create_study()
    try:
        study.optimize(objective, n_trials=args.optuna_trials)
    except:
        print('error in optimize...')
        pass
    joblib.dump(study, 'results/study.pkl')
    with open('results/optunalogs','a') as foptunalogs:
        pprint.pprint(study.best_trial,foptunalogs)
        pprint.pprint(study.trials,foptunalogs)
    with open('results/optunacurve','w') as foptunacurve:
        for scorei in scores:
            print(scorei,end=',',file=foptunacurve)
    plt.figure()
    plt.plot(scores)
    plt.savefig('results/optunacurve.png', figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
    plt.close()
class Info_recorder:
    def __init__(self,args):
        self.args = args
        self.last_scores = deque(maxlen=100)
        self.last_scores_best = float("-inf")
        self.last_scores_means = []
    def record(self,info):
        for infoi in info:#enumerate(info):
            if self.args.optuna_trials:
                last_score_name = optunascoring()
                if last_score_name in infoi:
                    self.last_scores.append(infoi[last_score_name])
            else:
                if 'last_score' in infoi:
                    self.last_scores.append(infoi['last_score'])
        if n%self.args.roll_num==0 and t%(self.args.max_train_steps//100)==0:#len(last_scores) == last_scores.maxlen:
            if len(self.last_scores)!=0:
                self.last_scores_mean = np.mean(last_scores)
                self.last_scores_means.append(last_scores_mean)
            #if last_scores_mean >= last_scores_best and t > args.max_train_steps*0.8:
            #    last_scores_best = last_scores_mean
            #    agt.save(str(args.env_seed))#+'_'+str(t))
@profile # perf_counter used, not process_time,count real time,not cpu time
def train(args,env,agt):#synchronized multienv oneagent
    starttime = time.time()
    with open(args.exp_dir[:-1]+'_args', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    try:
        if args.to_load: agt.load()
        obs = env.reset()
        args.max_train_steps = int(args.max_steps // args.env_num // args.roll_num)
        iterator = tqdm.tqdm(range(args.max_train_steps))
        for t in iterator:
            for n in range(args.roll_num):
                #print(n,'start')
                act, act_info = agt.getaction(obs,explore=True)
                #print(n,'1')
                new_obs, rew, done, info = env.step(act)#must create a new_obs each step
                #print(n,'2')
                agt.memoexps(new_obs, rew, done, info)#must not to change new_obs
                #print(n,'3')
                obs = new_obs
                #print(n,'end')
            agt.update(t, args.max_train_steps, info_in={})
        agt.save(str(args.env_seed)+'_'+str(t))
    except KeyboardInterrupt:
        agt.save(str(args.env_seed)+'_'+str(t)) # if pass, files will not have enough time to close...
    with open(args.exp_dir[:-1]+'_configs','a') as fconfigs:
        pprint.pprint(args,fconfigs)
        print(time.ctime(starttime),' ------ ',time.ctime(time.time()),'        ',
            (time.time()-starttime)//3600,'hours',np.round((time.time()-starttime)%3600/60,1),'minutes',file=fconfigs)
def test(args,env,agt):
    agt.load()
    obs = env.reset()
    args.max_test_steps = int(args.test_steps // args.env_num)
    iterator = tqdm.tqdm(range(args.max_test_steps))
    for t in iterator:
        if args.render: env.render(mode='rgb_array')
        act, act_info = agt.getaction(obs,explore=False)
        new_obs, rew, done, info = env.step(act)
        agt.memoexps(new_obs, rew, done, info)
        obs = new_obs
def mainloop(args):
    random.seed(args.env_seed)
    np.random.seed(args.env_seed)
    args.exp_dir = 'results/'+args.env_name+'_'+str(args.env_num)+'_'+str(args.roll_num)
    envirs.add_strings(args)
    agents.add_strings(args)
    algos.add_strings(args)
    args.exp_dir = args.exp_dir+'/'
    print('exp_dir length: ',len(args.exp_dir)) # check whether folder name over length limits
    os.makedirs(args.exp_dir,exist_ok=True)
    args.max_steps = int(float(args.max_stepsM)*1e6)
    args.lr        = float(args.lr_M)*1e-6
    if args.test_steps and args.render: args.env_num = 1
    env = envirs.getEnvir(args)
    agt = agents.getAgent(args,env)
    print(args.env_seed,':',args.fin_seed)
    if args.test_steps: test(args,env,agt)
    else:              train(args,env,agt)
    env.close()
def main():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--start-step', type=int, default=0, help='number of environment step that start to train (default: 0)')
    parser.add_argument('--minmax-score', default='0.0,0.0', help='min max score (default: 0.0 to 0.0)')
    parser.add_argument('--env-seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--fin-seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--teamagts', default='1', help='pomdp parameters')
    parser.add_argument('--env-type', default='', help='atari or sc2 flag (default: )')
    parser.add_argument('--env-name', default='', help='environment name (default: )')#BreakoutNoFrameskip-v4
    # parser.add_argument('--env-nums', type=int, default=1, help='how many training CPU processes to use (default: 12)')
    parser.add_argument('--env-num', type=int, default=1, help='how many training CPU processes to use (default: 1)')
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

    parser.add_argument('--debug', action='store_true', default=False, help='debug flag')
    parser.add_argument('--timer', action='store_true', default=False, help='timer flag')
    parser.add_argument('--optuna-trials', type=int, default=0, help='optuna trial times')
    parser.add_argument('--optuna-nloops', type=int, default=2, help='number of seeds for each optuna trial')
    parser.add_argument('--optuna-tlimit', type=int, default=300, help='train time limit for each optuna trial')

    parser.add_argument('--infowidth', type=int, default=16, help='width for info render (default: 16)')
    parser.add_argument('--infoheight', type=int, default=16, help='height for info render (default: 16)')
    parser.add_argument('--infodpi', type=int, default=36, help='dpi for info render (default: 36)')
    args = parser.parse_args()
    if args.optuna_trials: optunaloop(args)
    else:                  mainloop(args)
if __name__ == '__main__':
    main()
