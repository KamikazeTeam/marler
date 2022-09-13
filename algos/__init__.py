def add_arguments(parser):
    parser.add_argument('--alg-mode', default='PTa2c1', help='algo to use: TFa2c1 | PTa2c1 | PTppo (tensorflow or pytorch)')#algo
    parser.add_argument('--lr-M', default='700', help='learning rate (default: 700e-6)')
    parser.add_argument('--decay', default='linear', help='decay to use: linear | exp | cos | coscos | cosdec')
    parser.add_argument('--decayparas', default='0.01,,', help='decay parameters')#0.01,55556,0.8
    parser.add_argument('--opt', default='RMSprop', help='optimizer to use: RMSprop | Adam | ooooo ()')
    parser.add_argument('--opt-eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--opt-alpha', type=float, default=0.99, help='RMSprop optimizer apha (default: 0.99)')

    parser.add_argument('--lossfunc', default='one', help='lossfunc to use: one | xxx | xxx (one)')#loss function
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--vlossratio', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument('--entropycoef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')

    parser.add_argument('--aprxfunc', default='cnnmlp', help='approximate function to use: cnnmlp | mlp | ooooo (cnnmlp)')#approximate function
    parser.add_argument('--apfparas', default='8,8,4,4,32,1^4,4,2,2,64,1^3,3,1,1,64,1=512=64', help='approximate function parameters')

    parser.add_argument('--use-proper-time-limits', action='store_true', default=False, help='compute returns taking into account time limits')#other
    parser.add_argument('--use-gae', action='store_true', default=False, help='use generalized advantage estimation')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--clip-param', type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--ppo-epoch', type=int, default=4, help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32, help='number of batches for ppo (default: 32)')
def add_strings(args):
    args.exp_dir=args.exp_dir+':'+args.alg_mode+'_'+str(args.lr_M)+'_'+args.decay+'_'+args.decayparas+'_'+args.opt
    args.exp_dir=args.exp_dir+':'+args.aprxfunc+'_'+args.apfparas
import importlib
def getAlgo(obs_space,act_space,args):
    mod = importlib.import_module('algos.'+args.alg_mode)
    return mod.fAlgo(obs_space,act_space,args)

import torch,math,os,glob
class PTAlgo():
    def __init__(self,obs_space,act_space,args):
        self.obs_space, self.act_space, self.args = obs_space, act_space, args
        torch.manual_seed(args.env_seed)
        torch.cuda.manual_seed_all(args.env_seed)
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    def create_optimizer(self,model):
        if self.args.opt=='SGD':     optimizer = torch.optim.SGD(model.parameters(), self.args.lr, weight_decay=self.args.opt_eps, momentum=self.args.opt_alpha, nesterov=True)#dampening=d)
        if self.args.opt=='RMSprop': optimizer = torch.optim.RMSprop(model.parameters(), self.args.lr, eps=self.args.opt_eps, alpha=self.args.opt_alpha)
        if self.args.opt=='Adam':    optimizer = torch.optim.Adam(model.parameters(), self.args.lr, eps=self.args.opt_eps)
        if self.args.opt=='Adadelta':optimizer = torch.optim.Adadelta(model.parameters(), self.args.lr)
        return optimizer
    def create_scheduler(self,optimizer):
        scheduler = None
        decayparas = self.args.decayparas.split(',') # eta_min_ratio, T_2, exp_gamma
        if self.args.decay=='const':  pass
        if self.args.decay=='linear': self.eta_min_ratio = float(decayparas[0])
        if self.args.decay=='step':   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(decayparas[1]), gamma=float(decayparas[2]))
        if self.args.decay=='exp':    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(decayparas[2]))
        if self.args.decay=='cos':    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(decayparas[1]), eta_min=float(decayparas[0])*self.args.lr)
        return scheduler
    def update_scheduler(self,crt_step,max_step,scheduler,optimizer):
        if self.args.decay=='const': return
        if self.args.decay=='linear':
            lr = self.args.lr*((1-self.eta_min_ratio)*(1-crt_step/max_step)+self.eta_min_ratio)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            return
        scheduler.step()
    def memoexps(self, new_obs, rew, done, info):
        pass
    def save(self,name,prefix=''):
        foldername = self.args.exp_dir+'models/'
        os.makedirs(foldername,exist_ok=True)
        torch.save(self.model, foldername+prefix+name)
    def load(self,prefix='',folder=''):
        try:
            print('load_model folder:',folder)
            print('load_model prefix:',prefix)
            if folder=='': flist = glob.glob(self.args.exp_dir+'models/'+prefix+'*')
            else:          flist = glob.glob(folder+prefix+'*')
            print('load_model flist:',[fname.split('/')[-1] for fname in flist])
            flist = [ffile for ffile in flist if os.path.isfile(ffile)]
            print('load_model files:',[fname.split('/')[-1] for fname in flist])
            ffile = max(flist, key=os.path.getmtime)#ctime)
            print('load_model ffile:',ffile.split('/')[-1])
            self.model = torch.load(ffile)
            self.model.eval()
            #torch.save(self.model.state_dict(),'./pymodel')
            return ffile
        except:
            print('Error when trying to load model...Skipped.')
            print('load_model folder:',folder)
            print('load_model prefix:',prefix)
            return None
