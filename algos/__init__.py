import torch
import os
import glob


class PTAlgo:
    def __init__(self, obs_space, act_space, args):
        self.obs_space, self.act_space, self.args = obs_space, act_space, args
        torch.manual_seed(args.env_seed)
        torch.cuda.manual_seed_all(args.env_seed)
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def create_optimizer(self, model):
        if self.args.opt == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), self.args.lr, weight_decay=self.args.opt_eps,
                                        momentum=self.args.opt_alpha, nesterov=True)  # dampening=d)
        if self.args.opt == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), self.args.lr,
                                            eps=self.args.opt_eps, alpha=self.args.opt_alpha)
        if self.args.opt == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), self.args.lr,
                                         eps=self.args.opt_eps)
        if self.args.opt == 'Adadelta':
            optimizer = torch.optim.Adadelta(model.parameters(), self.args.lr)
        return optimizer

    def create_scheduler(self, optimizer):
        scheduler = None
        decay_paras = self.args.decay_paras.split(',')  # eta_min_ratio, T_2, exp_gamma
        if self.args.decay == 'const':
            pass
        if self.args.decay == 'linear':
            self.eta_min_ratio = float(decay_paras[0])
        if self.args.decay == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(decay_paras[1]),
                                                        gamma=float(decay_paras[2]))
        if self.args.decay == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(decay_paras[2]))
        if self.args.decay == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(decay_paras[1]),
                                                                   eta_min=float(decay_paras[0]) * self.args.lr)
        return scheduler

    def update_scheduler(self, crt_step, max_step, scheduler, optimizer):
        if self.args.decay == 'const':
            return
        if self.args.decay == 'linear':
            lr = self.args.lr * ((1 - self.eta_min_ratio) * (1 - crt_step / max_step) + self.eta_min_ratio)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            return
        scheduler.step()

    def memoexps(self, new_obs, rew, done, info):
        pass

    def save(self, name, prefix=''):
        folder_name = self.args.exp_dir + 'models/'
        os.makedirs(folder_name, exist_ok=True)
        torch.save(self.model, folder_name + prefix + name)

    def load(self, prefix='', folder=''):
        try:
            print('load_model folder:', folder)
            print('load_model prefix:', prefix)
            if folder == '':
                flist = glob.glob(self.args.exp_dir + 'models/' + prefix + '*')
            else:
                flist = glob.glob(folder + prefix + '*')
            print('load_model flist:', [fname.split('/')[-1] for fname in flist])
            flist = [ffile for ffile in flist if os.path.isfile(ffile)]
            print('load_model files:', [fname.split('/')[-1] for fname in flist])
            ffile = max(flist, key=os.path.getmtime)  # ctime)
            print('load_model ffile:', ffile.split('/')[-1])
            self.model = torch.load(ffile)
            self.model.eval()
            # torch.save(self.model.state_dict(),'./pymodel')
            return ffile
        except:
            print('Error when trying to load model...Skipped.')
            print('load_model folder:', folder)
            print('load_model prefix:', prefix)
            return None
