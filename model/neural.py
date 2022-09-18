import numpy as np
import torch
import importlib
from network import dists
import os
import glob


class Policy(torch.nn.Module):
    def __init__(self, args, obs_space, act_space):
        super(Policy, self).__init__()
        self.args = args
        self.device = torch.device("cuda:0")
        approx_func_paras = args.approx_func_paras.split('=')
        cnn_paras = approx_func_paras[0].split('^')
        mlp_paras = approx_func_paras[1].split('^')
        inputs_shape = [obs_space.shape[0], obs_space.shape[1], args.stack_num, obs_space.shape[2]]
        mod = importlib.import_module('network.' + args.approx_func)
        self.base = mod.Network(num_inputs=inputs_shape, num_outputs=int(mlp_paras[-1]), paraslist=cnn_paras)
        if act_space.__class__.__name__ == "Discrete":
            self.dist = dists.Categorical(self.base.num_outputs, act_space.n)
        # elif act_space.__class__.__name__ == "Box":
        #     self.dist = dists.DiagGaussian(self.base.num_outputs, act_space.shape[0])
        self.base = self.base.to(self.device)
        self.dist = self.dist.to(self.device)
        args.num_of_paras = int(sum([np.prod(p.size()) for p in self.parameters() if p.requires_grad]))  # p.numel()
        print(args.num_of_paras)
        for p in self.parameters():
            if p.requires_grad:
                print(np.prod(p.size()))
        self.act_space = act_space

        # create optimizer
        if self.args.opt == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), self.args.lr, weight_decay=self.args.opt_eps,
                                             momentum=self.args.opt_alpha, nesterov=True)  # dampening=d)
        if self.args.opt == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), self.args.lr,
                                                 eps=self.args.opt_eps, alpha=self.args.opt_alpha)
        if self.args.opt == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), self.args.lr, eps=self.args.opt_eps)
        if self.args.opt == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(self.parameters(), self.args.lr)
        # create scheduler
        self.scheduler = None
        decay_paras = self.args.decay_paras.split(',')  # eta_min_ratio, T_2, exp_gamma
        if self.args.decay == 'const':
            pass
        if self.args.decay == 'linear':
            self.eta_min_ratio = float(decay_paras[0])
        if self.args.decay == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(decay_paras[1]),
                                                             gamma=float(decay_paras[2]))
        if self.args.decay == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=float(decay_paras[2]))
        if self.args.decay == 'cos':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=int(decay_paras[1]),
                                                                        eta_min=float(decay_paras[0]) * self.args.lr)

    def scheduler_step(self, crt_step, max_step):
        if self.args.decay == 'const':
            pass
        elif self.args.decay == 'linear':
            lr = self.args.lr * ((1 - self.eta_min_ratio) * (1 - crt_step / max_step) + self.eta_min_ratio)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.scheduler.step()

    def forward(self, inputs):
        raise NotImplementedError

    def input_permute(self, _inputs):
        return _inputs.permute(0, 4, 1, 2, 3) / 255.0  # go to batch,channel,stack,width,height

    def get_action(self, _inputs, explore):
        inputs = torch.from_numpy(_inputs).float().to(self.device)

        inputs = self.input_permute(inputs)
        with torch.no_grad():
            value, actor_features = self.base(inputs)
            dist = self.dist(actor_features)
            if not explore:
                action = dist.mode()
            else:
                action = dist.sample()
        info_p = {}

        torch.cuda.synchronize()
        action = action.cpu().numpy()
        return action, info_p

    def get_value(self, inputs):
        inputs = self.input_permute(inputs)
        with torch.no_grad():
            value, _ = self.base(inputs)
        return value

    def get_loss(self, inputs, actions):
        inputs = self.input_permute(inputs)
        values, actor_features = self.base(inputs)
        dist = self.dist(actor_features)
        if self.act_space.__class__.__name__ == "Discrete":
            action_log_probs = dist.log_probs(actions.squeeze(-1)).unsqueeze(-1)
        else:
            action_log_probs = dist.log_probs(actions)
        dist_entropy = dist.entropy().mean()
        return values, action_log_probs, dist_entropy

    def save(self, name, prefix=''):
        folder_name = self.args.exp_dir + 'models/'
        os.makedirs(folder_name, exist_ok=True)
        torch.save(self.state_dict(), folder_name + prefix + name)

    def load(self, folder='', prefix=''):
        try:
            print('load_model folder:', folder)
            print('load_model prefix:', prefix)
            if folder == '':
                file_list = glob.glob(self.args.exp_dir + 'models/' + prefix + '*')
            else:
                file_list = glob.glob(folder + prefix + '*')
            print('load_model file_list all :', [file_name.split('/')[-1] for file_name in file_list])
            file_list = [file_name for file_name in file_list if os.path.isfile(file_name)]
            print('load_model file_list file:', [file_name.split('/')[-1] for file_name in file_list])
            file_name = max(file_list, key=os.path.getmtime)  # ctime)
            print('load_model file_name:', file_name.split('/')[-1])
            self.load_state_dict(torch.load(file_name))
            self.eval()
            # torch.save(self.model.state_dict(),'./pymodel')
            return file_name
        except:
            print('Error when trying to load model...Skipped.')
            print('load_model folder:', folder)
            print('load_model prefix:', prefix)
            return None


def get_model(args, obs_space, act_space):
    model = Policy(args, obs_space, act_space)
    return model
