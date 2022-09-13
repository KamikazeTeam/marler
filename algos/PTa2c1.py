import numpy as np
import torch, algos
import torch.nn as nn
import importlib
from PTparts import dists
from models import *

'''LeNet in PyTorch.'''
import torch.nn.functional as F


class testLeNet(nn.Module):
    def __init__(self):
        super(testLeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class net_wrapper(torch.nn.Module):
    def __init__(self, net, inputs_shape):
        super(net_wrapper, self).__init__()
        self.num_outputs = list(net.children())[-1].in_features
        self.main = net
        self.main.classifier = Identity()  ### to do, need to get last layer...
        self.main.linear = Identity()

        if inputs_shape[2] == 1:
            print('Squeeze stack axis!')
            layer_inputs = inputs_shape[3]
            self.squeezeaxis = 2
        elif inputs_shape[3] == 1:
            print('Squeeze channel axis!')
            layer_inputs = inputs_shape[2]
            self.squeezeaxis = 1
        else:
            print('At least one of channel or stack need to be 1!')
            exit()

        def init(module, weight_init, bias_init, gain=1):
            weight_init(module.weight.data, gain=gain)
            bias_init(module.bias.data)
            return module

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.critic_linear = init_(nn.Linear(self.num_outputs, 1))

    def forward(self, inputs):
        inputs = inputs.squeeze(dim=self.squeezeaxis)
        # print(inputs.shape)
        inputs = inputs.contiguous()  # for .view function in networks
        x = self.main(inputs)
        return self.critic_linear(x), x


class Policy(torch.nn.Module):
    def __init__(self, obs_space, act_space, args, device):
        super(Policy, self).__init__()
        self.obs_space, self.act_space, self.args = obs_space, act_space, args
        self.device = device
        apfparas = args.apfparas.split('=')
        cnncnnparas = apfparas[0].split('^')
        cnnmlpparas = apfparas[1].split('^')
        inputs_shape = [obs_space.shape[0], obs_space.shape[1], args.stack_num, obs_space.shape[2]]
        ### use models start
        netname, net = args.aprxfunc, None
        if netname == 'lenet': net = LeNet()
        if netname == 'vgg11': net = VGG('VGG11')
        if netname == 'vgg13': net = VGG('VGG13')
        if netname == 'vgg16': net = VGG('VGG16')
        if netname == 'vgg19': net = VGG('VGG19')
        if netname == 'res18': net = ResNet18()
        if netname == 'res34': net = ResNet34()
        if netname == 'res50': net = ResNet50()
        if netname == 'res101': net = ResNet101()
        if netname == 'res152': net = ResNet152()
        if netname == 'pre': net = PreActResNet18()
        if netname == 'gln': net = GoogLeNet()
        if netname == 'den': net = DenseNet121()
        if netname == 'rex': net = ResNeXt29_2x64d()
        if netname == 'mob': net = MobileNet()
        if netname == 'mo2': net = MobileNetV2()
        if netname == 'dpn': net = DPN92()
        if netname == 'shf': net = ShuffleNetG2()
        if netname == 'sen': net = SENet18()
        if netname == 'sh2': net = ShuffleNetV2(1)
        if netname == 'eff': net = EfficientNetB0()
        if netname == 'reg': net = RegNetX_200MF()
        if netname == 'sim': net = SimpleDLA()
        if netname == 'rnn': net = RNN()
        if net != None:
            self.base = net_wrapper(net, inputs_shape)
        else:  ### use models end
            mod = importlib.import_module('PTparts.' + args.aprxfunc)
            self.base = mod.Network(num_inputs=inputs_shape, num_outputs=int(cnnmlpparas[-1]), paraslist=cnncnnparas)
        if act_space.__class__.__name__ == "Discrete":
            self.dist = dists.Categorical(self.base.num_outputs, act_space.n)
        elif act_space.__class__.__name__ == "Box":
            self.dist = dists.DiagGaussian(self.base.num_outputs, act_space.shape[0])
        elif act_space.__class__.__name__ == "MultiBinary":
            self.dist = dists.Bernoulli(self.base.num_outputs, act_space.shape[0])
        else:
            raise NotImplementedError
        self.base = self.base.to(self.device)
        self.dist = self.dist.to(self.device)
        args.numparas = int(sum([np.prod(p.size()) for p in self.parameters() if p.requires_grad]))  # p.numel()
        print(args.numparas)
        for p in self.parameters():
            if p.requires_grad:
                print(np.prod(p.size()))

    def forward(self, inputs):
        raise NotImplementedError

    def get_action(self, inputs, explore):
        inputs = inputs.permute(0, 4, 1, 2, 3) / 255.0  # go to batch,channel,stack,width,height
        with torch.no_grad():
            value, actor_features = self.base(inputs)
            dist = self.dist(actor_features)
            if not explore:
                action = dist.mode()
            else:
                action = dist.sample()
        info_p = {}
        return action, info_p

    def get_value(self, inputs):
        inputs = inputs.permute(0, 4, 1, 2, 3) / 255.0  # go to batch,channel,stack,width,height
        with torch.no_grad():
            value, _ = self.base(inputs)
        return value

    def get_loss(self, inputs, actions):
        inputs = inputs.permute(0, 4, 1, 2, 3) / 255.0  # go to batch,channel,stack,width,height
        values, actor_features = self.base(inputs)
        dist = self.dist(actor_features)
        if self.act_space.__class__.__name__ == "Discrete":
            action_log_probs = dist.log_probs(actions.squeeze(-1)).unsqueeze(-1)
        else:
            action_log_probs = dist.log_probs(actions)
        dist_entropy = dist.entropy().mean()
        return values, action_log_probs, dist_entropy


class Algo(algos.PTAlgo):
    def __init__(self, obs_space, act_space, args):
        algos.PTAlgo.__init__(self, obs_space, act_space, args)
        self.obs_space, self.act_space, self.args = obs_space, act_space, args
        self.device = torch.device("cuda:0")
        self.model = Policy(obs_space, act_space, args, self.device)
        self.optimizer = self.create_optimizer(self.model)
        self.scheduler = self.create_scheduler(self.optimizer)
        if self.act_space.__class__.__name__ == 'Discrete':
            self.action_shape = 1
        else:
            self.action_shape = self.act_space.shape[0]

    def get_action(self, inputs, explore):
        inputs = torch.from_numpy(inputs).float().to(self.device)
        if self.args.env_mode == 'supervise': explore = False
        action, info_p = self.model.get_action(inputs, explore)
        torch.cuda.synchronize()
        action = action.cpu().numpy()
        return action, info_p

    def get_value(self, inputs):
        value = self.model.get_value(inputs)
        return value

    def update(self, crt_step, max_step, info_in):
        self.update_scheduler(crt_step, max_step, self.scheduler, self.optimizer)

        self.mb_obs = torch.from_numpy(info_in['mb_obs']).to(self.device).float()
        self.mb_act = torch.from_numpy(info_in['mb_act']).to(self.device)
        if self.act_space.__class__.__name__ == 'Discrete': self.mb_act = self.mb_act.long()
        self.mb_nob = torch.from_numpy(info_in['mb_new_obs']).to(self.device).float()
        self.mb_rew = torch.from_numpy(info_in['mb_rew']).to(self.device).float().unsqueeze(-1)
        mb_done_int = info_in['mb_done'].astype(int)
        mb_done_inv = np.ones(mb_done_int.shape) - mb_done_int
        self.mb_mask = torch.from_numpy(np.expand_dims(mb_done_inv, -1)).to(self.device).float()
        # self.mb_mask   = torch.from_numpy(memo_done).float().unsqueeze(-1).to(self.device)

        # returnsshape   = (info_in['mb_rew'].shape[0]+1, *info_in['mb_rew'].shape[1:], 1)
        self.returns = torch.zeros(info_in['mb_rew'].shape[0] + 1, *info_in['mb_rew'].shape[1:], 1).to(
            self.device).float()

        self.returns[-1] = self.get_value(self.mb_nob[-1]).detach()
        for step in reversed(range(self.mb_rew.size(0))):
            self.returns[step] = self.returns[step + 1] * self.args.gamma * self.mb_mask[step] + self.mb_rew[step]
        returns = self.returns[:-1]

        self.mb_obs_batch = self.mb_obs.view(-1, self.args.stack_num, *self.obs_space.shape)
        self.mb_act_batch = self.mb_act.view(-1, self.action_shape)
        values, action_log_probs, dist_entropy = self.model.get_loss(self.mb_obs_batch, self.mb_act_batch)

        returns = returns.view(-1, 1)
        advantages = returns - values
        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.optimizer.zero_grad()
        (value_loss * self.args.vlossratio + action_loss - dist_entropy * self.args.entropycoef).backward()
        torch.cuda.synchronize()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        return value_loss.item(), action_loss.item(), dist_entropy.item()


def fAlgo(obs_space, act_space, args):
    algo = Algo(obs_space, act_space, args)
    return algo
