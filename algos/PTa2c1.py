import numpy as np
import torch
import algos
import importlib
from PTparts import dists


class Policy(torch.nn.Module):
    def __init__(self, obs_space, act_space, args, device):
        super(Policy, self).__init__()
        self.obs_space, self.act_space, self.args = obs_space, act_space, args
        self.device = device
        approx_func_paras = args.approx_func_paras.split('=')
        cnn_paras = approx_func_paras[0].split('^')
        mlp_paras = approx_func_paras[1].split('^')
        inputs_shape = [obs_space.shape[0], obs_space.shape[1], args.stack_num, obs_space.shape[2]]
        mod = importlib.import_module('PTparts.' + args.approx_func)
        self.base = mod.Network(num_inputs=inputs_shape, num_outputs=int(mlp_paras[-1]), paraslist=cnn_paras)
        self.dist = dists.Categorical(self.base.num_outputs, act_space.n)
        self.base = self.base.to(self.device)
        self.dist = self.dist.to(self.device)
        args.num_of_paras = int(sum([np.prod(p.size()) for p in self.parameters() if p.requires_grad]))  # p.numel()
        print(args.num_of_paras)
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
        loss = value_loss * self.args.loss_value_weight + action_loss - dist_entropy * self.args.loss_entropy_weight
        loss.backward()
        torch.cuda.synchronize()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_max_norm)
        self.optimizer.step()
        return value_loss.item(), action_loss.item(), dist_entropy.item()


def get_algo(obs_space, act_space, args):
    algo = Algo(obs_space, act_space, args)
    return algo
