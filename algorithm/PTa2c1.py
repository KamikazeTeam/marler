import numpy as np
import torch


class Algorithm:
    def __init__(self, args):
        self.args = args

    def update(self, crt_step, max_step, info_in, model):
        device = model.device
        model.scheduler_step(crt_step, max_step)

        mb_obs = torch.from_numpy(info_in['mb_obs']).to(device).float()
        mb_act = torch.from_numpy(info_in['mb_act']).to(device)
        if model.act_space.__class__.__name__ == 'Discrete':  #
            mb_act = mb_act.long()
        mb_nob = torch.from_numpy(info_in['mb_new_obs']).to(device).float()
        mb_rew = torch.from_numpy(info_in['mb_rew']).to(device).float().unsqueeze(-1)
        mb_done_int = info_in['mb_done'].astype(int)
        mb_done_inv = np.ones(mb_done_int.shape) - mb_done_int
        mb_mask = torch.from_numpy(np.expand_dims(mb_done_inv, -1)).to(device).float()
        returns = torch.zeros(info_in['mb_rew'].shape[0] + 1, *info_in['mb_rew'].shape[1:], 1).to(device).float()

        returns[-1] = model.get_value(mb_nob[-1]).detach()
        for step in reversed(range(mb_rew.size(0))):
            returns[step] = returns[step + 1] * self.args.gamma * mb_mask[step] + mb_rew[step]
        returns = returns[:-1]

        mb_obs_batch = mb_obs.view(mb_obs.shape[0] * mb_obs.shape[1], *mb_obs.shape[2:])
        if model.act_space.__class__.__name__ == 'Discrete':  #
            mb_act_batch = mb_act.view(-1, 1)
        else:
            mb_act_batch = mb_act.view(-1, model.act_space.shape[0])
        values, action_log_probs, dist_entropy = model.get_loss(mb_obs_batch, mb_act_batch)

        returns = returns.view(-1, 1)
        advantages = returns - values
        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()

        model.optimizer.zero_grad()
        loss = value_loss * self.args.loss_value_weight + action_loss - dist_entropy * self.args.loss_entropy_weight
        loss.backward()
        torch.cuda.synchronize()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_max_norm)
        model.optimizer.step()
        return value_loss.item(), action_loss.item(), dist_entropy.item()


def get_algorithm(args):
    algorithm = Algorithm(args)
    return algorithm
