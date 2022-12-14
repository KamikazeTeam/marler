import torch
import torch.nn as nn
import torch.distributions as dist

debug = 0


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


log_prob_cat = dist.Categorical.log_prob
dist.Categorical.mode = lambda self: self.probs.argmax(dim=-1)
dist.Categorical.log_probs = dist.Categorical.log_prob


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return dist.Categorical(logits=x)


log_prob_normal = dist.Normal.log_prob
dist.Normal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)
normal_entropy = dist.Normal.entropy
dist.Normal.entropy = lambda self: normal_entropy(self).sum(-1)
dist.Normal.mode = lambda self: self.mean


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias


class Gaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Gaussian, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.log_std = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()
        action_log_std = self.log_std(zeros)
        return dist.Normal(action_mean, action_log_std.exp())
