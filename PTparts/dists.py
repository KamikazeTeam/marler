import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist
debug = 0
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
log_prob_bernoulli = tdist.Bernoulli.log_prob
tdist.Bernoulli.log_probs = lambda self, actions: log_prob_bernoulli(self, actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)
bernoulli_entropy = tdist.Bernoulli.entropy
tdist.Bernoulli.entropy = lambda self: bernoulli_entropy(self).sum(-1)
tdist.Bernoulli.mode = lambda self: torch.gt(self.probs, 0.5).float()
class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.linear = init_(nn.Linear(num_inputs, num_outputs))
    def forward(self, x):
        x = self.linear(x)
        return tdist.Bernoulli(logits=x)

log_prob_cat = tdist.Categorical.log_prob
#tdist.Categorical.log_probs = lambda self, actions: log_prob_cat(self, actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)
tdist.Categorical.mode = lambda self: self.probs.argmax(dim=-1)
tdist.Categorical.log_probs = tdist.Categorical.log_prob#lambda self, actions: log_prob_cat(self, actions)#.squeeze(-1))#.view(actions.size(0), -1).sum(-1).unsqueeze(-1)
#tdist.Categorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)
#old_sample = tdist.Categorical.sample
#tdist.Categorical.sample = lambda self: old_sample(self).unsqueeze(-1)
class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
        self.linear = init_(nn.Linear(num_inputs, num_outputs))
    def forward(self, x):
        x = self.linear(x)
        return tdist.Categorical(logits=x)

log_prob_normal = tdist.Normal.log_prob
tdist.Normal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)
normal_entropy = tdist.Normal.entropy
tdist.Normal.entropy = lambda self: normal_entropy(self).sum(-1)
tdist.Normal.mode = lambda self: self.mean
# Necessary for my KFAC implementation.
class AddBias(nn.Module):###can not change name!!!
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))
    def forward(self, x):
        if x.dim() == 2: bias = self._bias.t().view(1, -1)
        else:            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias
class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))
    def forward(self, x):
        action_mean = self.fc_mean(x)
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda: zeros = zeros.cuda()
        action_logstd = self.logstd(zeros)
        return tdist.Normal(action_mean, action_logstd.exp())
