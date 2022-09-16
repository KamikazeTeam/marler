import torch.nn as nn
import torch.distributions as tdist

debug = 0


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


log_prob_cat = tdist.Categorical.log_prob
tdist.Categorical.mode = lambda self: self.probs.argmax(dim=-1)
tdist.Categorical.log_probs = tdist.Categorical.log_prob


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return tdist.Categorical(logits=x)
