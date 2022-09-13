import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist
debug = 0
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)
class Network(nn.Module):
    def __init__(self, num_inputs, num_outputs, paraslist):
        self.debugflag = True
        super(Network, self).__init__()
        print('cnn3d',paraslist)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        layers = []
        layer_inputs = num_inputs[3]
        if debug or self.debugflag: print('layer s :', num_inputs)
        for i,paras in enumerate(paraslist):
            para = paras.split(',')
            para = [int(parai) for parai in para]
            layer = init_(nn.Conv3d(layer_inputs, para[9], kernel_size=(para[2],para[0],para[1]), 
                                    stride=(para[5],para[3],para[4]), padding=(para[8],para[6],para[7])))
            layers.append(layer)
            layer_inputs = para[9]
            layers.append(nn.ReLU())
            num_inputs[0] = (num_inputs[0]+para[6]*2-(para[0]-para[3]))//para[3]
            num_inputs[1] = (num_inputs[1]+para[7]*2-(para[1]-para[4]))//para[4]
            num_inputs[2] = (num_inputs[2]+para[8]*2-(para[2]-para[5]))//para[5]
            if debug or self.debugflag: print('layer', i, ':', num_inputs[:3], layer_inputs) ###
        if debug or self.debugflag: print('layer l :', num_outputs) ###
        layers.append(Flatten())
        layers.append(init_(nn.Linear(layer_inputs*num_inputs[2]*num_inputs[0]*num_inputs[1], num_outputs)))
        layers.append(nn.ReLU())
        self.main = nn.Sequential(*layers)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(num_outputs, 1))
        self.num_outputs = num_outputs
        self.train()
    def forward(self, inputs):
        x = self.main(inputs)
        return self.critic_linear(x), x
