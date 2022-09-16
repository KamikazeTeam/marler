import torch.nn as nn

debug = 0


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Network(nn.Module):
    def __init__(self, num_inputs, num_outputs, paraslist):
        self.debugflag = True
        super(Network, self).__init__()
        print('cnn2d', paraslist)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        layers = []
        if num_inputs[2] == 1:
            print('Squeeze stack axis!')
            layer_inputs = num_inputs[3]
            self.squeezeaxis = 2
        elif num_inputs[3] == 1:
            print('Squeeze channel axis!')
            layer_inputs = num_inputs[2]
            self.squeezeaxis = 1
        else:
            print('At least one of channel or stack need to be 1!')
            exit()
        if debug or self.debugflag: print('layer s :', num_inputs)
        for i, paras in enumerate(paraslist):
            para = paras.split(',')
            para = [int(parai) for parai in para]
            if para[7] == 0: padding_mode = 'zeros'
            if para[7] == 1: padding_mode = 'circular'  # 'reflect', 'replicate' or 'circular'
            layer = init_(nn.Conv2d(layer_inputs, para[6], kernel_size=(para[0], para[1]), stride=(para[2], para[3]),
                                    padding=(para[4], para[5]), padding_mode=padding_mode))
            layers.append(layer)
            layer_inputs = para[6]
            layers.append(nn.ReLU())
            num_inputs[0] = (num_inputs[0] + para[4] * 2 - (para[0] - para[2])) // para[2]
            num_inputs[1] = (num_inputs[1] + para[5] * 2 - (para[1] - para[3])) // para[3]
            if debug or self.debugflag: print('layer', i, ':', num_inputs[:2], layer_inputs)
        if debug or self.debugflag: print('layer l :', num_outputs)
        layers.append(nn.Flatten())  # Flatten())
        layers.append(init_(nn.Linear(layer_inputs * num_inputs[0] * num_inputs[1], num_outputs)))
        layers.append(nn.ReLU())
        self.num_outputs = num_outputs
        self.main = nn.Sequential(*layers)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(num_outputs, 1))
        self.train()

    def forward(self, inputs):
        # print('self.debugflag',self.debugflag)
        # print('self.squeezeaxis',self.squeezeaxis)
        inputs = inputs.squeeze(dim=self.squeezeaxis)
        x = self.main(inputs)
        return self.critic_linear(x), x
