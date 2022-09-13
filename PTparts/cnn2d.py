import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist
import torch.nn.functional as F
debug = 0
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
class Network(nn.Module):
    def __init__(self, num_inputs, num_outputs, paraslist):
        self.debugflag = True
        super(Network, self).__init__()
        print('cnn2d',paraslist)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        layers = []
        if   num_inputs[2]==1:
            print('Squeeze stack axis!')
            layer_inputs = num_inputs[3]
            self.squeezeaxis = 2
        elif num_inputs[3]==1:
            print('Squeeze channel axis!')
            layer_inputs = num_inputs[2]
            self.squeezeaxis = 1
        else:
            print('At least one of channel or stack need to be 1!')
            exit()
        if debug or self.debugflag: print('layer s :', num_inputs)
        for i,paras in enumerate(paraslist):
            para = paras.split(',')
            para = [int(parai) for parai in para]
            if para[7]==0: padding_mode = 'zeros'
            if para[7]==1: padding_mode = 'circular'#'reflect', 'replicate' or 'circular'
            layer = init_(nn.Conv2d(layer_inputs, para[6], kernel_size=(para[0],para[1]), stride=(para[2],para[3]), padding=(para[4],para[5]), padding_mode=padding_mode))
            layers.append(layer)
            layer_inputs = para[6]
            layers.append(nn.ReLU())
            num_inputs[0] = (num_inputs[0]+para[4]*2-(para[0]-para[2]))//para[2]
            num_inputs[1] = (num_inputs[1]+para[5]*2-(para[1]-para[3]))//para[3]
            if debug or self.debugflag: print('layer', i, ':', num_inputs[:2], layer_inputs)
        if debug or self.debugflag: print('layer l :', num_outputs)
        layers.append(nn.Flatten())#Flatten())
        layers.append(init_(nn.Linear(layer_inputs*num_inputs[0]*num_inputs[1], num_outputs)))
        layers.append(nn.ReLU())
        self.num_outputs = num_outputs
        self.main = nn.Sequential(*layers)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(num_outputs, 1))
        self.train()
    def forward(self, inputs):
        #print('self.debugflag',self.debugflag)
        #print('self.squeezeaxis',self.squeezeaxis)
        inputs = inputs.squeeze(dim=self.squeezeaxis)
        x = self.main(inputs)
        return self.critic_linear(x), x

class NetworkL2(nn.Module):
    def __init__(self, num_inputs, num_outputs, paraslist):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.num_outputs = 500
    def forward(self, x):
        x = x.squeeze(dim=2)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.reshape(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        return torch.tensor(0),x
class NetworkL(nn.Module):#"LeNet" sgd momentum=0.9
    def __init__(self, num_inputs, num_outputs, paraslist):
        super(Network, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        layers = []
        layers.append(init_(nn.Conv2d(1, 20, 5, 1))) #10
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        layers.append(init_(nn.Conv2d(20, 50, 5, 1))) #20
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Flatten())
        layers.append(init_(nn.Linear(4*4*50, 500))) #320, 50
        layers.append(nn.ReLU())
        self.num_outputs = 500
        self.main = nn.Sequential(*layers)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(self.num_outputs, 1))
        self.train()
    def forward(self, inputs):
        inputs = inputs.squeeze(dim=2)
        x = self.main(inputs)
        return torch.tensor(0),x#self.critic_linear(x), x
#mnist --aprxfunc cnn2d --apfparas 6,6,2,2,0,0,64,0^3,3,2,2,1,1,64,0^3,3,2,2,1,1,64,0=144#576 batch_size=500(2min10sec) or 50(6min) Maxstep=1.4M 6min
#--algo PTa2c1 --lr-M 20000 --decay linear --decayparas 0.01,, --opt Adam --alpha 0.0 --vlossratio 0.0 --entropycoef 0.0 \
class NetworkC(nn.Module):#cifartest
    def __init__(self, num_inputs, num_outputs, paraslist):
        super(Network, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        layers = []
        layers.append(init_(nn.Conv2d(3, 48, 3, 1, 1)))
        layers.append(nn.ReLU())
        #layers.append(nn.MaxPool2d(2))
        layers.append(init_(nn.Conv2d(48, 96, 3, 1, 1)))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        layers.append(init_(nn.Conv2d(96, 192, 3, 1, 1)))
        layers.append(nn.ReLU())
        #layers.append(nn.MaxPool2d(2))
        layers.append(init_(nn.Conv2d(192, 256, 3, 1, 1)))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Flatten())
        layers.append(init_(nn.Linear(8*8*256, 512)))
        layers.append(nn.ReLU())
        layers.append(init_(nn.Linear(512, 64)))
        layers.append(nn.ReLU())
        self.num_outputs = 64#576#144
        self.main = nn.Sequential(*layers)
        #init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        #self.critic_linear = init_(nn.Linear(self.num_outputs, 1))
        #self.train()
    def forward(self, inputs):
        inputs = inputs.squeeze(dim=2)
        x = self.main(inputs)
        return torch.tensor(0), x
class NetworkA(nn.Module): # mnist test Adadelta StepLR
    def __init__(self, num_inputs, num_outputs, paraslist):
        super(Network, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        layers = []
        layers.append(init_(nn.Conv2d(1, 32, 3, 1)))
        layers.append(nn.ReLU())
        layers.append(init_(nn.Conv2d(32, 64, 3, 1)))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Flatten())
        layers.append(init_(nn.Linear(9216, 128)))
        layers.append(nn.ReLU())
        self.num_outputs = 128
        self.main = nn.Sequential(*layers)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(self.num_outputs, 1))
        self.train()
    def forward(self, inputs):
        inputs = inputs.squeeze(dim=2)
        x = self.main(inputs)
        return self.critic_linear(x), x
