import numpy as np
import torch


class Policy(torch.nn.Module):
    def __init__(self, args, obs_space, act_space):
        super(Policy, self).__init__()
        self.args = args
        self.act_space = act_space

    def scheduler_step(self, crt_step, max_step):
        pass

    def optimizer_zero_grad(self):
        pass

    def optimizer_step(self):
        pass

    def forward(self, inputs):
        raise NotImplementedError

    @staticmethod
    def input_permute(_inputs):
        return _inputs.permute(0, 4, 1, 2, 3) / 255.0  # go to batch,channel,stack,width,height

    def get_action(self, _inputs, explore):
        # inputs = torch.from_numpy(_inputs).float().to(self.device)
        info_p = {}
        # action = np.zeros((_inputs.shape[0],))
        action = []
        for _ in range(_inputs.shape[0]):
            action.append(self.act_space.sample())
        action = np.array(action)
        action = action.squeeze()
        return action, info_p

    def get_value(self, inputs):
        value = torch.zeros(inputs.shape[0], 1)
        return value

    def get_loss(self, inputs, actions):
        values = torch.zeros(inputs.shape[0], 1)
        action_log_probs = torch.zeros(inputs.shape[0], 1)
        dist_entropy = torch.zeros(1)
        return values, action_log_probs, dist_entropy

    def save(self, name, prefix=''):
        pass

    def load(self, folder='', prefix=''):
        pass


def get_model(args, obs_space, act_space):
    model = Policy(args, obs_space, act_space)
    return model
