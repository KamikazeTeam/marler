import numpy as np
import torch


class Algorithm:
    def __init__(self, args):
        self.args = args

    def update(self, crt_step, max_step, info_in, model):
        return 0, 0, 0


def get_algorithm(args):
    algorithm = Algorithm(args)
    return algorithm
