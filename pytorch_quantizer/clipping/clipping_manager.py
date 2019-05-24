import torch
from torch.autograd import Function
import numpy as np
import int_quantization
import math
from utils.monitor import Monitor
from pytorch_quantizer.quantization.inference.statistic_manager import StatisticManager as SM


class StatisticalClipper(Function):
    def __init__(self, rho):
        self.rho = rho

    def __call__(self, tensor, tag="", stat_id=None, inplace=False):
        cls_layer = (tag == 'activation_linear' and tensor.shape[1] == 1000)
        if stat_id is not None and not cls_layer:
            kind_max = {'min': 'min', 'max': 'max', 'mean': 'mean', 'std': 'mean', 'range': 'mean', 'mean_abs': 'mean',
                    'b': 'mean'}
            kind_avg = {'min': 'mean', 'max': 'mean', 'mean': 'mean', 'std': 'mean', 'range': 'mean', 'mean_abs': 'mean',
                    'b': 'mean'}
            min_min, max_max, _, _, _, _, _ = SM().get_tensor_stats(stat_id, kind_max)
            avg_min, avg_max, _, _, _, _, _ = SM().get_tensor_stats(stat_id, kind_avg)

            if avg_min == 0 or avg_max == 0:
                # Do not clip not symetrical activations
                return tensor

            k1 = max_max / avg_max if avg_max != 0 else self.rho * max_max
            k2 = min_min / avg_min if avg_min != 0 else self.rho * min_min
            max_ = self.rho * max_max / k1 if k1 != 0 else max_max
            min_ = self.rho * min_min / k2 if k2 != 0 else min_min
            if (max_ > 0 and min_ < 0):
                # clip symetrical range only
                maxabs = torch.max(torch.abs(tensor.max()), torch.abs(tensor.min()))
                upper_bound = maxabs * self.rho
                lower_bound = -maxabs * self.rho
                if inplace:
                    tensor.clamp_(lower_bound, upper_bound)
                else:
                    tensor = torch.clamp(tensor, lower_bound, upper_bound)

        return tensor


class RatioClipper(Function):
    def __init__(self, rho):
        self.rho = rho

    def __call__(self, tensor, tag="", inplace=False):
        max_ = tensor.max()
        min_ = tensor.min()
        if (max_ > 0 and min_ < 0):
            # clip symetrical range only
            maxabs = torch.max(torch.abs(tensor.max()), torch.abs(tensor.min()))
            upper_bound = maxabs * self.rho
            lower_bound = -maxabs * self.rho
            if inplace:
                tensor.clamp_(lower_bound, upper_bound)
            else:
                tensor = torch.clamp(tensor, lower_bound, upper_bound)

        return tensor
