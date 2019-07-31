import torch
from abc import ABC, abstractmethod
import numpy as np
from utils.misc import Singleton
import os
import shutil
import uuid


def patch_call(instance, func):
    class _(type(instance)):
        def __call__(self, *arg, **kwarg):
            return func(*arg, **kwarg)
    instance.__class__ = _



class Monitor(metaclass=Singleton):
    def __init__(self, dump_dir=None):
        if dump_dir is None:
            raise Exception('dump_dir must be provided')

        self.dump_dir = dump_dir
        if os.path.exists(dump_dir):
            shutil.rmtree(dump_dir)
        os.makedirs(dump_dir)

        self.observed_tensors = dict()
        self.observed_operations = dict()

    def register_tensor(self, tensor, key, retain_grad=False):
        if retain_grad:
            tensor.retain_grad()
        self.observed_tensors[key] = tensor

    def dump_tensors(self, epoch, step):
        grad_keys = []
        for key in self.observed_tensors.keys():
            tensor = self.observed_tensors[key]
            if tensor.grad is not None:
                grad_keys.append(key)
        for key in grad_keys:
            self.observed_tensors[key + '_grad'] = self.observed_tensors[key].grad
            self.observed_tensors[key] = self.observed_tensors[key].detach()
        for key in self.observed_tensors.keys():
            self.observed_tensors[key] = self.observed_tensors[key].cpu()
        fname = 'epoch_' + str(epoch) + '_step_' + str(step) + '.pt'
        torch.save(self.observed_tensors, os.path.join(self.dump_dir, fname))
        self.observed_tensors.clear()

    def clear_tensors(self):
        self.observed_tensors.clear()

    def register_operation(self, operation, key):
        self.observed_operations[key] = operation

    def dump_operations(self, epoch, step):
        for op_key in self.observed_operations.keys():
            grad_keys = []
            operation = self.observed_operations[op_key]
            for key in operation.keys():
                if isinstance(operation[key], torch.Tensor):
                    tensor = operation[key]
                    if tensor.grad is not None:
                        grad_keys.append(key)
            for key in grad_keys:
                operation[key + '_grad'] = operation[key].grad
                operation[key] = operation[key].detach()
            for key in operation.keys():
                if isinstance(operation[key], torch.Tensor):
                    operation[key] = operation[key].cpu()

        fname = 'epoch_' + str(epoch) + '_step_' + str(step) + '.pt'
        torch.save(self.observed_operations, os.path.join(self.dump_dir, fname))
        self.observed_operations.clear()

    def clear_operations(self):
        self.observed_operations.clear()

    def register_Conv2d(self, Conv2d, retain_grad=False):
        Conv2d_dict = self.observed_operations[id(Conv2d)] = dict()
        Conv2d_dict['in_channels'] = Conv2d.in_channels
        Conv2d_dict['out_channels'] = Conv2d.out_channels
        Conv2d_dict['kernel_size'] = Conv2d.kernel_size
        Conv2d_dict['stride'] = Conv2d.stride
        Conv2d_dict['padding'] = Conv2d.padding
        Conv2d_dict['dilation'] = Conv2d.dilation
        Conv2d_dict['groups'] = Conv2d.groups
        if Conv2d.bias:
            Conv2d_dict['bias'] = Conv2d.bias
        Conv2d_dict['weight'] = Conv2d.weight
        __call__ = Conv2d.__call__

        def call_warpper(input):
            Conv2d_dict['input'] = input
            if retain_grad:
                input.retain_grad()

            output = __call__(input)

            Conv2d_dict['output'] = output
            if retain_grad:
                output.retain_grad()

            return output

        patch_call(Conv2d, call_warpper)
