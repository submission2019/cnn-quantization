import torch
import torch.nn as nn
import torchvision
from pytorch_quantizer.quantization import qtypes
from utils.misc import Singleton
from utils import attacher
from utils.monitor import Monitor
from .statistic_manager import StatisticManager
from .statistic_manager_perchannel import StatisticManagerPerChannel
from .distance_stats import MeasureStatistics as MS
# from .measure_statistics import MeasureStatistics as MS
from pytorch_quantizer.quantization.quantization_manager import QuantizationManagerBase
from enum import Enum
from itertools import count
import os
import numpy as np
from utils.dump_manager import DumpManager as DM
from pytorch_quantizer.clipping.clipping_manager import StatisticalClipper, RatioClipper
from pytorch_quantizer.quantization.qtypes.dummy_quantizer import DummyQuantizer

VERBOSE = True

class StatsMode(Enum):
    no_stats = 1
    collect_stats = 2
    use_stats = 3

class ReLUWithId(nn.ReLU):
    _id = count(0)
    def __init__(self, inplace=False):
        super(ReLUWithId, self).__init__(inplace)

    def forward(self, input):
        out = super(ReLUWithId, self).forward(input)
        # id = next(self._id)
        # out_id = 'relu%d_activation' % id
        # if QMI().enabled:
        #     if QMI().stats_mode is StatsMode.collect_stats:
        #         QMI().stats_manager.save_tensor_stats(out, out_id)
        #     # TODO: enable quantization of relu after quantization map is ready
        #     elif QMI().stats_mode is StatsMode.use_stats:
        #         # Quantize using statistics
        #         out = QMI().quantize_instant(out, "activation_relu", stat_id=out_id)
        #     else:
        #         # No stats, quantize using actual values
        #         out = QMI().quantize_instant(out, "activation_relu")

        return out

class MaxPool2dWithId(nn.MaxPool2d):
    _id = count(0)
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPool2dWithId, self).__init__(kernel_size, stride, padding, dilation,
                 return_indices, ceil_mode)
        self.id = next(self._id)

    def forward(self, input):
        out = super(MaxPool2dWithId, self).forward(input)
        out_id = 'maxpool%d_out' % self.id

        # Uncomment to enable dump
        # torch.save(out, os.path.join('dump', out_id + '.pt'))
        if QMI().enabled:
            if QMI().stats_mode is StatsMode.collect_stats:
                QMI().stats_manager.save_tensor_stats(out, 'activation_pooling', out_id)
            elif QMI().stats_mode is StatsMode.use_stats:
                # Quantize using statistics
                out = QMI().quantize_instant(out, out_id, "activation_pooling", stat_id=out_id, verbose=QMI().verbose)
            else:
                # No stats, quantize using actual values
                out = QMI().quantize_instant(out, out_id, "activation_pooling", verbose=QMI().verbose)

        return out

class AvgPool2dWithId(nn.AvgPool2d):
    _id = count(0)
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(AvgPool2dWithId, self).__init__(kernel_size, stride, padding, ceil_mode,
                                              count_include_pad)
        self.id = next(self._id)

    def forward(self, input):
        out = super(AvgPool2dWithId, self).forward(input)
        out_id = 'avgpool%d_out' % self.id
        tag_act = 'activation_classifier' if out.shape[1] == 1000 else 'activation_pooling'

        # Uncomment to enable dump
        # torch.save(out, os.path.join('dump', out_id + '.pt'))
        if QMI().enabled:
            if QMI().stats_mode is StatsMode.collect_stats:
                QMI().stats_manager.save_tensor_stats(out, tag_act, out_id)
            elif QMI().stats_mode is StatsMode.use_stats:
                # Quantize using statistics
                out = QMI().quantize_instant(out, tag_act, stat_id=out_id, verbose=QMI().verbose)
            else:
                # No stats, quantize using actual values
                out = QMI().quantize_instant(out, tag_act, verbose=QMI().verbose)

        return out


class DeviceCache:
    def __init__(self):
        self.devices_cache = {}

    def store(self, tid, t):
        if tid in self:
            return

        # Store t in cache
        if t.device not in self.devices_cache:
            self.devices_cache[t.device] = {}

        self.devices_cache[t.device][tid] = t

    def get(self, tid, device):
        # If tid not cached on any device None will be returned
        t = None

        # Check if tensor with this id already stored on this device
        if device in self.devices_cache and tid in self.devices_cache[device]:
            t = self.devices_cache[device][tid]
        else:
            # Otherwise search on other devices and copy to current if exist
            for dev in self.devices_cache:
                if tid in self.devices_cache[dev]:
                    t = self.devices_cache[dev][tid]
                    break

            if t is not None:
                # t was found, add it to the cache on this device
                if device not in self.devices_cache:
                    self.devices_cache[device] = {}
                self.devices_cache[device][tid] = t.to(device)
                t = self.devices_cache[device][tid]

        return t

    def __contains__(self, tid):
        for dev in self.devices_cache:
            if tid in self.devices_cache[dev]:
                return True

        return False


bias_corr_cache = DeviceCache()


class Conv2dWithId(nn.Conv2d):
    _id = count(0)
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dWithId, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.id = next(self._id)
        self.eps = torch.tensor([1e-8])
        # print('conv_%d' % self.id)

    def forward(self, input):
        activation_id = 'conv%d_activation' % self.id

        if not QMI().enabled:
            out = super(Conv2dWithId, self).forward(input)
            # Uncomment to enable dump
            # torch.save(out, os.path.join('dump', activation_id + '.pt'))
        else:
            out = super(Conv2dWithId, self).forward(input)
            tag_act = 'activation_classifier' if out.shape[1] == 1000 else 'activation'

            if QMI().stats_mode is StatsMode.collect_stats:
                QMI().stats_manager.save_tensor_stats(out, self.internal_name, activation_id)
            elif QMI().stats_mode is StatsMode.use_stats:
                # Quantize using statistics
                out_q = QMI().quantize_instant(out, activation_id, tag_act, stat_id=activation_id,
                                               half_range=hasattr(self, 'before_relu'), verbose=QMI().verbose)
                # print("%s: %d" % (activation_id, out.shape[2]*out.shape[3]))
                if QMI().bcorr_act:
                    # if activation_id in bias_corr_cache:
                    #     q_bias = bias_corr_cache.get(activation_id, out_q.device)
                    # else:
                    #     # TODO: cover case of not per channel quantization
                    #     q_bias = out.transpose(0, 1).contiguous().view(out.shape[1], -1).mean(-1) - out_q.transpose(0, 1).contiguous().view(out_q.shape[1], -1).mean(-1)
                    #     bias_corr_cache.store(activation_id, q_bias)

                    if hasattr(self, 'before_relu') or QMI().op_manager.fused_relu:
                        out = torch.nn.functional.relu(out)

                    temp = out.transpose(0, 1).contiguous().view(out.shape[1], -1)
                    q_bias = temp.sum(-1) - out_q.transpose(0, 1).contiguous().view(out_q.shape[1], -1).sum(-1)
                    count = (temp > 0).sum(-1).type(q_bias.dtype)
                    q_bias /= (count + self.eps.to(q_bias.device))

                    out_q += (out_q > 0).type(out_q.dtype) * q_bias.view(1, q_bias.numel(), 1, 1)

                    # norm_corr = torch.sqrt(torch.norm(out.transpose(0, 1).contiguous().view(out.shape[1], -1), dim=-1) / \
                    #             (torch.norm(out_q.transpose(0, 1).contiguous().view(out_q.shape[1], -1), dim=-1) + self.eps.to(out_q.device)))
                    # norm_corr = torch.sqrt(torch.norm(out) / torch.norm(out_q))
                    # out_q = out_q * norm_corr#.view(1, norm_corr.numel(), 1, 1)

                out = out_q

            else:
                # No stats, quantize using actual values
                out = QMI().quantize_instant(out, activation_id, tag_act, half_range=hasattr(self, 'before_relu'), verbose=QMI().verbose)

        if QMI().measure_stats.enabled:
            QMI().measure_stats.save_measure(out, activation_id)

        # Uncomment for debug
        # t = out.transpose(0, 1).contiguous().view(out.shape[1], -1)  # C x N x H x W
        # print(np.max([torch.unique(t[i]).numel() for i in range(t.shape[0])]))
        # print(torch.unique(out).numel())

        return out


class LinearWithId(nn.Linear):
    _id = count(0)

    def __init__(self, in_features, out_features, bias=True):
        super(LinearWithId, self).__init__(in_features, out_features, bias)
        self.id = next(self._id)

    def forward(self, input):
        activation_id = 'linear%d_activation' % self.id
        if not QMI().enabled:
            out = super(LinearWithId, self).forward(input)
        else:
            tag_act = 'activation_classifier' if self.weight.shape[0] == 1000 else 'activation_linear'
            half_range = hasattr(self, 'before_relu') if self.weight.shape[0] != 1000 else False
            out = super(LinearWithId, self).forward(input)

            if QMI().stats_mode is StatsMode.collect_stats:
                QMI().stats_manager.save_tensor_stats(out, tag_act, activation_id, force_global_min_max=('classifier' in tag_act))
            elif QMI().stats_mode is StatsMode.use_stats:
                out_q = QMI().quantize_instant(out, activation_id, tag_act, stat_id=activation_id, half_range=half_range,
                                               verbose=QMI().verbose)

                out = out_q

            else:
                out = QMI().quantize_instant(out, activation_id, tag_act, half_range=half_range, verbose=QMI().verbose)

        if QMI().measure_stats.enabled:
            QMI().measure_stats.save_measure(out, activation_id)

        return out


# TODO: batch norm folding
class BatchNorm2dWithId(nn.BatchNorm2d):
    _id = count(0)
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm2dWithId, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.id = next(self._id)
        # print('bn_%d' % self.id)

    def forward(self, input):
        activation_id = 'bn%d_activation' % self.id
        if QMI().bn_folding and hasattr(self, 'absorbed'):
            return input

        if not QMI().enabled:
            out = super(BatchNorm2dWithId, self).forward(input)
        else:
            out = super(BatchNorm2dWithId, self).forward(input)
            if QMI().stats_mode is StatsMode.collect_stats:
                QMI().stats_manager.save_tensor_stats(out, 'activation', activation_id)
            elif QMI().stats_mode is StatsMode.use_stats:
                # Quantize using statistics
                out = QMI().quantize_instant(out, "activation", stat_id=activation_id, half_range=hasattr(self, 'before_relu'), verbose=QMI().verbose)
            else:
                # No stats, quantize using actual values
                out = QMI().quantize_instant(out, "activation", half_range=hasattr(self, 'before_relu'), verbose=QMI().verbose)

        if QMI().measure_stats.enabled:
            QMI().measure_stats.save_measure(out, activation_id)

        return out


class QuantizationManagerInference(QuantizationManagerBase):
    def __init__(self, args, qparams):
        super(QuantizationManagerInference, self).__init__()
        self.args = args
        self.verbose = False
        self.quantize = args.qtype is not None
        self.disable_quantization = args.q_off
        self.op_manager = self.createTruncationManager(args, qparams)
        self.enabled = False
        self.bn_folding = False
        self.bcorr_act = args.bias_corr_act
        self.bcorr_weight = args.bias_corr_weight
        self.vcorr_weight = args.var_corr_weight
        sf = args.stats_folder if args.stats_folder is not None else args.arch
        if args.kld_threshold:
            sf += '_kld_' + args.qtype

        self.stats_manager = None
        if args.stats_mode == 'collect':
            print("Collecting statistics...")
            self.stats_mode = StatsMode.collect_stats
            if args.per_channel_quant_act:
                self.stats_manager = StatisticManagerPerChannel(sf, load_stats=False, batch_avg=args.stats_batch_avg)
            else:
                self.stats_manager = StatisticManager(sf, load_stats=False, kld_threshold=args.kld_threshold, batch_avg=args.stats_batch_avg)
        elif args.stats_mode == 'use':
            self.stats_mode = StatsMode.use_stats
            if args.per_channel_quant_act:
                StatisticManagerPerChannel(sf, load_stats=True)
            StatisticManager(sf, load_stats=True)
        else:
            self.stats_mode = StatsMode.no_stats
            self.stats_manager = None

        self.measure_stats = MS(args.arch)
        if args.measure_stats:
            # enable measuring statistics
            self.measure_stats.__enter__()


    def __exit__(self, *args):
        self.op_manager.__exit__(args)
        if self.stats_manager is not None:
            self.stats_manager.__exit__()
        if self.measure_stats is not None:
            self.measure_stats.__exit__()
        super(QuantizationManagerInference, self).__exit__(args)

    def createTruncationManager(self, args, qparams):
        op_manager = TruncationOpManagerInference(args, qparams)
        if args.qtype == 'int4':
            ignore_ids = [0]
            op_manager.set_8bit_list(['conv%d_activation'%id for id in ignore_ids])

        return op_manager

    def quantize_instant(self, tensor, id, tag="", stat_id=None, half_range=False, override_att=None, verbose=False):
        return self.op_manager.quantize_instant(tensor, id, tag, stat_id, half_range, override_att, verbose)

    def set_8bit_list(self, ignore_ids):
        self.op_manager.set_8bit_list(ignore_ids)

    def reset_counters(self):
        ReLUWithId._id = count(0)
        pass

    def quantize_model(self, model):
        if self.args.stats_mode == 'collect':
            return

        for n, m in model.named_modules():
            weight_q = None
            if isinstance(m, torch.nn.Conv2d):
                # In case of inceptionV3 leave first and second layer at 8 bit
                if isinstance(model, torchvision.models.Inception3) and \
                            (n == 'Conv2d_1a_3x3.conv' or n == 'Conv2d_2a_3x3.conv'):
                    weight_q = QMI().quantize_instant(m.weight, n + '.weight', "weight", override_att=('num_bits', 8), verbose=True)
                elif m.weight.shape[1] == 3:
                    # first layer leave in 8 bit
                    weight_q = QMI().quantize_instant(m.weight, n + '.weight', "weight", override_att=('num_bits', 8), verbose=True)
                else:
                    weight_q = QMI().quantize_instant(m.weight, n + '.weight', "weight", verbose=True)

            elif isinstance(m, torch.nn.Linear):
                tag_weight = 'weight_classifier' if m.weight.shape[0] == 1000 else 'weight'
                weight_q = QMI().quantize_instant(m.weight, n + '.weight', tag_weight, verbose=True)

            if weight_q is not None:
                if self.vcorr_weight or self.bcorr_weight:
                    bias_q = weight_q.view(weight_q.shape[0], -1).mean(-1)
                    bias_q = bias_q.view(bias_q.numel(), 1, 1, 1) if len(weight_q.shape) == 4 else bias_q.view(bias_q.numel(), 1)
                    bias_orig = m.weight.view(m.weight.shape[0], -1).mean(-1)
                    bias_orig = bias_orig.view(bias_orig.numel(), 1, 1, 1) if len(weight_q.shape) == 4 else bias_orig.view(bias_orig.numel(), 1)

                if self.vcorr_weight:
                    eps = torch.tensor([1e-8]).to(weight_q.device)
                    var_corr = m.weight.view(m.weight.shape[0], -1).std(dim=-1) / \
                            (weight_q.view(weight_q.shape[0], -1).std(dim=-1) + eps)
                    var_corr = (var_corr.view(var_corr.numel(), 1, 1, 1) if len(weight_q.shape) == 4 else var_corr.view(var_corr.numel(), 1))

                    # Correct variance
                    weight_q = (weight_q - bias_q) * var_corr + bias_q

                if self.bcorr_weight:
                    # Correct mean
                    weight_q = weight_q - bias_q + bias_orig

                m.weight.data = weight_q


# Alias
QMI = QuantizationManagerInference


class TruncationOpManagerInference:
    def __load_quantizer__(self, qtype, qparams):
        qtype_name = qtype.rstrip('1234567890')
        quant_params = qparams[qtype_name] if qtype_name in qparams else {}
        quantizer = qtypes.__dict__[qtype_name + "_quantizer"](qtype, quant_params)
        return quantizer, quant_params

    def __fill_quantizers__(self, qtype, qparams, arch=None, qweight='int8'):
        classifier_quantizer, _ = self.__load_quantizer__('int8', qparams)
        classifier_quantizer.clipping = 'no'
        classifier_quantizer.kld = False
        classifier_quantizer.pcq_w = False
        classifier_quantizer.pcq_a = False
        classifier_quantizer.sm = StatisticManager
        classifier_quantizer.stats_kind = 'max'
        classifier_quantizer.measure_entropy = False
        self.quantizers['activation_classifier'] = classifier_quantizer

        if qweight == 'f32':
            weights_quantizer = DummyQuantizer()
        else:
            weights_quantizer, _ = self.__load_quantizer__(qweight, qparams)
            weights_quantizer.pcq_a = False
            weights_quantizer.clipping = 'no'
            weights_quantizer.kld = False
            weights_quantizer.bit_alloc = False
            weights_quantizer.stats_kind = 'max'
        self.quantizers['weight'] = weights_quantizer

        weights_quantizer, _ = self.__load_quantizer__('int8', qparams)
        weights_quantizer.pcq_a = False
        weights_quantizer.clipping = 'no'
        weights_quantizer.kld = False
        weights_quantizer.bit_alloc = False
        weights_quantizer.stats_kind = 'max'
        weights_quantizer.measure_entropy = False
        self.quantizers['weight_classifier'] = weights_quantizer

        bias_quantizer, _ = self.__load_quantizer__('int8', qparams)
        bias_quantizer.pcq_w = False
        bias_quantizer.pcq_a = False
        bias_quantizer.clipping = 'no'
        bias_quantizer.kld = False
        bias_quantizer.bit_alloc = False
        # self.quantizers['bias'] = bias_quantizer
        self.quantizers['bias'] = DummyQuantizer()

        quantizer_ignored, _ = self.__load_quantizer__('int8', qparams)
        quantizer_ignored.pcq_w = False
        quantizer_ignored.pcq_a = False
        quantizer_ignored.sm = StatisticManager
        quantizer_ignored.clipping = 'no'
        quantizer_ignored.kld = False
        self.quantizers['ignored'] = quantizer_ignored

        activation_quantizer, _ = self.__load_quantizer__(qtype, qparams)
        activation_quantizer.force_positive = self.fused_relu
        activation_quantizer.pcq_w = False
        self.quantizers['activation'] = activation_quantizer

        activation_linear_quantizer, _ = self.__load_quantizer__(qtype, qparams)
        activation_linear_quantizer.force_positive = self.fused_relu
        activation_linear_quantizer.pcq_w = False
        activation_linear_quantizer.pcq_a = False
        activation_linear_quantizer.sm = StatisticManager
        self.quantizers['activation_linear'] = activation_linear_quantizer

        # Pooling is currently not working well with clipping. Leave it in 8 bit.
        pooling_quantizer, _ = self.__load_quantizer__('int8', qparams)
        pooling_quantizer.pcq_w = False
        pooling_quantizer.pcq_a = False
        pooling_quantizer.sm = StatisticManager
        pooling_quantizer.clipping = 'no'
        pooling_quantizer.kld = False
        pooling_quantizer.bit_alloc = False
        pooling_quantizer.measure_entropy = False
        self.quantizers['activation_pooling'] = pooling_quantizer

    def __init__(self, args, qparams):
        self.verbose = False
        self.activation_quantizer = None
        self.origin_linear = nn.Linear
        self.origin_conv2d = nn.Conv2d
        self.origin_batch_norm = nn.BatchNorm2d
        self.orig_maxpool = nn.MaxPool2d
        self.orig_avgpool = nn.AvgPool2d
        self.orig_relu = nn.ReLU
        self.ignore_ids = []

        self.rho_act = qparams['qmanager']['rho_act'] if 'qmanager' in qparams else None
        self.rho_weight = qparams['qmanager']['rho_weight'] if 'qmanager' in qparams else None
        self.fp32_clip = self.rho_act is not None or self.rho_weight is not None
        self.fused_relu = args.arch is not None and (args.arch == 'alexnet' or args.arch == 'vgg16' or args.arch == 'vgg16_bn' or args.arch == 'inception_v3' or 'squeezenet' in args.arch)

        if args.qtype is not None:
            self.quantizers = {}
            self.quantize = True
            if 'bfloat' in args.qtype:
                self.quantizer_default, _ = self.__load_quantizer__(args.qtype, qparams)
                self.linear_layer_quantizer = DummyQuantizer()
            else:
                self.__fill_quantizers__(args.qtype, qparams, args.arch, args.qweight)
                self.quantizer_default, _ = self.__load_quantizer__('int8', qparams)
            self.activations_clipper = StatisticalClipper(self.rho_act)
            self.weights_clipper = RatioClipper(self.rho_weight)

    def __exit__(self, *args):
        pass

    def get_quantizer(self, tag, tensor=None):
        if tag in self.quantizers:
            return self.quantizers[tag]
        else:
            return self.quantizer_default

    def set_8bit_list(self, ignore_list):
        self.ignore_ids = ignore_list

    def enable(self):
        # self.quantize_matmul()
        nn.Linear = LinearWithId
        nn.Conv2d = Conv2dWithId
        nn.BatchNorm2d = BatchNorm2dWithId
        nn.MaxPool2d = MaxPool2dWithId
        nn.AvgPool2d = AvgPool2dWithId
        nn.ReLU = ReLUWithId

    def disable(self):
        nn.Linear = self.origin_linear
        nn.Conv2d = self.origin_conv2d
        nn.BatchNorm2d = self.origin_batch_norm
        nn.MaxPool2d = self.orig_maxpool
        nn.AvgPool2d = self.orig_avgpool
        nn.ReLU = self.orig_relu

    # quantizes origin matmul
    def quantize_matmul(self):
        def quantized_matmul(tensor1, tensor2):
            tensor1_ = attacher.pytorch_attach(tensor1, self.activation_quantizer, None)
            tensor2_ = attacher.pytorch_attach(tensor2, self.activation_quantizer, None)
            res = self.origin_matmul(tensor1_, tensor2_)
            return attacher.pytorch_attach(res, self.activation_quantizer, None)

        torch.Tensor.matmul = quantized_matmul

    def quantize_tensor(self, tensor, fprop=True, bprop=True):
        fprop = self.activation_quantizer if fprop else None
        return attacher.pytorch_attach(tensor, fprop, None)

    def quantize_instant(self, tensor, id, tag="", stat_id=None, half_range=False, override_att=None, verbose=False):
        # ignore quantization of first and last layer
        ignore_cond = False
        if stat_id is not None:
            ignore_cond = np.array([l == stat_id for l in self.ignore_ids]).any()

        qtag = 'ignored' if ignore_cond else tag
        q = self.get_quantizer(qtag)
        q.half_range = half_range

        if verbose:
            print("Quantize {0:21} | Id - {1:18} | {2:} | {3:}".format(tag, str(stat_id), str(q), str(tensor.device)))

        return q(tensor, id, tag, stat_id, override_att)
