import torch
import torch.nn as nn
from pytorch_quantizer.quantization import qtypes
from utils.misc import Singleton
from utils import attacher
from utils.monitor import Monitor
from .statistic_manager import StatisticManager
from .statistic_manager_perchannel import StatisticManagerPerChannel
from .measure_statistics import MeasureStatistics as MS
from pytorch_quantizer.quantization.quantization_manager import QuantizationManagerBase
from enum import Enum
from itertools import count
import os
import numpy as np
from pytorch_quantizer.quantization.qtypes.dummy_quantizer import DummyQuantizer


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
                out = QMI().quantize_instant(out, "activation_pooling", stat_id=out_id)
            else:
                # No stats, quantize using actual values
                out = QMI().quantize_instant(out, "activation_pooling")

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
                out = QMI().quantize_instant(out, tag_act, stat_id=out_id)
            else:
                # No stats, quantize using actual values
                out = QMI().quantize_instant(out, tag_act)

        return out

class Conv2dWithId(nn.Conv2d):
    _id = count(0)
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dWithId, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.id = next(self._id)
        # print('conv_%d' % self.id)

    def forward(self, input):
        activation_id = 'conv%d_activation' % self.id

        if not QMI().enabled:
            out = super(Conv2dWithId, self).forward(input)
            # Uncomment to enable dump
            # torch.save(out, os.path.join('dump', activation_id + '.pt'))
        else:
            orig_weight = self.weight.data
            if self.bias is not None:
                orig_bias = self.bias.data
            if QMI().stats_mode is not StatsMode.collect_stats:
                if input.shape[1] == 3:
                    # first layer leave in 8 bit
                    self.weight.data = QMI().quantize_instant(self.weight, "weight", override_att=('num_bits', 8))
                else:
                    self.weight.data = QMI().quantize_instant(self.weight, "weight")
                if self.bias is not None:
                    self.bias.data = QMI().quantize_instant(self.bias, "bias")

            out = super(Conv2dWithId, self).forward(input)
            tag_act = 'activation_classifier' if out.shape[1] == 1000 else 'activation'
            self.weight.data = orig_weight
            if self.bias is not None:
                self.bias.data = orig_bias

            if QMI().stats_mode is StatsMode.collect_stats:
                out_lowp = QMI().quantize_instant(out, tag_act, half_range=hasattr(self, 'before_relu'), override_att=('clipping', 'no'))
                out_gaus = QMI().quantize_instant(out, tag_act, half_range=hasattr(self, 'before_relu'), override_att=('clipping', 'gaus'))
                out_laplace = QMI().quantize_instant(out, tag_act, half_range=hasattr(self, 'before_relu'), override_att=('clipping', 'laplace'))

                q = QMI().op_manager.get_quantizer(tag_act)
                if hasattr(self, 'before_relu') or q.force_positive:  # Currently applicable only for resnet
                    out_pos = torch.clamp(out, 0, out.max())
                else:
                    out_pos = out
                QMI().stats_manager.save_tensor_stats(out, self.internal_name, activation_id, tensors_q={'orig': out_pos, 'lowp': out_lowp, 'gaus': out_gaus, 'laplace': out_laplace})
            elif QMI().stats_mode is StatsMode.use_stats:
                # Quantize using statistics
                out = QMI().quantize_instant(out, tag_act, stat_id=activation_id, half_range=hasattr(self, 'before_relu'))
            else:
                # No stats, quantize using actual values
                out = QMI().quantize_instant(out, tag_act, half_range=hasattr(self, 'before_relu'))

            if MS().enabled:
                out_orig = super(Conv2dWithId, self).forward(input)
                if self.bias is not None:
                    out_orig = out_orig - self.bias.data.view(1,self.bias.shape[0],1,1)
                    out_nb = out - self.bias.data.view(1,self.bias.shape[0],1,1)
                    MS().save_measure(out_orig, out_nb, input, self.weight.data, activation_id)
                else:
                    MS().save_measure(out_orig, out, input, self.weight.data, activation_id)

        return out


class LinearWithId(nn.Linear):
    _id = count(0)
    def __init__(self, in_features, out_features, bias=True):
        super(LinearWithId, self).__init__(in_features, out_features, bias)
        self.id = next(self._id)

    def forward(self, input):
        if not QMI().enabled:
            return super(LinearWithId, self).forward(input)
        else:
            tag_act = 'activation_classifier' if self.weight.shape[0] == 1000 else 'activation_linear'
            tag_weight = 'weight_classifier' if self.weight.shape[0] == 1000 else 'weight'
            half_range = hasattr(self, 'before_relu') if self.weight.shape[0] != 1000 else False
            orig_weight = self.weight.data
            if self.bias is not None:
                orig_bias = self.bias.data
            if QMI().stats_mode is not StatsMode.collect_stats:
                self.weight.data = QMI().quantize_instant(self.weight, tag_weight)
                if self.bias is not None:
                    self.bias.data = QMI().quantize_instant(self.bias, "bias")
            out = super(LinearWithId, self).forward(input)
            self.weight.data = orig_weight
            if self.bias is not None:
                self.bias.data = orig_bias

            activation_id = 'linear%d_activation' % self.id
            if QMI().stats_mode is StatsMode.collect_stats:
                QMI().stats_manager.save_tensor_stats(out, tag_act, activation_id, global_min_max=('classifier' in tag_act))
            elif QMI().stats_mode is StatsMode.use_stats:
                out = QMI().quantize_instant(out, tag_act, stat_id=activation_id, half_range=half_range)
            else:
                out = QMI().quantize_instant(out, tag_act, half_range=half_range)

            if MS().enabled:
                out_orig = super(LinearWithId, self).forward(input)
                MS().save_measure(out_orig, out, input, self.weight.data, activation_id)

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
        if not QMI().enabled:
            return super(BatchNorm2dWithId, self).forward(input)
        else:
            if QMI().bn_folding and hasattr(self, 'absorbed'):
                return input
            # else:
            #     # Do regular BN if floding is set
            #     return super(BatchNorm2dWithId, self).forward(input)

            orig_weight = self.weight.data
            if self.bias is not None:
                orig_bias = self.bias.data
            if QMI().stats_mode is not StatsMode.collect_stats:
                self.weight.data = QMI().quantize_instant(self.weight, "weight")
                if self.bias is not None:
                    self.bias.data = QMI().quantize_instant(self.bias, "bias")

            out = super(BatchNorm2dWithId, self).forward(input)
            self.weight.data = orig_weight
            if self.bias is not None:
                self.bias.data = orig_bias

            activation_id = 'bn%d_activation' % self.id
            if QMI().stats_mode is StatsMode.collect_stats:
                QMI().stats_manager.save_tensor_stats(out, 'activation', activation_id)
            elif QMI().stats_mode is StatsMode.use_stats:
                # Quantize using statistics
                out = QMI().quantize_instant(out, "activation", stat_id=activation_id, half_range=hasattr(self, 'before_relu'))
            else:
                # No stats, quantize using actual values
                out = QMI().quantize_instant(out, "activation", half_range=hasattr(self, 'before_relu'))

            if MS().enabled:
                out_orig = super(BatchNorm2dWithId, self).forward(input)
                if self.bias is not None:
                    out_orig = out_orig - self.bias.data.view(1,self.bias.shape[0],1,1)
                    out_nb = out - self.bias.data.view(1,self.bias.shape[0],1,1)
                    MS().save_measure(out_orig, out_nb, input, self.weight.data, activation_id)
                else:
                    MS().save_measure(out_orig, out, input, self.weight.data, activation_id)

            return out


class QuantizationManagerInference(QuantizationManagerBase):
    def __init__(self, args, qparams):
        super(QuantizationManagerInference, self).__init__()
        self.quantize = args.qtype is not None
        self.disable_quantization = args.q_off
        self.op_manager = self.createTruncationManager(args, qparams)
        self.enabled = False
        self.bn_folding = False
        sf = args.stats_folder if args.stats_folder is not None else args.arch
        if args.kld_threshold:
            sf += '_kld_' + args.qtype

        self.stats_manager = None
        if args.stats_mode == 'collect':
            self.stats_mode = StatsMode.collect_stats
            if args.per_channel_quant_act:
                self.stats_manager = StatisticManagerPerChannel(sf, load_stats=False)
            else:
                self.stats_manager = StatisticManager(sf, load_stats=False, kld_threshold=args.kld_threshold)
        elif args.stats_mode == 'use':
            self.stats_mode = StatsMode.use_stats
            if args.per_channel_quant_act:
                StatisticManagerPerChannel(sf, load_stats=True)
            StatisticManager(sf, load_stats=True)
        else:
            self.stats_mode = StatsMode.no_stats
            self.stats_manager = None

    def __exit__(self, *args):
        self.op_manager.__exit__(args)
        if self.stats_manager is not None:
            self.stats_manager.__exit__()
        super(QuantizationManagerInference, self).__exit__(args)

    def createTruncationManager(self, args, qparams):
        op_manager = TruncationOpManagerInference(args, qparams)
        if args.qtype == 'int4':
            ignore_ids = [0]
            op_manager.set_8bit_list(['conv%d_activation'%id for id in ignore_ids])

        return op_manager

    def quantize_instant(self, tensor, tag="", stat_id=None, half_range=False, override_att=None):
        return self.op_manager.quantize_instant(tensor, tag, stat_id, half_range, override_att)

    def set_8bit_list(self, ignore_ids):
        self.op_manager.set_8bit_list(ignore_ids)

    def reset_counters(self):
        ReLUWithId._id = count(0)
        pass


# Alias
QMI = QuantizationManagerInference


class TruncationOpManagerInference:
    def __load_quantizer__(self, qtype, qparams):
        qtype_name = qtype.rstrip('1234567890')
        quant_params = qparams[qtype_name] if qtype_name in qparams else {}
        quantizer = qtypes.__dict__[qtype_name + "_quantizer"](qtype, quant_params)
        return quantizer, quant_params

    def __fill_quantizers__(self, qtype, qparams, arch=None, qweight='int8'):
        fused_relu = arch is not None and (arch == 'alexnet' or arch == 'vgg16' or arch == 'vgg16_bn' or arch == 'inception_v3' or 'squeezenet' in arch)

        classifier_quantizer, _ = self.__load_quantizer__('int8', qparams)
        classifier_quantizer.clipping = 'no'
        classifier_quantizer.kld = False
        classifier_quantizer.pcq_w = False
        classifier_quantizer.pcq_a = False
        classifier_quantizer.sm = StatisticManager
        classifier_quantizer.stats_kind = 'max'
        self.quantizers['activation_classifier'] = classifier_quantizer

        if qweight == 'f32':
            weights_quantizer = DummyQuantizer()
        else:
            weights_quantizer, _ = self.__load_quantizer__(qweight, qparams)
            weights_quantizer.pcq_a = False
            weights_quantizer.clipping = 'no'
            weights_quantizer.kld = False
        self.quantizers['weight'] = weights_quantizer

        weights_quantizer, _ = self.__load_quantizer__('int8', qparams)
        weights_quantizer.pcq_a = False
        weights_quantizer.clipping = 'no'
        weights_quantizer.kld = False
        self.quantizers['weight_classifier'] = weights_quantizer

        bias_quantizer, _ = self.__load_quantizer__('int8', qparams)
        bias_quantizer.pcq_w = False
        bias_quantizer.pcq_a = False
        bias_quantizer.clipping = 'no'
        bias_quantizer.kld = False
        self.quantizers['bias'] = bias_quantizer
        # self.quantizers['bias'] = DummyQuantizer()

        quantizer_ignored, _ = self.__load_quantizer__('int8', qparams)
        quantizer_ignored.pcq_w = False
        quantizer_ignored.pcq_a = False
        quantizer_ignored.sm = StatisticManager
        quantizer_ignored.clipping = 'no'
        quantizer_ignored.kld = False
        self.quantizers['ignored'] = quantizer_ignored

        activation_quantizer, _ = self.__load_quantizer__(qtype, qparams)
        activation_quantizer.force_positive = fused_relu
        activation_quantizer.pcq_w = False
        self.quantizers['activation'] = activation_quantizer

        activation_linear_quantizer, _ = self.__load_quantizer__(qtype, qparams)
        activation_linear_quantizer.force_positive = fused_relu
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

        if args.qtype is not None:
            self.quantizers = {}
            self.quantize = True
            if 'bfloat' in args.qtype:
                self.quantizer_default, _ = self.__load_quantizer__(args.qtype, qparams)
                self.linear_layer_quantizer = DummyQuantizer()
            else:
                self.__fill_quantizers__(args.qtype, qparams, args.arch, args.qweight)
                self.quantizer_default, _ = self.__load_quantizer__('int8', qparams)

        if args.measure_stats:
            self.measure_stats = MS()
            self.measure_stats.folder = 'results/measure_stats/%s' % args.arch if args.measure_stats_folder is None else args.measure_stats_folder
            if self.fp32_clip:
                if self.rho_act is not None:
                    self.measure_stats.subfolder = 'act_clipping_rho_%f' % self.rho_act
                if self.rho_weight is not None:
                    self.measure_stats.subfolder = 'weights_clipping_rho_%f' % self.rho_weight
            else:
                self.measure_stats.subfolder = args.qtype
            self.measure_stats.__enter__()
        else:
            self.measure_stats = None

    def __exit__(self, *args):
        if self.measure_stats is not None:
            self.measure_stats.__exit__()

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

    def quantize_instant(self, tensor, tag="", stat_id=None, half_range=False, override_att=None):
        # ignore quantization of first and last layer
        ignore_cond = False
        if stat_id is not None:
            ignore_cond = np.array([l == stat_id for l in self.ignore_ids]).any()

        qtag = 'ignored' if ignore_cond else tag
        q = self.get_quantizer(qtag)
        q.half_range = half_range
        return q(tensor, tag, stat_id, override_att)
