import torch
from torch.autograd import Function
import numpy as np
import int_quantization
import math
from utils.monitor import Monitor
from pytorch_quantizer.quantization.inference.statistic_manager import StatisticManager
from pytorch_quantizer.quantization.inference.statistic_manager_perchannel import StatisticManagerPerChannel
from utils.entropy import shannon_entropy, most_requent_value_compression
import scipy.optimize as opt

# Alpha coeficients for for gaussian clipping
# [1.71063519 2.15159277 2.55913646 2.93620062 3.28691474 3.6151146 3.92403714]

# Alpha coeficients for for laplace clipping
# [2.83068299 3.89722946 5.02864014 6.20476633 7.41312622 8.64561995 9.89675982]

# Alpha coeficients for for exponential clipping
# [3.89722946  5.02864014  6.20476633  7.41312622  8.64561995  9.89675982 11.16268502]

def to_cuda(t, device):
    if isinstance(t, torch.Tensor):
        return t.to(device)
    else:
        return torch.tensor(t, dtype=torch.float32).to(device)

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    else:
        return tensor


def laplace_prior_mse(b, alpha, omega):
    return 2 * (b ** 2) * np.exp(-alpha / b) + (alpha ** 2 / (3 * omega**2))

def half_laplace_prior_mse(b, alpha, omega):
    return (b ** 2) * np.exp(-alpha / b) + (alpha ** 2 / (24 * omega**2))

# Numpy code to find optimal alpha for real omega
resolution = 20
omega_table = np.concatenate([np.linspace(0.01, 0.1, resolution, endpoint=False),
                             np.linspace(0.1, 1, resolution, endpoint=False),
                             np.linspace(1, 10, resolution, endpoint=False),
                             np.linspace(10, 100, resolution, endpoint=False),
                             np.linspace(100, 1000, resolution, endpoint=False)])

alpha_table = np.array([opt.minimize_scalar(lambda x: laplace_prior_mse(b=1, alpha=x, omega=w)).x for w in omega_table])
alpha_table = np.concatenate([[0], alpha_table])

omega_table = np.concatenate([[0], omega_table])

count = 0


class IntQuantizer(Function):
    def __init__(self, size, params):
        self.num_bits = size
        # TODO: expose as cmd line parameters
        self.stochastic = False
        self.int_exp = False
        self.enforce_true_zero = True #params['true_zero']
        self.clipping = params['clipping'] if 'clipping' in params else 'no'
        self.stats_kind = params['stats_kind'] if 'stats_kind' in params else 'mean'
        self.kld = params['kld'] if 'kld' in params else False
        self.pcq_w = params['pcq_weights']
        self.pcq_a = params['pcq_act']
        self.bit_alloc_act = params['bit_alloc_act']
        self.bit_alloc_weight = params['bit_alloc_weight']
        self.bcorr_act = params['bcorr_act']
        self.bcorr_weight = params['bcorr_weight']
        self.vcorr_weight = params['vcorr_weight']
        self.bit_alloc_round = params['bit_alloc_rmode'] == 'round'
        self.bit_alloc_prior = params['bit_alloc_prior']
        self.bit_alloc_target_act = params['bit_alloc_target_act'] if params['bit_alloc_target_act'] is not None else self.num_bits
        self.bit_alloc_target_weight = params['bit_alloc_target_weight'] if params['bit_alloc_target_weight'] is not None else self.num_bits
        self.measure_entropy = params['measure_entropy']
        self.logger = params['logger']
        self.mtd_quant = params['mtd_quant']

        self.alpha_gaus = {1: 1.24, 2: 1.71, 3: 2.15, 4: 2.55, 5: 2.93, 6: 3.28, 7: 3.61, 8: 3.92}
        self.alpha_gaus_positive = {1: 1.71, 2: 2.15, 3: 2.55, 4: 2.93, 5: 3.28, 6: 3.61, 7: 3.92, 8: 4.2}

        self.alpha_laplace = {0: 1.05, 1: 1.86, 2: 2.83, 3: 3.89, 4: 5.03, 5: 6.2, 6: 7.41, 7: 8.64, 8: 9.89}
        self.alpha_laplace_positive = {0: 1.86, 1: 2.83, 2: 3.89, 3: 5.02, 4: 6.2, 5: 7.41, 6: 8.64, 7: 9.89, 8: 11.16}

        self.gaussian_const = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) ** 0.5)
        self.sm = StatisticManagerPerChannel if params['pcq_act'] else StatisticManager
        self.force_positive = False
        self.half_range = False

    def __call__(self, tensor, id, tag="", stat_id=None, override_att=None):
        if override_att is not None:
            orig_att = getattr(self, override_att[0])
            setattr(self, override_att[0], override_att[1])
        if self.kld:
            res = self.gemmlowpKldQuantize(tensor, tag, stat_id=stat_id)
        elif self.clipping != 'no':
            # print("clipping %s: %d" % (tag, self.num_bits))
            if self.mtd_quant:
                res = self.mid_tread_quantize_activation(tensor, id)
            else:
                res = self.gemmlowpClippingQuantize(tensor, id, tag, stat_id=stat_id, clip_type=self.clipping)
        elif self.pcq_w:
            # print("pcq_w %s: %d" % (tag, self.num_bits))
            if self.mtd_quant:
                res = self.mid_tread_quantize_weights_per_channel(tensor, id)
            else:
                res = self.gemmlowpQuantizeWeightsPerChannel(tensor, id)
        elif self.pcq_a and len(tensor.shape) > 3 and (tensor.shape[2] > 1 or tensor.shape[3] > 1):
            # print("pcq_a %s: %d" % (tag, self.num_bits))
            if self.mtd_quant:
                res = self.mid_tread_quantize_activation_per_channel(tensor, id)
            else:
                res = self.gemmlowpQuantizeActivationPerChannel(tensor, id, tag, stat_id=stat_id)
        else:
            # print("no clipping %s: %d" % (tag, self.num_bits))
            res = self.gemmlowpMinMaxQuantize(tensor, tag, stat_id=stat_id)

        if override_att is not None:
            setattr(self, override_att[0], orig_att)
        return res

    def __repr__(self):
        return 'IntQuantizer - [bits: {}, clipping: {}, bit_alloc_act: {}, bit_alloc_weight: {}, bit_alloc_round: {}, pcq_w: {}, pcq_a: {}, bcorr_act: {}, bcorr_weight: {}, vcorr_weight: {}, kind: {}]'\
            .format(self.num_bits, self.clipping, self.bit_alloc_act, self.bit_alloc_weight, self.bit_alloc_round, self.pcq_w, self.pcq_a, self.bcorr_act, self.bcorr_weight, self.vcorr_weight, self.stats_kind)

    @staticmethod
    def get_omega(sigma, target_bins):
        B = len(sigma) * target_bins

        # Calculate bit allocation
        p = sigma ** (2./3)
        omega = (B * p) / p.sum()
        return omega

    @staticmethod
    def get_alpha_mult(omega, sym=True):
        omega = omega.cpu().numpy()
        if not sym:
            omega *= 2
        i = omega_table.searchsorted(omega)
        inc = (alpha_table[i] - alpha_table[i - 1]) / (omega_table[i] - omega_table[i - 1])
        alpha = alpha_table[i] - inc * (omega_table[i] - omega)
        return alpha

    def mid_tread_quantize_weights_per_channel(self, tensor, id):
        # Assume weights with dimensions [OFM,IFM,K1,K2]
        t = tensor.view(tensor.shape[0], -1)

        tq, entropy = self.mid_tread_quantization(t, id, self.bit_alloc_target_weight, clip=False, sym=True)
        if entropy is not None and self.logger is not None:
            self.logger.log_metric(id + '.entropy', entropy.item(), step='auto', meterId='avg.entropy.weight',
                                   weight=tq.numel())

        return tq.view(tensor.shape)

    def mid_tread_quantize_activation(self, tensor, id):
        # Assume [N, C, H, W] or [N, M]
        if self.pcq_a and len(tensor.shape) > 3 and (tensor.shape[2] > 1 or tensor.shape[3] > 1):
            # scale per channel
            out = self.mid_tread_quantize_activation_per_channel(tensor, id)
        else:
            # scale per tensor
            symmetric = not (self.force_positive or self.half_range)
            out, entropy = self.mid_tread_quantization(tensor.view(1, -1), id, self.bit_alloc_target_act, clip=True, sym=symmetric)

        return out.view(tensor.shape)

    def mid_tread_quantize_activation_per_channel(self, tensor, id):
        N, C, H, W = tensor.shape  # N x C x H x W
        t = tensor.detach().transpose(0, 1).contiguous()  # C x N x H x W
        t = t.view(t.shape[0], -1)

        symmetric = not (self.force_positive or self.half_range)
        tq, entropy = self.mid_tread_quantization(t, id, self.bit_alloc_target_act, clip=True, sym=symmetric)

        if entropy is not None and self.logger is not None:
            self.logger.log_metric(id + '.entropy', entropy.item(), step='auto', meterId='avg.entropy.act',
                                   weight=tensor.numel())

        output = tq.view(C, N, H, W).transpose(0, 1).contiguous()  # N x C x H x W
        return output.view(tensor.shape)

    def mid_tread_quantization(self, tensor, id, target, clip=False, sym=True):
        std = tensor.std(-1)
        omega = self.get_omega(std, target_bins=(2**target)).round()

        if clip:
            alpha_mult = tensor.new_tensor(self.get_alpha_mult(omega, sym=sym))
            mu = tensor.mean(dim=-1)
            b = torch.mean(torch.abs(tensor - mu.unsqueeze(-1)), dim=-1)

            rng = (2 * alpha_mult * b) if sym else (torch.max(mu, mu.new_tensor([0.])) + alpha_mult * b)
        else:
            rng = (tensor.max(-1)[0] - tensor.min(-1)[0]) if sym else tensor.max(-1)[0]

        Delta = torch.where(omega > 0, rng / omega,
                            tensor.new_tensor([np.finfo(np.float32).max]))

        # quantize
        out = tensor / Delta.unsqueeze(-1)
        out.round_()

        # clamp
        if clip:
            # Centralize quantization range around mean and make it non-negative for asymetric case
            mu_q = mu / Delta if sym else torch.max(mu, mu.new_tensor([0.])) / Delta
            c_max = mu_q + (omega / 2 if sym else omega)
            c_min = ((mu_q - omega / 2) if sym else tensor.new_tensor([0]))

            # In practice all the parameters Delta, omega, c_max, c_min can be pre-calculated based on statistics
            out = torch.min(out, c_max.unsqueeze(-1))
            out = torch.max(out, c_min.unsqueeze(-1))

        if self.measure_entropy:
            entropy = shannon_entropy(out, handle_negative=True)
            # workaround for out of memory issue
            torch.cuda.empty_cache()
        else:
            entropy = None

        # dequantize
        out.mul_(Delta.unsqueeze(-1))
        return out, entropy

    def get_alpha_laplace(self, tensor, stat_id=None, kind='mean', per_channel=False):
        if stat_id is not None:
            b = self.sm().get_tensor_stat(stat_id, 'b', kind=kind)
        else:
            if per_channel:
                b = self.__act_stats_perchannel__(tensor, ['b'], avg_over_batch=False)['b']
            else:
                b = self.__act_stats__(tensor, ['b'], avg_over_batch=False)['b']

        if self.bit_alloc_act and per_channel and self.num_bits <= 4:
            prior = 'std' if self.bit_alloc_prior == 'gaus' else 'b'
            if stat_id is not None:
                std = self.sm().get_tensor_stat(stat_id, prior, kind='mean')
                std = to_cuda(std, tensor.device)
            else:
                if per_channel:
                    std = self.__act_stats_perchannel__(tensor, [prior], avg_over_batch=False)[prior]
                else:
                    std = self.__act_stats__(tensor, [prior], avg_over_batch=False)[prior]

            bit_alloc = self.get_bits_alloc_fixed_target(std, self.bit_alloc_target_act, self.bit_alloc_round)
            aciq_factor = np.array([(self.alpha_laplace_positive[nbit.item()] if (self.force_positive or self.half_range) else self.alpha_laplace[nbit.item()]) for nbit in bit_alloc])
            aciq_factor = to_cuda(aciq_factor, tensor.device)
        else:
            aciq_factor = (self.alpha_laplace_positive[self.num_bits] if (self.force_positive or self.half_range) else self.alpha_laplace[self.num_bits])

        return to_cuda(b, tensor.device) * aciq_factor

    def get_alpha_gaus(self, tensor, tag, stat_id=None, per_channel=False):
        if stat_id is not None:
            std = self.sm().get_tensor_stat(stat_id, 'std', 'mean')
        else:
            if per_channel:
                std = self.__act_stats_perchannel__(tensor, ['std'], avg_over_batch=False)['std']
            else:
                std = self.__act_stats__(tensor, ['std'], avg_over_batch=False)['std']

        return std * (self.alpha_gaus_positive[self.num_bits] if (self.force_positive or self.half_range) else self.alpha_gaus[self.num_bits])

    def get_alpha_pstd(self, tensor, p, tag, stat_id=None, per_channel=False):
        if stat_id is not None:
            std = self.sm().get_tensor_stat(stat_id, 'std', 'mean')
        else:
            if per_channel:
                std = self.__act_stats_perchannel__(tensor, ['std'], avg_over_batch=False)['std']
            else:
                std = self.__act_stats__(tensor, ['std'], avg_over_batch=False)['std']

        return p * std

    def get_alpha_exp(self, tensor, stat_id=None, per_channel=False):
        if stat_id is not None:
            mean_abs = self.sm().get_tensor_stat(stat_id, 'mean')
        else:
            mean_abs = torch.mean(tensor.abs())
        return self.alpha_exp[self.num_bits] * mean_abs

    def alpha2DeltaOffset(self, alpha, max_value, min_value, mean, clip2max=False):
        alpha = to_numpy(alpha)
        max_value = to_numpy(max_value)
        min_value = to_numpy(min_value)
        mean = to_numpy(mean)
        if self.force_positive or self.half_range:
            delta = np.maximum(np.array(mean), 0) + alpha
            if clip2max:
                delta = np.minimum(delta, max_value)
            offset = 0
        else:
            delta = 2 * alpha
            if clip2max:
                delta = np.minimum(delta, max_value - min_value)
            offset = np.maximum(min_value, mean - alpha)

        return delta, offset

    def get_alpha(self, tensor, tag="", stat_id=None, clip_type='laplace', per_channel=False):
        if clip_type == 'laplace':
            alpha = self.get_alpha_laplace(tensor, stat_id, per_channel=per_channel)  # laplace clipping
        elif clip_type == 'gaus':
            alpha = self.get_alpha_gaus(tensor, tag, stat_id, per_channel=per_channel)  # gaussian clipping
        elif 'std' in clip_type:
            p = float(clip_type.replace('std', ''))
            alpha = self.get_alpha_pstd(tensor, p, tag, stat_id, per_channel=per_channel)  # 2std clipping
        elif clip_type == 'mix':
            mse_laplace = self.sm().get_tensor_stat(stat_id, 'mse_laplace', 'mean')
            mse_gaus = self.sm().get_tensor_stat(stat_id, 'mse_gaus', 'mean')
            mse_lowp = self.sm().get_tensor_stat(stat_id, 'mse_lowp', 'mean')

            alpha_laplace = self.get_alpha_laplace(tensor, stat_id, per_channel=per_channel)  # laplace clipping
            alpha_gaus = self.get_alpha_gaus(tensor, tag, stat_id, per_channel=per_channel)  # gaussian clipping
            # simulate alpha range for gemm_lowp
            min_ = self.sm().get_tensor_stat(stat_id, 'min', 'mean')
            max_ = self.sm().get_tensor_stat(stat_id, 'max', 'mean')
            alpha_lowp = (max_ - min_)/2

            alpha = np.where(mse_gaus < mse_laplace, alpha_gaus, alpha_laplace)
            alpha = np.where(mse_lowp < mse_gaus, alpha_lowp, alpha)

        return alpha

    def gemmlowpClippingQuantize(self, tensor, id, tag="", stat_id=None, clip_type='laplace'):
        if stat_id is not None:
            min_value = self.sm().get_tensor_stat(stat_id, 'min', 'mean')
            max_value = self.sm().get_tensor_stat(stat_id, 'max', 'mean')
            mean = self.sm().get_tensor_stat(stat_id, 'mean', 'mean')
        else:
            if self.pcq_a and len(tensor.shape) > 3 and (tensor.shape[2] > 1 or tensor.shape[3] > 1):
                stats = self.__act_stats_perchannel__(tensor, ['min', 'max'], avg_over_batch=False)
                mean = self.__act_stats_perchannel__(tensor, ['mean'], avg_over_batch=True)['mean']
            else:
                stats = self.__act_stats__(tensor, ['min', 'max', 'mean'], avg_over_batch=False)
                mean = stats['mean']
            min_value = stats['min']
            max_value = stats['max']
            # mean = stats['mean']

        if self.pcq_a and len(tensor.shape) > 3 and (tensor.shape[2] > 1 or tensor.shape[3] > 1) \
                and len(min_value) > 1 and len(max_value) > 1:
            # min_value = self.sm().get_tensor_stat(stat_id, 'min', 'min')
            # max_value = self.sm().get_tensor_stat(stat_id, 'max', 'max')
            alpha = self.get_alpha(tensor, tag, stat_id, clip_type, per_channel=True)
            range, min_value = self.alpha2DeltaOffset(alpha, max_value, min_value, mean)
            min_value = to_cuda(min_value, tensor.device)
            range = to_cuda(range, tensor.device)
            max_ = min_value + range
            res = self.gemmlowpQuantizeActivationPerChannel(tensor.contiguous(), id, tag, stat_id, min_=min_value, max_=max_)
        else:
            alpha = self.get_alpha(tensor, tag, stat_id, clip_type, per_channel=False)
            max_value = float(max_value); min_value = float(min_value); mean = float(mean); alpha = float(alpha)
            range, min_value = self.alpha2DeltaOffset(alpha, max_value, min_value, mean)
            res = self.__gemmlowpQuantize1__(tensor.contiguous(), to_cuda(range, tensor.device), to_cuda(min_value, tensor.device))

        return res

    def gemmlowpMinMaxQuantize(self, tensor, tag="", stat_id=None):
        if stat_id is not None:
            if self.stats_kind == 'mean':
                kind = {'min': 'mean', 'max': 'mean', 'mean': 'mean', 'std': 'mean', 'mean_abs': 'mean', 'b': 'mean'}
            else:
                kind = {'min': 'min', 'max': 'max', 'mean': 'mean', 'std': 'mean', 'mean_abs': 'mean', 'b': 'mean'}

            min_ = self.sm().get_tensor_stat(stat_id, 'min', kind['min'])
            max_ = self.sm().get_tensor_stat(stat_id, 'max', kind['max'])
            # print("use stats  for %s, min %f, max %f" % (stat_id, min_, max_))
        else:
            stats = self.__act_stats__(tensor, ['min', 'max'], avg_over_batch=('activation' in tag and 'classifier' not in tag))
            min_ = stats['min']
            max_ = stats['max']

        if self.force_positive or self.half_range:
            min_ = 0

        return self.__gemmlowpQuantize__(tensor, max_ - min_, min_)

    @staticmethod
    def get_bits_alloc(alpha, num_bits, round=False):
        B = len(alpha) * 2 ** num_bits

        # Calculate bit allocation
        p = alpha ** (2. / 3)
        bin_alloc = (B * p) / p.sum()
        bit_alloc = torch.round(torch.log2(bin_alloc)) if round else torch.ceil(torch.log2(bin_alloc))
        bit_alloc[bit_alloc < 0] = 0
        bit_alloc[bit_alloc > 8] = 8
        return bit_alloc

    @staticmethod
    def get_bits_alloc_fixed_target(alpha, num_bits, round=False):
        eps = 0.01
        goal_bits = num_bits
        target_bits = goal_bits
        delta = 1.
        iter = 0
        max_iter = 10
        while abs(2 * delta) > eps and iter < max_iter:
            iter += 1
            bit_alloc = IntQuantizer.get_bits_alloc(alpha, num_bits=target_bits, round=round)
            delta = (goal_bits - bit_alloc.mean()) / 2
            target_bits += delta.item()

        return bit_alloc

    def gemmlowpQuantizeActivationPerChannel(self, tensor, id, tag="", stat_id=None, min_=None, max_=None):
        if min_ is None:
            if self.force_positive or self.half_range:
                min_ = 0  # np.zeros(min_.shape)
            elif stat_id is not None:
                min_ = self.sm().get_tensor_stat(stat_id, 'min', kind=self.stats_kind)
            else:
                min_ = self.__act_stats_perchannel__(tensor, ['min'], avg_over_batch=False)['min']
        min_ = to_cuda(min_, tensor.device)

        if max_ is None:
            if stat_id is not None:
                max_ = self.sm().get_tensor_stat(stat_id, 'max', kind=self.stats_kind)
            else:
                max_ = self.__act_stats_perchannel__(tensor, ['max'], avg_over_batch=False)['max']
        max_ = to_cuda(max_, tensor.device)

        N, C, H, W = tensor.shape  # N x C x H x W
        t = tensor.detach().transpose(0, 1).contiguous()  # C x N x H x W
        t = t.view(t.shape[0], -1)

        if self.bit_alloc_act and self.num_bits <= 4:
            prior = 'std' if self.bit_alloc_prior == 'gaus' else 'b'
            if stat_id is not None:
                alpha = self.sm().get_tensor_stat(stat_id, prior, kind='mean')
                alpha = to_cuda(alpha, tensor.device)
            else:
                alpha = self.__act_stats_perchannel__(tensor, [prior], avg_over_batch=False)[prior]

            bit_alloc = self.get_bits_alloc_fixed_target(alpha, self.bit_alloc_target_act, self.bit_alloc_round)
        else:
            bit_alloc = None

        if self.measure_entropy:
            output, entropy = self.__gemmlowpQuantize1__(t, max_ - min_, min_, bit_alloc=bit_alloc, measure_entropy=True)
            if self.logger is not None:
                self.logger.log_metric(id + '.entropy', entropy.item(), step='auto', meterId='avg.entropy.act', weight=output.numel())
        else:
            output = self.__gemmlowpQuantize1__(t, max_ - min_, min_, bit_alloc=bit_alloc,
                                                         measure_entropy=self.measure_entropy)

        output = output.view(C, N, H, W).transpose(0, 1).contiguous()  # N x C x H x W
        return output.view(tensor.shape)

    def gemmlowpQuantizeWeightsPerChannel(self, tensor, id, min_=None, max_=None):
        # Assume weights with dimensions [OFM,IFM,K1,K2]
        t = tensor.view(tensor.shape[0], -1)

        # per output channel min, max
        if min_ is None:
            min_ = t.min(-1)[0]
        if max_ is None:
            max_ = t.max(-1)[0]

        if self.bit_alloc_weight and self.num_bits <= 4:
            alpha = t.std(-1)
            bit_alloc = self.get_bits_alloc_fixed_target(alpha, self.bit_alloc_target_weight, self.bit_alloc_round)
        else:
            bit_alloc = None

        if self.measure_entropy:
            output, entropy = self.__gemmlowpQuantize1__(t, max_ - min_, min_, bit_alloc=bit_alloc, measure_entropy=True)
            if self.logger is not None:
                self.logger.log_metric(id + '.entropy', entropy.item(), step='auto', meterId='avg.entropy.weight', weight=output.numel())
        else:
            output = self.__gemmlowpQuantize1__(t, max_ - min_, min_, bit_alloc=bit_alloc)

        return output.view(tensor.shape)

    def gemmlowpKldQuantize(self, tensor, tag="", stat_id=None):
        min_ = self.sm().get_tensor_stat(stat_id, 'min', 'mean')
        max_ = self.sm().get_tensor_stat(stat_id, 'max', 'mean')
        kld_th = self.sm().get_tensor_stat(stat_id, 'kld_th', 'mean')
        mean = self.sm().get_tensor_stat(stat_id, 'mean', 'mean')

        range, offset = self.alpha2DeltaOffset(kld_th, max_, min_, mean)

        return self.__gemmlowpQuantize__(tensor, range, offset)

    def symlowpQuantize(self, tensor):
        maxabs = torch.max(tensor.detach().abs())
        return self.__symlowpQuantize__(tensor, maxabs)

    # @staticmethod
    # def mse_laplace(b, alpha, num_bits):
    #     return 2 * (b ** 2) * np.exp(-alpha / b) + ((alpha ** 2) / (3 * 2 ** (2 * num_bits)))
    #
    # @staticmethod
    # def mse_exponential(mean_abs, alpha, num_bits):
    #     return 2 * (mean_abs ** 2) * np.exp(-alpha / mean_abs) + ((alpha ** 2) / (3 * 2 ** (2 * num_bits)))
    #
    # @staticmethod
    # def mse_gaus(sigma, alpha, num_bits):
    #     clipping_err = (sigma ** 2 + (alpha ** 2)) * (1 - math.erf(alpha / (sigma * np.sqrt(2.0)))) - \
    #                    np.sqrt(2.0 / np.pi) * alpha * sigma * (np.e ** ((-1) * (0.5 * (alpha ** 2)) / sigma ** 2))
    #     quant_err = (alpha ** 2) / (3 * (2 ** (2 * num_bits)))
    #     return clipping_err + quant_err

    @staticmethod
    def __act_stats__(tensor, stats, avg_over_batch=False):
        # Assume activation dimentions [N,C,H,W]
        t = tensor.view(tensor.shape[0], -1) if avg_over_batch else tensor.view(-1) # [N, CxHxW] or [NxCxHxW]

        stats_dict = {}
        for s in stats:
            if s == 'max':
                stats_dict[s] = t.max(dim=-1)[0] if avg_over_batch else t.max()
            elif s == 'min':
                stats_dict[s] = t.min(dim=-1)[0] if avg_over_batch else t.min()
            elif s == 'mean':
                stats_dict[s] = t.mean(dim=-1) if avg_over_batch else t.mean()
            elif s == 'b':
                stats_dict[s] = torch.mean(torch.abs(t - t.mean(dim=-1).unsqueeze(-1)), dim=-1) if avg_over_batch else torch.mean(torch.abs(t - t.mean()))
            elif s == 'std':
                stats_dict[s] = torch.std(t, dim=-1, unbiased=True) if avg_over_batch else t.std(unbiased=True)

            if avg_over_batch:
                stats_dict[s] = torch.mean(stats_dict[s], dim=0)

        return stats_dict

    @staticmethod
    def __act_stats_perchannel__(tensor, stats, avg_over_batch=False):
        # Assume activation dimentions [N,C,H,W]
        if not avg_over_batch:
            t = tensor.transpose(0, 1).contiguous()  # [C, N, H, W]
            t = t.view(t.shape[0], -1) # [C, NxHxW]
        else:
            t = tensor.view(tensor.shape[0], tensor.shape[1], -1)  # [N, C, HxW]

        stats_dict = {}
        for s in stats:
            if s == 'max':
                stats_dict[s] = t.max(dim=-1)[0]
            elif s == 'min':
                stats_dict[s] = t.min(dim=-1)[0]
            elif s == 'mean':
                stats_dict[s] = t.mean(dim=-1)
            elif s == 'b':
                stats_dict[s] = torch.mean(torch.abs(t - t.mean(dim=-1).unsqueeze(-1)), dim=-1)
            elif s == 'std':
                stats_dict[s] = torch.std(t, dim=-1, unbiased=True)

            if avg_over_batch:
                stats_dict[s] = torch.mean(stats_dict[s], dim=0)

        return stats_dict

    def __gemmlowpQuantize1__(self, tensor, delta, offset, bit_alloc=None, measure_entropy=False):
        qmin = 0.
        if bit_alloc is None:
            qmax = 2.**self.num_bits - 1.
            scale = (delta) / (qmax - qmin)
        else:
            qmax = 2.**bit_alloc - 1.
            scale = torch.where(qmax > 0, (delta) / (qmax - qmin), torch.tensor([0.]).to(tensor.device))

        scale = torch.max(scale, torch.tensor([1e-8]).to(tensor.device))

        output = tensor.detach()
        if self.enforce_true_zero:
            initial_zero_point = qmin - offset / scale
            # make zero exactly represented
            zero_point = torch.round(initial_zero_point)
            output = torch.div(output, scale.unsqueeze(-1))
            output = torch.add(output, zero_point.unsqueeze(-1))
        else:
            output = torch.add(output, -offset.unsqueeze(-1))
            output = torch.div(output, scale.unsqueeze(-1))

        if bit_alloc is None:
            output.clamp_(qmin, qmax).round_()  # quantize
        else:
            qmax = qmax.view(qmax.numel(), 1)
            output = torch.where(output.gt(qmax), qmax, output)
            output.clamp_(qmin).round_()

        if measure_entropy:
            entropy = shannon_entropy(output.int())
            # entropy = most_requent_value_compression(output.int())

        if self.enforce_true_zero:
            output = torch.add(output, -zero_point.unsqueeze(-1))
            output = torch.mul(output, scale.unsqueeze(-1))  # dequantize
        else:
            output = torch.mul(output, scale.unsqueeze(-1))
            output = torch.add(output, offset.unsqueeze(-1))  # dequantize

        # workaround for out of memory issue
        torch.cuda.empty_cache()

        if measure_entropy:
            return output.view(tensor.shape), entropy
        else:
            return output.view(tensor.shape)

    def __gemmlowpQuantize__(self, tensor, delta, offset):
        if self.stochastic:
            # Generate noise for stochastic rounding
            noise = tensor.new(tensor.shape).uniform_(-0.5, 0.5)
        else:
            noise = torch.cuda.FloatTensor(tensor.shape).fill_(0)

        # if enforce_true_zero and zero in range
        preserve_zero = self.enforce_true_zero and (offset + delta) > 0 and offset < 0
        return int_quantization.float2gemmlowp(tensor.contiguous(), delta, offset, self.num_bits, self.int_exp, preserve_zero, noise)

    def __symlowpQuantize__(self, tensor, maxabs):
        if self.stochastic:
            # Generate noise for stochastic rounding
            noise = tensor.new(tensor.shape).uniform_(-0.5, 0.5)
        else:
            noise = torch.cuda.FloatTensor(tensor.shape).fill_(0)

        return int_quantization.float2symlowp(tensor.contiguous(), maxabs, self.num_bits, self.int_exp, noise)


def int_quantizer(qtype, quant_params):
    if len(qtype) > len('int'):
        size = int(qtype[len('int'):])
    else:
        size = 32

    return IntQuantizer(size, quant_params)
