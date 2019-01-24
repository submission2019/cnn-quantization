import torch
from . import qtypes
from utils.misc import Singleton
from utils import attacher
from utils.monitor import Monitor
import abc

INFERENCE_ONLY = False

class QuantizationManagerBase(metaclass=Singleton):
    def __init__(self):
        pass

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *args):
        self.disable()

    @abc.abstractclassmethod
    def createTruncationManager(self, args, qparams):
        return

    def enable(self):
        if self.quantize:
            self.enabled = True if not self.disable_quantization else False
            self.op_manager.enable()

    def disable(self):
        self.enabled = False
        self.op_manager.disable()

    def reload(self, args, qparams={}):
        self.disable()
        self.op_manager = self.createTruncationManager(args, qparams)
        self.enable()

    def reduce_logging_verbosity(self):
        self.op_manager.verbose = False

    def quantize_tensor(self, tensor, fprop=True, bprop=True, quantize_tensor=False):
        if self.enabled and quantize_tensor:
            return self.op_manager.quantize_tensor(tensor, fprop, bprop)
        else:
            return tensor

    def quantize_fprop(self, tensor):
        return self.quantize_tensor(tensor, fprop=True, bprop=False)

    def quantize_bprop(self, tensor):
        return self.quantize_tensor(tensor, fprop=False, bprop=True)

    def quantize_instant(self, tensor, tag="", quantize_tensor=False):
        if self.quantize and quantize_tensor:
            return self.op_manager.quantize_instant(tensor, tag)
        else:
            return tensor

class QuantizationManager(QuantizationManagerBase):
    def __init__(self, args, qparams):
        super(QuantizationManager, self).__init__()
        self.inference_only = INFERENCE_ONLY
        self.dual_precision = False
        self.quantize_batchnorm = args.quantize_bn
        self.quantize = args.qtype_fprop is not None or args.qtype_bprop is not None or args.qtype is not None
        self.op_manager = TruncationOpManager(args, qparams, self.inference_only, self.dual_precision)
        self.enabled = False

        def createTruncationManager(self, args, qparams):
            return TruncationOpManager(args, qparams)


class TruncationOpManager:
    def __load_quantizer__(self, qtype, qparams):
        qtype_name = qtype.rstrip('1234567890')
        quant_params = qparams[qtype_name] if qtype_name in qparams else {}
        quantizer = qtypes.__dict__[qtype_name + "_quantizer"](qtype, quant_params)
        return quantizer, quant_params

    def __init__(self, args, qparams, inference_only=False, dual_precision=False):
        self.inference_only = inference_only
        self.dual_precision = dual_precision
        self.verbose = False
        self.bprop_quantizer = self.fprop_quantizer = None

        self.origin_matmul = torch.Tensor.matmul
        self.origin_linear = torch.nn.functional.linear
        self.origin_conv2d = torch.nn.functional.conv2d

        if args.qtype_fprop is not None:
            self.quantize = True
            self.fprop_quantizer, self.fprop_qparams = self.__load_quantizer__(args.qtype_fprop, qparams)
        if args.qtype_bprop is not None:
            self.quantize = True
            self.bprop_quantizer, self.bprop_qparams = self.__load_quantizer__(args.qtype_bprop, qparams)

        if args.qtype_fprop is None and args.qtype_bprop is None and args.qtype is not None:
            self.quantize = True
            self.fprop_quantizer, self.fprop_qparams = self.__load_quantizer__(args.qtype, qparams)
            self.bprop_quantizer, self.bprop_qparams = self.__load_quantizer__(args.qtype, qparams)

    def enable(self):
        self.quantize_matmul()
        # self.quantize_linear()
        self.quantize_conv2d()

    def disable(self):
        torch.Tensor.matmul = self.origin_matmul
        torch.nn.functional.linear = self.origin_linear
        torch.nn.functional.conv2d = self.origin_conv2d

    # quantizes origin matmul
    def quantize_matmul(self):
        def quantized_matmul(tensor1, tensor2):
            assert False
            tensor1_ = attacher.pytorch_attach(tensor1, self.fprop_quantizer, self.bprop_quantizer)
            tensor2_ = attacher.pytorch_attach(tensor2, self.fprop_quantizer, self.bprop_quantizer)
            res = self.origin_matmul(tensor1_, tensor2_)
            return attacher.pytorch_attach(res, self.fprop_quantizer, self.bprop_quantizer)

        torch.Tensor.matmul = quantized_matmul

    # quantizes origin linear
    def quantize_linear(self):
        def quantized_linear(input, weight, bias=None):
            if self.inference_only:
                weight_ = self.quantize_instant(weight, "weight")
                res = self.origin_linear(input, weight_, bias)
                return self.quantize_instant(res, "activation_linear")
            elif self.dual_precision:
                return self.dual_prec_linear(input, weight, bias)
            else:
                input_ = attacher.pytorch_attach(input, self.fprop_quantizer, self.bprop_quantizer, tag='activation/in')
                weight_ = attacher.pytorch_attach(weight, self.fprop_quantizer, self.bprop_quantizer, tag='weight')
                if bias is not None:
                    bias_ = attacher.pytorch_attach(bias, self.fprop_quantizer, self.bprop_quantizer, tag='bias')
                else:
                    bias_ = bias

                res = self.origin_linear(input_, weight_, bias_)
                return attacher.pytorch_attach(res, self.fprop_quantizer, self.bprop_quantizer, tag='activation_linear')

        torch.nn.functional.linear = quantized_linear

    # quantizes origin conv2d
    def quantize_conv2d(self):
        def quantized_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
            if self.inference_only:
                weight_ = self.quantize_instant(weight, "weight")
                return self.origin_conv2d(input, weight_, bias, stride, padding, dilation, groups)
            elif self.dual_precision:
                return self.dual_prec_conv2d(input, weight, bias, stride, padding, dilation, groups)
            else:
                input_ = attacher.pytorch_attach(input, self.fprop_quantizer, self.bprop_quantizer, tag='activation/in')
                weight_ = attacher.pytorch_attach(weight, self.fprop_quantizer, self.bprop_quantizer, tag='weight')
                if bias is not None:
                    bias_ = attacher.pytorch_attach(bias, self.fprop_quantizer, self.bprop_quantizer, tag='bias')
                else:
                    bias_ = bias

                res = self.origin_conv2d(input_, weight_, bias_, stride, padding, dilation, groups)
                return attacher.pytorch_attach(res, self.fprop_quantizer, self.bprop_quantizer, tag='activation')

        torch.nn.functional.conv2d = quantized_conv2d

    def quantize_tensor(self, tensor, fprop=True, bprop=True):
        fprop = self.fprop_quantizer if fprop else None
        bprop = self.bprop_quantizer if bprop else None
        return attacher.pytorch_attach(tensor, fprop, bprop)

    def quantize_instant(self, tensor, tag=""):
        return self.fprop_quantizer(tensor, tag)

    def dual_prec_conv2d(self, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        # fprop conv2d quantized by fprop_quantizer
        input_fprop = attacher.pytorch_attach(input, self.fprop_quantizer, None, tag='activation/in')
        weight_fprop = attacher.pytorch_attach(weight, self.fprop_quantizer, None, tag='weight')
        if bias is not None:
            bias_fprop = attacher.pytorch_attach(bias, self.fprop_quantizer, None, tag='bias')
        else:
            bias_fprop = bias
        conv_fprop = self.origin_conv2d(input_fprop, weight_fprop, bias_fprop, stride, padding, dilation, groups)
        conv_fprop = attacher.pytorch_attach(conv_fprop, self.fprop_quantizer, None, tag='activation')

        # bprop conv2d quantized by bprop_quantizer
        input_bprop = attacher.pytorch_attach(input, None, self.bprop_quantizer, tag='activation/in')
        weight_bprop = attacher.pytorch_attach(weight, None, self.bprop_quantizer, tag='weight')
        if bias is not None:
            bias_bprop = attacher.pytorch_attach(bias, None, self.bprop_quantizer, tag='bias')
        else:
            bias_bprop = bias
        conv_bprop = self.origin_conv2d(input_bprop, weight_bprop, bias_bprop, stride, padding, dilation, groups)
        conv_bprop = attacher.pytorch_attach(conv_bprop, None, self.bprop_quantizer, tag='activation')
        return conv_fprop.detach() + conv_bprop - conv_bprop.detach()

    def dual_prec_linear(self, input, weight, bias=None):
        # fprop linear quantized by fprop_quantizer
        input_fprop = attacher.pytorch_attach(input, self.fprop_quantizer, None, tag='activation/in')
        weight_fprop = attacher.pytorch_attach(weight, self.fprop_quantizer, None, tag='weight')
        if bias is not None:
            bias_fprop = attacher.pytorch_attach(bias, self.fprop_quantizer, None, tag='bias')
        else:
            bias_fprop = bias
        linear_fprop = self.origin_linear(input_fprop, weight_fprop, bias_fprop)
        linear_fprop = attacher.pytorch_attach(linear_fprop, self.fprop_quantizer, None, tag='activation_linear')

        # bprop linear quantized by bprop_quantizer
        input_bprop = attacher.pytorch_attach(input, None, self.bprop_quantizer, tag='activation/in')
        weight_bprop = attacher.pytorch_attach(weight, None, self.bprop_quantizer, tag='weight')
        if bias is not None:
            bias_bprop = attacher.pytorch_attach(bias, None, self.bprop_quantizer, tag='bias')
        else:
            bias_bprop = bias
        linear_bprop = self.origin_linear(input_bprop, weight_bprop, bias_bprop)
        linear_bprop = attacher.pytorch_attach(linear_bprop, None, self.bprop_quantizer, tag='activation_linear')
        return linear_fprop.detach() + linear_bprop - linear_bprop.detach()
