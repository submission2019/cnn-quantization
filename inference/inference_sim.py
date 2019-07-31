import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(dir_path, os.path.pardir)
sys.path.append(root_dir)
import argparse
import time
import logging
import random
import shutil
import time
import collections
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils.meters import AverageMeter, accuracy
from pytorch_quantizer.quantization.inference.inference_quantization_manager import QuantizationManagerInference as QM
from utils.log import EvalLog
from utils.absorb_bn import search_absorbe_bn
from utils.mark_relu import resnet_mark_before_relu
from utils.model_naming import set_node_names
import numpy as np
from utils.dump_manager import DumpManager as DM
# import pretrainedmodels
# import pretrainedmodels.utils as mutils
from pathlib import Path

from utils.mllog import MLlogger


torch.backends.cudnn.deterministic = True


home = str(Path.home())
IMAGENET_FOR_INFERENCE = '/home/cvds_lab/datasets/ILSVRC2012/'

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
model_names.append('shufflenet')
model_names.append('mobilenetv2')
# model_names+=pretrainedmodels.model_names

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default=IMAGENET_FOR_INFERENCE,
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--device', default='cuda',
                    help='device assignment ("cpu" or "cuda")')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='device ids assignment (e.g 0 1 2 3')

parser.add_argument('--qtype', default=None, help='data type: bfloat[N], int[N]')
parser.add_argument('--qweight', '-qw', default='int8', help='quantizer for weights')
parser.add_argument('--qmodel', '-qm', type=int, default=None, help='load quantized model')
parser.add_argument('--no_bias_corr', '-nb', action='store_true', help='Load model w/o bias correction')
parser.add_argument('--q_off', action='store_true', help='dissable quantization')
parser.add_argument('--shuffle', '-sh', action='store_true', help='shuffle data')
parser.add_argument('--stochastic', '-s', action='store_true', help='Stochastic rounding.', default=False)
parser.add_argument('--hw_scale', '-hs', action='store_true', help='Force scale to be HW compatible', default=False)
parser.add_argument('--preserve_zero', '-pz', action='store_true', help='Preserve zero during quantization', default=False)
parser.add_argument('--eval_precision', '-ep', action='store_true', default=False, help='Evaluate different precisions, to csv.')
parser.add_argument('--clipping', '-c', default='no', help='Clipping type: [no, gaus, exp, laplace]')
parser.add_argument('--rho_act', '-ra', default=None, type=float, help='Rho parameter for activations clipping')
parser.add_argument('--rho_weight', '-rw', default=None, type=float, help='Rho parameter for weights clipping')
parser.add_argument('--stats_mode', '-sm', default='no', help='Specify if collect stats, use or not stats: [collect, use, no]')
parser.add_argument('--stats_kind', '-sk', default='mean', help='Specify kind of stats to use: [mean, max]')
parser.add_argument('--stats_folder', '-sf', default=None, help='Specify directory of for statistics')
parser.add_argument('--stats_batch_avg', '-sba', action='store_true', help='Whether average statistics across the batch')
parser.add_argument('--custom_test', '-ct', action='store_true', default=False, help='Perform some custom test.')
parser.add_argument('--dump_dir', '-dd', default=None, help='Directory to dump tensors')
parser.add_argument('--measure_stats', '-m', action='store_true', help='Measure statistics of activations during runtime', default=False)
parser.add_argument('--measure_stats_folder', '-mf', help='Folder to save measured statistics of activations during runtime', default=None)
parser.add_argument('--kld_threshold', '-kld', action='store_true', help='Measure statistics of activations during runtime', default=False)
parser.add_argument('--aciq_cal', '-ac', action='store_true', help='Enable aciq calibration mode', default=False)
parser.add_argument('--cal_set_size', '-cs', default=5120, type=int, help='Size of calibration set for threshold evaluation (default: 2048)')
parser.add_argument('--subset', '-ss', default=None, type=int, help='Run on subset of data')
parser.add_argument('--per_channel_quant_weights', '-pcq_w', action='store_true', help='Per channel quantization of weights', default=False)
parser.add_argument('--per_channel_quant_act', '-pcq_a', action='store_true', help='Per channel quantization of activations', default=False)
parser.add_argument('--bit_alloc_act', '-baa', action='store_true', help='Optimal bit allocation for each channel of activations', default=False)
parser.add_argument('--bit_alloc_weight', '-baw', action='store_true', help='Optimal bit allocation for each channel of weights', default=False)
parser.add_argument('--bit_alloc_rmode', '-bam', help='One of [round, ceil]', default='round')
parser.add_argument('--bit_alloc_prior', '-bap', help='One of [gaus, laplace]', default='gaus')
parser.add_argument('--bit_alloc_target_act', '-bata', type=float, help='Target value for bit allocation quota of activations', default=None)
parser.add_argument('--bit_alloc_target_weight', '-batw', type=float, help='Target value for bit allocation quota of weights', default=None)
parser.add_argument('--bias_corr_act', '-bca', action='store_true', help='Bias correction for activations', default=False)
parser.add_argument('--bias_corr_weight', '-bcw', action='store_true', help='Bias correction for weights', default=False)
parser.add_argument('--var_corr_weight', '-vcw', action='store_true', help='Variance correction for weights', default=False)
parser.add_argument('--measure_entropy', '-me', action='store_true', help='Measure entropy of activations', default=False)
parser.add_argument('--mlf_experiment', '-mlexp', help='Name of experiment', default=None)
args = parser.parse_args()

if args.arch == 'resnet50':
    max_mse_order_id = ['linear0_activation', 'conv52_activation', 'conv49_activation', 'conv46_activation', 'conv43_activation', 'conv2_activation', 'conv25_activation', 'conv5_activation', 'conv1_activation', 'conv3_activation', 'conv9_activation', 'conv50_activation', 'conv12_activation', 'conv6_activation', 'conv13_activation', 'conv51_activation', 'conv44_activation', 'conv48_activation', 'conv22_activation', 'conv8_activation', 'conv41_activation', 'conv29_activation', 'conv26_activation', 'conv19_activation', 'conv47_activation', 'conv40_activation', 'conv32_activation', 'conv45_activation', 'conv38_activation', 'conv18_activation', 'conv35_activation', 'conv37_activation', 'conv21_activation', 'conv16_activation', 'conv34_activation', 'conv28_activation', 'conv4_activation', 'conv31_activation', 'conv11_activation', 'conv27_activation', 'conv15_activation', 'conv14_activation', 'conv42_activation', 'conv17_activation', 'conv20_activation', 'conv10_activation', 'conv24_activation', 'conv23_activation', 'conv30_activation', 'conv39_activation', 'conv7_activation', 'conv36_activation', 'conv33_activation']
if args.arch == 'resnet18':
    max_mse_order_id = ['linear0_activation', 'conv19_activation', 'conv4_activation', 'conv17_activation', 'conv1_activation', 'conv2_activation', 'conv3_activation', 'conv7_activation', 'conv12_activation', 'conv8_activation', 'conv6_activation', 'conv9_activation', 'conv11_activation', 'conv14_activation', 'conv13_activation', 'conv18_activation', 'conv16_activation', 'conv15_activation', 'conv5_activation', 'conv10_activation']
elif args.arch == 'vgg16':
    max_mse_order_id = ['conv7_activation', 'conv8_activation', 'conv6_activation', 'conv5_activation', 'conv9_activation', 'conv4_activation', 'conv10_activation', 'conv11_activation', 'conv3_activation', 'conv12_activation', 'linear0_activation', 'conv2_activation', 'linear2_activation', 'linear1_activation', 'conv1_activation']
elif args.arch == 'vgg16_bn':
    max_mse_order_id = ['linear2_activation', 'linear0_activation', 'linear1_activation', 'conv12_activation', 'conv1_activation', 'conv3_activation', 'conv2_activation', 'conv10_activation', 'conv11_activation', 'conv6_activation', 'conv4_activation', 'conv8_activation', 'conv5_activation', 'conv7_activation', 'conv9_activation']
elif args.arch == 'resnet101':
    max_mse_order_id = ['linear0_activation', 'conv103_activation', 'conv100_activation', 'conv97_activation', 'conv94_activation', 'conv2_activation', 'conv3_activation', 'conv25_activation', 'conv1_activation', 'conv102_activation', 'conv13_activation', 'conv95_activation', 'conv9_activation', 'conv99_activation', 'conv101_activation', 'conv22_activation', 'conv8_activation', 'conv26_activation', 'conv98_activation', 'conv12_activation', 'conv96_activation', 'conv19_activation', 'conv91_activation', 'conv21_activation', 'conv92_activation', 'conv88_activation', 'conv18_activation', 'conv85_activation', 'conv82_activation', 'conv86_activation', 'conv56_activation', 'conv59_activation', 'conv89_activation', 'conv67_activation', 'conv4_activation', 'conv27_activation', 'conv83_activation', 'conv14_activation', 'conv5_activation', 'conv11_activation', 'conv53_activation', 'conv16_activation', 'conv6_activation', 'conv62_activation', 'conv64_activation', 'conv77_activation', 'conv47_activation', 'conv50_activation', 'conv68_activation', 'conv79_activation', 'conv65_activation', 'conv80_activation', 'conv61_activation', 'conv73_activation', 'conv76_activation', 'conv55_activation', 'conv32_activation', 'conv58_activation', 'conv71_activation', 'conv46_activation', 'conv49_activation', 'conv70_activation', 'conv74_activation', 'conv15_activation', 'conv24_activation', 'conv44_activation', 'conv41_activation', 'conv43_activation', 'conv52_activation', 'conv40_activation', 'conv31_activation', 'conv93_activation', 'conv23_activation', 'conv38_activation', 'conv20_activation', 'conv17_activation', 'conv90_activation', 'conv87_activation', 'conv35_activation', 'conv37_activation', 'conv84_activation', 'conv81_activation', 'conv10_activation', 'conv78_activation', 'conv34_activation', 'conv60_activation', 'conv63_activation', 'conv69_activation', 'conv7_activation', 'conv29_activation', 'conv51_activation', 'conv54_activation', 'conv75_activation', 'conv66_activation', 'conv72_activation', 'conv48_activation', 'conv57_activation', 'conv28_activation', 'conv33_activation', 'conv45_activation', 'conv42_activation', 'conv39_activation', 'conv36_activation', 'conv30_activation']
elif args.arch == 'inception_v3':
    max_mse_order_id = ['conv5_activation', 'conv12_activation', 'conv1_activation', 'conv7_activation', 'conv4_activation', 'conv2_activation', 'conv14_activation', 'conv19_activation', 'conv10_activation', 'conv92_activation', 'conv21_activation', 'conv22_activation', 'conv9_activation', 'conv77_activation', 'conv16_activation', 'conv47_activation', 'conv48_activation', 'conv17_activation', 'conv58_activation', 'conv8_activation', 'conv55_activation', 'conv56_activation', 'conv40_activation', 'conv63_activation', 'conv15_activation', 'conv62_activation', 'conv84_activation', 'conv54_activation', 'conv57_activation', 'conv52_activation', 'conv65_activation', 'conv91_activation', 'conv76_activation', 'conv34_activation', 'conv51_activation', 'conv85_activation', 'conv53_activation', 'conv83_activation', 'conv35_activation', 'conv50_activation', 'conv46_activation', 'conv82_activation', 'conv61_activation', 'conv30_activation', 'conv37_activation', 'conv67_activation', 'conv75_activation', 'conv64_activation', 'conv29_activation', 'conv66_activation', 'conv44_activation', 'conv33_activation', 'conv43_activation', 'conv38_activation', 'conv45_activation', 'conv42_activation', 'conv23_activation', 'conv36_activation', 'conv60_activation', 'conv32_activation', 'conv41_activation', 'conv79_activation', 'conv6_activation', 'conv13_activation', 'conv78_activation', 'conv20_activation', 'conv73_activation', 'conv74_activation', 'conv80_activation', 'conv31_activation', 'conv27_activation', 'conv81_activation', 'conv88_activation', 'conv68_activation', 'conv28_activation', 'conv26_activation', 'conv89_activation', 'conv72_activation', 'conv93_activation', 'conv90_activation', 'conv94_activation', 'conv3_activation', 'conv24_activation', 'conv87_activation', 'conv18_activation', 'conv69_activation', 'conv59_activation', 'conv25_activation', 'conv49_activation', 'linear1_activation', 'conv39_activation', 'conv86_activation', 'conv11_activation', 'conv95_activation']

torch.manual_seed(12345)


class InferenceModel:
    def __init__(self, ml_logger=None):
        self.ml_logger = ml_logger
        global args, best_prec1

        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')

        if 'cuda' in args.device and torch.cuda.is_available():
            if args.seed is not None:
                torch.cuda.manual_seed_all(args.seed)
            torch.cuda.set_device(args.device_ids[0])
            cudnn.benchmark = True
        else:
            args.device_ids = None

        # create model
        print("=> using pre-trained model '{}'".format(args.arch))
        if args.arch == 'shufflenet':
            import models.ShuffleNet as shufflenet
            self.model = shufflenet.ShuffleNet(groups=8)
            params = torch.load('ShuffleNet_1g8_Top1_67.408_Top5_87.258.pth.tar')
            self.model = torch.nn.DataParallel(self.model, args.device_ids)
            self.model.load_state_dict(params)
        # elif args.arch == 'mobilenetv2':
        #     from models.MobileNetV2 import MobileNetV2 as mobilenetv2
        #     self.model = mobilenetv2()
        #     params = torch.load('mobilenetv2_Top1_71.806_Top2_90.410.pth.tar')
        #     self.model = torch.nn.DataParallel(self.model, args.device_ids)
        #     self.model.load_state_dict(params)
        # elif args.arch not in models.__dict__ and args.arch in pretrainedmodels.model_names:
        #     self.model = pretrainedmodels.__dict__[args.arch](num_classes=1000, pretrained='imagenet')
        else:
            self.model = models.__dict__[args.arch](pretrained=True)

        set_node_names(self.model)

        # Mark layers before relue for fusing
        if 'resnet' in args.arch:
            resnet_mark_before_relu(self.model)

        # BatchNorm folding
        if 'resnet' in args.arch or args.arch == 'vgg16_bn' or args.arch == 'inception_v3':
            print("Perform BN folding")
            search_absorbe_bn(self.model)
            QM().bn_folding = True

        # if args.qmodel is not None:
        #     model_q_path = os.path.join(os.path.join(home, 'mxt-sim/models'), args.arch + '_lowp_pcq%dbit%s.pt' % (args.qmodel, ('' if args.no_bias_corr else '_bcorr')))
        #     model_q = torch.load(model_q_path)
        #     qldict = set_node_names(model_q, create_ldict=True)
        #     QM().ql_dict = qldict
        #     model_q.to(args.device)
        #     self.model.load_state_dict(model_q.state_dict())
        #     del model_q

        self.model.to(args.device)
        QM().quantize_model(self.model)

        if args.device_ids and len(args.device_ids) > 1 and args.arch != 'shufflenet' and args.arch != 'mobilenetv2':
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                self.model.features = torch.nn.DataParallel(self.model.features, args.device_ids)
            else:
                self.model = torch.nn.DataParallel(self.model, args.device_ids)

        # define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(args.device)

        cudnn.benchmark = True

        # Data loading code
        valdir = os.path.join(args.data, 'val')

        if args.arch not in models.__dict__ and args.arch in pretrainedmodels.model_names:
            dataparallel = args.device_ids is not None and len(args.device_ids) > 1
            tfs = [mutils.TransformImage(self.model.module if dataparallel else self.model)]
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            resize = 256 if args.arch != 'inception_v3' else 299
            crop_size = 224 if args.arch != 'inception_v3' else 299
            tfs = [
                transforms.Resize(resize),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize,
            ]

        self.val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose(tfs)),
            batch_size=args.batch_size, shuffle=(True if (args.kld_threshold or args.aciq_cal or args.shuffle) else False),
            num_workers=args.workers, pin_memory=True)

    def run(self):
        if args.eval_precision:
            elog = EvalLog(['dtype', 'val_prec1', 'val_prec5'])
            print("\nFloat32 no quantization")
            QM().disable()
            val_loss, val_prec1, val_prec5 = validate(self.val_loader, self.model, self.criterion)
            elog.log('fp32', val_prec1, val_prec5)
            logging.info('\nValidation Loss {val_loss:.4f} \t'
                         'Validation Prec@1 {val_prec1:.3f} \t'
                         'Validation Prec@5 {val_prec5:.3f} \n'
                         .format(val_loss=val_loss, val_prec1=val_prec1, val_prec5=val_prec5))
            print("--------------------------------------------------------------------------")

            for q in [8, 7, 6, 5, 4]:
                args.qtype = 'int{}'.format(q)
                print("\nQuantize to %s" % args.qtype)
                QM().quantize = True
                QM().reload(args, get_params())
                val_loss, val_prec1, val_prec5 = validate(self.val_loader, self.model, self.criterion)
                elog.log(args.qtype, val_prec1, val_prec5)
                logging.info('\nValidation Loss {val_loss:.4f} \t'
                             'Validation Prec@1 {val_prec1:.3f} \t'
                             'Validation Prec@5 {val_prec5:.3f} \n'
                             .format(val_loss=val_loss, val_prec1=val_prec1, val_prec5=val_prec5))
                print("--------------------------------------------------------------------------")
            print(elog)
            elog.save('results/precision/%s_%s_clipping.csv' % (args.arch, args.threshold))
        elif args.custom_test:
            log_name = 'results/custom_test/%s_max_mse_%s_cliping_layer_selection.csv' % (args.arch, args.threshold)
            elog = EvalLog(['num_8bit_layers', 'indexes', 'val_prec1', 'val_prec5'], log_name, auto_save=True)
            for i in range(len(max_mse_order_id)+1):
                _8bit_layers = ['conv0_activation'] + max_mse_order_id[0:i]
                print("it: %d, 8 bit layers: %d" % (i, len(_8bit_layers)))
                QM().set_8bit_list(_8bit_layers)
                val_loss, val_prec1, val_prec5 = validate(self.val_loader, self.model, self.criterion)
                elog.log(i+1, str(_8bit_layers), val_prec1, val_prec5)
            print(elog)
        else:
            val_loss, val_prec1, val_prec5 = validate(self.val_loader, self.model, self.criterion)
            if self.ml_logger is not None and self.ml_logger.mlflow.active_run() is not None:
                self.ml_logger.mlflow.log_metric('top1', val_prec1)
                self.ml_logger.mlflow.log_metric('top5', val_prec5)
                self.ml_logger.mlflow.log_metric('loss', val_loss)

            return val_loss, val_prec1, val_prec5


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    if args.dump_dir is not None:
        QM().disable()
        DM(args.dump_dir)

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if (args.stats_mode == 'collect' and i*args.batch_size >= args.cal_set_size and (args.kld_threshold or args.aciq_cal)) or \
                (args.subset is not None and i*args.batch_size >= args.subset):
                break
            if args.measure_entropy and i*args.batch_size >= args.subset:
                break
            # Uncomment to enable dump
            # QM().disable()
            # if i > 0:
            #     break
            if i == 0:
                QM().verbose = True
            input = input.to(args.device)
            target = target.to(args.device)
            if args.dump_dir is not None and i == 5:
                with DM(args.dump_dir):
                    DM().set_tag('batch%d'%i)
                    # compute output
                    output = model(input)
                    break
            else:
                output = model(input)

            QM().reset_counters()
            QM().verbose = False

            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(float(prec1), input.size(0))
            top5.update(float(prec5), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

def get_params(logger=None):
    qparams = {
        'int': {
            'clipping': args.clipping,
            'stats_kind': args.stats_kind,
            'true_zero': args.preserve_zero,
            'kld': args.kld_threshold,
            'pcq_weights': args.per_channel_quant_weights,
            'pcq_act': args.per_channel_quant_act,
            'bit_alloc_act': args.bit_alloc_act,
            'bit_alloc_weight': args.bit_alloc_weight,
            'bit_alloc_rmode': args.bit_alloc_rmode,
            'bit_alloc_prior': args.bit_alloc_prior,
            'bit_alloc_target_act': args.bit_alloc_target_act,
            'bit_alloc_target_weight': args.bit_alloc_target_weight,
            'bcorr_act': args.bias_corr_act,
            'bcorr_weight': args.bias_corr_weight,
            'vcorr_weight': args.var_corr_weight,
            'logger': logger,
            'measure_entropy': args.measure_entropy
        },
        'qmanager':{
            'rho_act': args.rho_act,
            'rho_weight': args.rho_weight
        }
    }  # TODO: add params for bfloat
    return qparams


if __name__ == '__main__':
    if args.stats_mode != 'collect':
        experiment = args.arch if args.mlf_experiment is None else args.mlf_experiment
        with MLlogger(os.path.join(home, 'mlruns_mxt'), experiment, args,
                      name_args=[args.arch, "W{}A{}".format(args.qweight, args.qtype)]) as ml_logger:
            with QM(args, get_params(ml_logger)):
                im = InferenceModel(ml_logger)
                im.run()
    else:
        with QM(args, get_params()):
            im = InferenceModel()
            im.run()
