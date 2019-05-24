from sklearn.cluster import KMeans
import numpy as np
import torch
from tqdm import tqdm
import torchvision.models as models
import sys
import time
import os
import copy
from utils.absorb_bn import search_absorbe_bn
import argparse
from pathlib import Path

def clip1d_kmeans(x, num_bits=8, n_jobs=-1):
    orig_shape = x.shape
    x = np.expand_dims(x.flatten(), -1)
    kmeans = KMeans(n_clusters=2**num_bits, random_state=0)
    kmeans.fit(x)
    x = np.clip(x, kmeans.cluster_centers_.min(), kmeans.cluster_centers_.max())
    return x.reshape(orig_shape)


def quantize1d_kmeans(x, num_bits=8, n_jobs=-1):
    orig_shape = x.shape
    x = np.expand_dims(x.flatten(), -1)
    # init = np.expand_dims(np.linspace(x.min(), x.max(), 2**num_bits), -1)
    kmeans = KMeans(n_clusters=2**num_bits, random_state=0, n_jobs=n_jobs)
    x_kmeans = kmeans.fit_predict(x)
    q_kmeans = np.array([kmeans.cluster_centers_[i] for i in x_kmeans])
    return q_kmeans.reshape(orig_shape)


def is_ignored(name, param):
    # classifier or first layer or bias
    return ('fc' in name and param.shape[0] == 1000) or \
           ('weight' in name and param.shape[1] == 3) or \
           ('bias' in name) or \
           ('AuxLogits' in name) or \
           (name == 'Conv2d_2a_3x3.conv.weight') # WA for inception_v3


def quantize_model_parameters(model, num_bits):
    # Quantize parameters of the model with 4 bit kmeans
    named_params = [np for np in model.named_parameters() if not is_ignored(*np)]
    for np in tqdm(named_params):
        np[1].data = torch.tensor(quantize1d_kmeans(np[1].detach().numpy(), num_bits=num_bits))


def clip_model_parameters(model, num_bits):
    # Quantize parameters of the model with 4 bit kmeans
    named_params = [np for np in model.named_parameters() if not is_ignored(*np)]
    for np in tqdm(named_params):
        np[1].data = torch.tensor(clip1d_kmeans(np[1].detach().numpy(), num_bits=num_bits))


def process_model(arch, num_bits, base_dir, task='quantize'):
    model = models.__dict__[arch](pretrained=True)
    search_absorbe_bn(model)

    # Quantize model by kmeans non uniform quantization
    model_qkmeans = copy.deepcopy(model)
    if task == 'quantize':
        quantize_model_parameters(model_qkmeans, num_bits=num_bits)
    elif task == 'clip':
        clip_model_parameters(model_qkmeans, num_bits=num_bits)
    else:
        print("Invalid argument task=%s" % task)
        exit(-1)

    # Save model to home dir
    model_path = os.path.join(base_dir, 'models')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = os.path.join(model_path, arch + ('_kmeans%dbit.pt' % num_bits))
    print("Saving quantized model to %s" % model_path)
    torch.save(model_qkmeans, model_path)

    # Per channel bias correction
    model_bcorr = copy.deepcopy(model_qkmeans)
    p_km = [np for np in model_bcorr.named_parameters()]
    p_orig = [np for np in model.named_parameters()]
    for i in tqdm(range(len(p_km))):
        if not is_ignored(p_km[i][0], p_km[i][1]):
            w_km = p_km[i][1]
            w_orig = p_orig[i][1]
            mean_delta = w_km.view(w_km.shape[0], -1).mean(dim=-1) - w_orig.view(w_orig.shape[0], -1).mean(dim=-1)
            p_km[i][1].data = (w_km.view(w_km.shape[0], -1) - mean_delta.view(mean_delta.shape[0], 1)).view(
                w_orig.shape)

    model_path = model_path.split('.')[0] + '_bcorr.pt'
    print("Saving quantized model with bias correction to %s" % model_path)
    torch.save(model_bcorr, model_path)


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
parser = argparse.ArgumentParser()
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-bits', '--num_bits', default=4, type=int,
                    help='Number of bits for quantization')
parser.add_argument('-t', '--task', default='quantize', help='[quantize, clip]')
args = parser.parse_args()


if __name__ == '__main__':
    home = str(Path.home())
    base_dir = os.path.join(home, 'mxt-sim')
    print('%s %s model to %d bits' % (args.task, args.arch, args.num_bits))
    process_model(args.arch, num_bits=args.num_bits, base_dir=base_dir, task=args.task)
    print('Done')
