# from skimage
import numpy as np
import torch


def shannon_entropy(t, handle_negative=False):
    # workaround for out of memory issue
    torch.cuda.empty_cache()

    pk = torch.unique(t.flatten(), return_counts=True)[1]

    probs = pk.float() / pk.sum()
    probs[probs == 0] = 1
    entropy = -probs * torch.log2(probs)
    res = entropy.sum()

    return res


# def shannon_entropy_lastdim(t):
#     pk = torch.cat([torch.bincount(t[i], minlength=t.max()+1).view(1, -1) for i in range(t.shape[0])]).float()
#     pk = pk / pk.sum(-1).unsqueeze(-1)
#     pk[pk == 0] = 1
#     entropy = -pk * torch.log2(pk)
#     return entropy.sum(-1)


def most_requent_value_compression(t, base_bit=8, compressed_bit=1):
    # workaround for out of memory issue
    torch.cuda.empty_cache()

    pk = torch.unique(t.flatten(), return_counts=True)[1]
    mfv_count = pk.max()
    res = (mfv_count * compressed_bit + (t.numel() - mfv_count) * base_bit) / t.numel()

    return res
