import math
from scipy.stats import entropy
from tqdm import tqdm
import numpy as np

def get_kld_threshold_15bins(arr):
    min_val, max_val, min_divergence, opt_th = _get_optimal_threshold(arr, num_bins=2001, num_quantized_bins=15)
    return opt_th


"""
This code taken from mxnet framework as is w/o any change
https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
"""
def _get_optimal_threshold(arr, num_bins=8001, num_quantized_bins=255):
    """Given a dataset, find the optimal threshold for quantizing it.
    Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    th = max(abs(min_val), abs(max_val))

    hist, hist_edges = np.histogram(arr, bins=num_bins, range=(-th, th))
    zero_bin_idx = num_bins // 2
    num_half_quantized_bins = num_quantized_bins // 2
    assert np.allclose(hist_edges[zero_bin_idx] + hist_edges[zero_bin_idx + 1],
                       0, rtol=1e-5, atol=1e-7)

    thresholds = np.zeros(num_bins // 2 + 1 - num_quantized_bins // 2)
    divergence = np.zeros_like(thresholds)
    quantized_bins = np.zeros(num_quantized_bins, dtype=np.int32)
    # i means the number of bins on half axis excluding the zero bin.
    for i in range(num_quantized_bins // 2, num_bins // 2 + 1):
        p_bin_idx_start = zero_bin_idx - i
        p_bin_idx_stop = zero_bin_idx + i + 1
        thresholds[i - num_half_quantized_bins] = hist_edges[p_bin_idx_stop]
        sliced_nd_hist = hist[p_bin_idx_start:p_bin_idx_stop]

        # generate reference distribution p
        p = sliced_nd_hist.copy()
        assert p.size % 2 == 1
        assert p.size >= num_quantized_bins
        # put left outlier count in p[0]
        left_outlier_count = np.sum(hist[0:p_bin_idx_start])
        p[0] += left_outlier_count
        # put right outlier count in p[-1]
        right_outlier_count = np.sum(hist[p_bin_idx_stop:])
        p[-1] += right_outlier_count
        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (sliced_nd_hist != 0).astype(np.int32)

        # calculate how many bins should be merged to generate quantized distribution q
        num_merged_bins = p.size // num_quantized_bins
        # merge hist into num_quantized_bins bins
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[num_quantized_bins * num_merged_bins:].sum()
        # expand quantized_bins into p.size bins
        q = np.zeros(p.size, dtype=np.float32)
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            if j == num_quantized_bins - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        q[sliced_nd_hist == 0] = 0
        p = _smooth_distribution(p)
        # There is a chance that q is an invalid probability distribution.
        try:
            q = _smooth_distribution(q)
        except ValueError:
            divergence[i - num_half_quantized_bins] = float("inf")
        divergence[i - num_half_quantized_bins] = entropy(p, q)
        quantized_bins[:] = 0

    min_divergence_idx = np.argmin(divergence)
    min_divergence = divergence[min_divergence_idx]
    opt_th = thresholds[min_divergence_idx]
    return min_val, max_val, min_divergence, opt_th

def _smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor and taking the
    corresponding amount off the non-zero values.
    Ref: http://web.engr.illinois.edu/~hanj/cs412/bk3/KL-divergence.pdf
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist
