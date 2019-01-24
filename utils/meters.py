import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class OnlineMeter(object):
    """Computes and stores the average and variance/std values of tensor"""

    def __init__(self):
        self.mean = torch.FloatTensor(1).fill_(-1)
        self.M2 = torch.FloatTensor(1).zero_()
        self.count = 0.
        self.needs_init = True

    def reset(self, x):
        self.mean = x.new(x.size()).zero_()
        self.M2 = x.new(x.size()).zero_()
        self.count = 0.
        self.needs_init = False

    def update(self, x):
        self.val = x
        if self.needs_init:
            self.reset(x)
        self.count += 1
        delta = x - self.mean
        self.mean.add_(delta / self.count)
        delta2 = x - self.mean
        self.M2.add_(delta * delta2)

    @property
    def var(self):
        if self.count < 2:
            return self.M2.clone().zero_()
        return self.M2 / (self.count - 1)

    @property
    def std(self):
        return self.var().sqrt()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().type_as(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AccuracyMeter(object):
    """Computes and stores the average and current topk accuracy"""

    def __init__(self, topk=(1,)):
        self.topk = topk
        self.reset()

    def reset(self):
        self._meters = {}
        for k in self.topk:
            self._meters[k] = AverageMeter()

    def update(self, output, target):
        n = target.nelement()
        acc_vals = accuracy(output, target, self.topk)
        for i, k in enumerate(self.topk):
            self._meters[k].update(acc_vals[i])

    @property
    def val(self):
        return {n: meter.val for (n, meter) in self._meters.items()}

    @property
    def avg(self):
        return {n: meter.avg for (n, meter) in self._meters.items()}

    @property
    def avg_error(self):
        return {n: 100. - meter.avg for (n, meter) in self._meters.items()}
