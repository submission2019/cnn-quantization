import torch
from torch.utils.data import Dataset
from numpy.random import choice

class RandomSamplerReplacment(torch.utils.data.sampler.Sampler):
    """Samples elements randomly, with replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(torch.from_numpy(choice(self.num_samples, self.num_samples, replace=True)))

    def __len__(self):
        return self.num_samples


class LimitDataset(Dataset):

    def __init__(self, dset, max_len):
        self.dset = dset
        self.max_len = max_len

    def __len__(self):
        return min(len(self.dset), self.max_len)

    def __getitem__(self, index):
        return self.dset[index]

class ByClassDataset(Dataset):

    def __init__(self, ds):
        self.dataset = ds
        self.idx_by_class = {}
        for idx, (_, c) in enumerate(ds):
            self.idx_by_class.setdefault(c, [])
            self.idx_by_class[c].append(idx)

    def __len__(self):
        return min([len(d) for d in self.idx_by_class.values()])

    def __getitem__(self, idx):
        idx_per_class = [self.idx_by_class[c][idx]
                         for c in range(len(self.idx_by_class))]
        labels = torch.LongTensor([self.dataset[i][1]
                                   for i in idx_per_class])
        items = [self.dataset[i][0] for i in idx_per_class]
        if torch.is_tensor(items[0]):
            items = torch.stack(items)

        return (items, labels)


class IdxDataset(Dataset):
    """docstring for IdxDataset."""

    def __init__(self, dset):
        super(IdxDataset, self).__init__()
        self.dset = dset
        self.idxs = range(len(self.dset))

    def __getitem__(self, idx):
        data, labels = self.dset[self.idxs[idx]]
        return (idx, data, labels)

    def __len__(self):
        return len(self.idxs)
