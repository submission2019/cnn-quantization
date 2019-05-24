from utils.misc import Singleton
import numpy as np
import pandas as pd
import os
import shutil
from utils.misc import sorted_nicely, cos_sim
import torch
import pickle
from pathlib import Path


home = str(Path.home())
base_dir = os.path.join(home, 'mxt-sim')

SAVE_FULL_STATS = False

class StatisticManagerPerChannel(metaclass=Singleton):
    def __init__(self, folder, load_stats, stats = ['max', 'min', 'std', 'mean', 'kurtosis', 'b', 'std_pos'], batch_avg=False, collect_err=False):
        self.name = folder
        self.folder = os.path.join(base_dir, 'statistics/per_channel', folder)
        self.stats_names = stats
        self.collect_err = collect_err
        self.batch_avg = batch_avg
        if collect_err:
            self.stats_names.append('mse_lowp')
            self.stats_names.append('mse_gaus')
            self.stats_names.append('mse_laplace')
            # self.stats_names.append('mae_gaus')
            # self.stats_names.append('mae_laplace')
            self.stats_names.append('cos_lowp')
            self.stats_names.append('cos_gaus')
            self.stats_names.append('cos_laplace')
            # self.stats_names.append('ang_gaus')
            # self.stats_names.append('ang_laplace')
        self.save_stats = not load_stats
        if load_stats:
            stats_file = os.path.join(self.folder, '%s_statistics_perchannel_summary.pkl' % self.name)
            assert os.path.exists(stats_file)
            f = open(stats_file, 'rb')
            self.stats = pickle.load(f)
        else:
            self.stats = {}
        pass

    def save_tensor_stats(self, tensor, tag, id, tensors_q={}, force_global_min_max=False):
        # ignore FC or 1x1 case
        if len(tensor.shape) < 3 or (tensor.shape[2] == 1 and tensor.shape[3] == 1):
            return

        # Assume activation dimentions [N,C,H,W]
        t = tensor.transpose(0, 1).contiguous()  # [C, N, H, W]
        t = t.view(t.shape[0], -1) # [C, NxHxW]

        mean_ = t.mean(-1)
        std_ = torch.std(t, dim=-1, unbiased=True)
        for sn in self.stats_names:
            if sn == 'kurtosis':
                st = torch.mean(((t - mean_.unsqueeze(-1)) / std_.unsqueeze(-1))**4, dim=-1) - 3
            elif sn == 'b':
                st = torch.mean(torch.abs(t - mean_.unsqueeze(-1)), dim=-1)
            elif sn == 'std':
                st = std_
            elif sn == 'std_pos':
                t_relu = torch.nn.functional.relu(t)
                st = torch.std(t_relu, dim=-1, unbiased=True)
            elif sn == 'mean':
                st = mean_
            elif sn == 'max':
                if force_global_min_max:
                    st = t.max(-1)[0]
                else:
                    st = torch.mean(tensor.view(tensor.shape[0], tensor.shape[1], -1).max(dim=-1)[0], dim=0) \
                        if self.batch_avg else t.max(-1)[0]
            elif sn == 'min':
                if force_global_min_max:
                    st = t.min(-1)[0]
                else:
                    st = torch.mean(tensor.view(tensor.shape[0], tensor.shape[1], -1).min(dim=-1)[0], dim=0)if self.batch_avg else \
                        torch.min(tensor.view(tensor.shape[0], tensor.shape[1], -1).min(dim=-1)[0], dim=0)[0]
            elif 'mse' in sn:
                if len(tensors_q) > 0:
                    t_orig = tensors_q['orig']
                    t_q = tensors_q[sn.split('_')[1]]
                    st = torch.mean(torch.mean(((t_orig - t_q)**2).view(t_orig.shape[0], t_orig.shape[1], -1), dim=-1), dim=0)
                else:
                    continue
            # elif 'mae' in sn:
            #     if len(tensors_q) > 0:
            #         t = tensors_q['orig'].view(t.shape)
            #         t_q = tensors_q[sn.split('_')[1]].view(t.shape)
            #         st = torch.mean(torch.abs(t - t_q), dim=-1)
            #     else:
            #         continue
            elif 'cos' in sn:
                if len(tensors_q) > 0:
                    t_orig = tensors_q['orig'].view(tensor.shape[0], tensor.shape[1], -1)
                    t_q = tensors_q[sn.split('_')[1]].view(tensor.shape[0], tensor.shape[1], -1)
                    st = cos_sim(t_orig, t_q, dims=[-1, 0])
                else:
                    continue
            # elif 'ang' in sn:
            #     if len(tensors_q) > 0:
            #         t = tensors_q['orig'].view(t.shape)
            #         t_q = tensors_q[sn.split('_')[1]].view(t.shape)
            #         cos = cos_sim(t, t_q)
            #         st = torch.acos(cos)
            #     else:
            #         continue
            else:
                pass

            st = st.cpu().numpy()
            if 'cos' in sn:
                st = np.nan_to_num(st)
                st[st == 0] = 1.

            if id not in self.stats:
                self.stats[id] = {}
            if sn not in self.stats[id]:
                self.stats[id][sn] = st
            else:
                # if len(st.shape) > 1:
                self.stats[id][sn] = np.vstack([self.stats[id][sn], st])
                # else:
                #     self.stats[id][sn] = np.concatenate([self.stats[id][sn], st])

    def get_tensor_stat(self, id, stat, kind='mean'):
        if self.stats is not None:
            s = self.stats[id]
            s = s['%s_%s' % (kind, stat)]
        else:
            s = None
        return s

    def __exit__(self, *args):
        if self.save_stats:
            # Save statistics
            if os.path.exists(self.folder):
                shutil.rmtree(self.folder)
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)

            # Avoid saving full stats by default since it takes huge amound of disk space
            if SAVE_FULL_STATS:
                path = os.path.join(self.folder, 'statistics_perchannel.pkl')
                f = open(path, "wb")
                pickle.dump(self.stats, f)
                f.close()

            self.__save_summry()

    def __save_summry(self):
        stats_summary = {}
        stats = self.stats_names
        columns = []
        for c in stats:
            columns.append('min_%s' % c)
            columns.append('mean_%s' % c)
            columns.append('max_%s' % c)

        for l in self.stats:
            df = pd.DataFrame(columns=columns)
            for s in stats:
                if s in self.stats[l]:
                    t = self.stats[l][s]
                    df['min_%s' % s] = t.min(axis=0) if len(t.shape) > 1 else [t.min(axis=0)]
                    df['mean_%s' % s] = t.mean(axis=0) if len(t.shape) > 1 else [t.mean(axis=0)]
                    df['max_%s' % s] = t.max(axis=0) if len(t.shape) > 1 else [t.max(axis=0)]
            stats_summary[l] = df

        path = os.path.join(self.folder, '%s_statistics_perchannel_summary.pkl' % self.name)
        f = open(path, "wb")
        pickle.dump(stats_summary, f)
        f.close()
