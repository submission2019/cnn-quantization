from utils.misc import Singleton
import numpy as np
import pandas as pd
import os
import shutil
from utils.misc import sorted_nicely, cos_sim
import torch
from .kld_threshold import get_kld_threshold_15bins
from tqdm import tqdm
from pathlib import Path
home = str(Path.home())
base_dir = os.path.join(home, 'mxt-sim')


class StatisticManager(metaclass=Singleton):
    def __init__(self, folder, load_stats, stats = ['max', 'min', 'std', 'mean', 'kurtosis', 'mean_abs', 'b', 'dim'], batch_avg=False, kld_threshold=False, collect_err=True):
        self.name = folder
        self.folder = os.path.join(base_dir, 'statistics', folder)
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
        self.stats = {}
        self.metadata = {}
        self.save_stats = not load_stats
        self.kld_threshold = kld_threshold
        if kld_threshold:
            self.stats_names.append('kld_th')
        if load_stats:
            stats_file = os.path.join(self.folder, '%s_summary.csv' % self.name)
            assert os.path.exists(stats_file)
            self.stats_df = pd.read_csv(stats_file, index_col=0)
        else:
            self.stats_df = None
        pass

    def save_tensor_stats(self, tensor, tag, id, tensors_q={}, force_global_min_max=False):
        stat_arr = []
        # Calculate tensor stats
        for sn in self.stats_names:
            # Calculate statistics over the batch except for min/max
            t = tensor.view(-1)
            if sn == 'kurtosis':
                st = torch.mean(((t - t.mean()) / t.std())**4) - 3
            elif sn == 'mean_abs':
                st = torch.mean(t.abs())
                # st = torch.mean((t - t.mean(-1).unsqueeze(-1)).abs(), dim=-1)
            elif sn == 'b':
                st = torch.mean(torch.abs(t - t.mean()))
            elif sn == 'std':
                st = t.std(unbiased=True)
            elif sn == 'mean':
                st = t.mean()
            elif sn == 'max':
                if force_global_min_max:
                    st = t.max()
                else:
                    st = torch.mean(tensor.view(tensor.shape[0], -1).max(dim=-1)[0]) \
                        if self.batch_avg else t.max()
            elif sn == 'min':
                if force_global_min_max:
                    st = t.min()
                else:
                    st = torch.mean(tensor.view(tensor.shape[0], -1).min(dim=-1)[0]) \
                        if self.batch_avg else t.min()
            # elif sn == 'dist':
            #     st = torch.sqrt(torch.sum(t**2, dim=-1))
            elif sn == 'dim':
                st = t.numel()
            elif sn == 'kld_th':
                t_np = tensor.cpu().numpy()
                st = np.max(np.array([get_kld_threshold_15bins(t_np[i]) for i in tqdm(range(t_np.shape[0]))]))
            elif 'mse' in sn:
                if len(tensors_q) > 0:
                    t = tensors_q['orig'].view(t.shape)
                    t_q = tensors_q[sn.split('_')[1]].view(t.shape)
                    st = torch.mean((t - t_q)**2, dim=-1)
                else:
                    st = torch.tensor(np.nan)
            elif 'mae' in sn:
                if len(tensors_q) > 0:
                    t = tensors_q['orig'].view(t.shape)
                    t_q = tensors_q[sn.split('_')[1]].view(t.shape)
                    st = torch.mean(torch.abs(t - t_q), dim=-1)
                else:
                    st = torch.tensor(np.nan)
            elif 'cos' in sn:
                if len(tensors_q) > 0:
                    t = tensors_q['orig'].view(t.shape)
                    t_q = tensors_q[sn.split('_')[1]].view(t.shape)
                    st = cos_sim(t, t_q)
                else:
                    st = torch.tensor(np.nan)
            elif 'ang' in sn:
                if len(tensors_q) > 0:
                    t = tensors_q['orig'].view(t.shape)
                    t_q = tensors_q[sn.split('_')[1]].view(t.shape)
                    cos = cos_sim(t, t_q)
                    st = torch.acos(cos)
                else:
                    st = torch.tensor(np.nan)
            else:
                pass

            stat_arr.append(st.cpu().numpy() if sn != 'dim' and sn != 'kld_th' else st)

        # Add to stats dictionary
        if id in self.stats:
            stat_arr = np.vstack(stat_arr).transpose()
            s = np.concatenate([self.stats[id], stat_arr])
            self.stats[id] = s
        else:
            self.stats[id] = np.vstack(stat_arr).transpose()
            self.metadata[id] = tag

    def get_tensor_stats(self, id, kind={'min':'mean', 'max':'mean', 'mean': 'mean','std':'mean', 'range':'mean', 'mean_abs':'mean', 'b':'mean'}):
        if self.stats_df is not None:
            # TODO: add different options for min/max
            min_ = self.stats_df.loc[id, '%s_min' % kind['min']]
            max_ = self.stats_df.loc[id, '%s_max' % kind['max']]
            mean_ = self.stats_df.loc[id, '%s_mean' % kind['mean']]
            std_ = self.stats_df.loc[id, '%s_std' % kind['std']]
            mean_abs_ = self.stats_df.loc[id, '%s_mean_abs' % kind['mean_abs']]
            b_ = self.stats_df.loc[id, '%s_b' % kind['b']]
            return min_, max_, mean_, std_, mean_abs_, b_
        else:
            return None, None, None, None, None, None

    def get_tensor_stat(self, id, stat, kind='mean'):
        if self.stats_df is not None:
            s = self.stats_df.loc[id, '%s_%s' % (kind, stat)]
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
            all_stats_df = {}
            for s_id in self.stats:
                path = os.path.join(self.folder, '%s.csv' % s_id)
                df = pd.DataFrame(columns=self.stats_names, data=self.stats[s_id])
                df.to_csv(path, index=False)
                all_stats_df[s_id] = df
            self.__save_summry(all_stats_df)

    def __save_summry(self, all_stats_df):
        columns = []
        c_names = self.stats_names
        for c in c_names:
            columns.append('min_%s' % c)
            columns.append('mean_%s' % c)
            columns.append('max_%s' % c)

        df_summary = pd.DataFrame(columns=['internal_name']+columns)
        for s_id in sorted_nicely(all_stats_df.keys()):
            df_summary.loc[s_id, 'internal_name'] = self.metadata[s_id]
            for c in c_names:
                df_summary.loc[s_id, 'min_%s' % c] = all_stats_df[s_id][c].min()
                df_summary.loc[s_id, 'mean_%s' % c] = all_stats_df[s_id][c].mean()
                df_summary.loc[s_id, 'max_%s' % c] = all_stats_df[s_id][c].max()
            df_summary.loc[s_id, 'dim'] = all_stats_df[s_id]['dim'][0]
        path = os.path.join(self.folder, '%s_summary.csv' % self.name)
        df_summary.to_csv(path, index=True)
