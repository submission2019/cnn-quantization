from utils.misc import Singleton
import numpy as np
import pandas as pd
import os
import shutil
from utils.misc import sorted_nicely
import torch

class MeasureStatistics(metaclass=Singleton):
    def __init__(self):
        self.enabled = False
        self.folder = None
        self.subfolder = None
        self.stats = {}
        self.stats_names = ['eps_norm', 'eps_mse', 'eps_cos_sim', 'eps_ang_dist', 'eps_mean', 'eps_var',
                            'w_mean', 'w_var', 'w_norm', 'w_size', 'x_mean', 'x_var', 'x_norm', 'x_size',
                            'y_mean', 'y_var', 'y_norm', 'y_size', 'c_out']#, 'cos_wx', 'cos_weps']

    def save_measure(self, y_, y_with_noise, x_, w, id):
        stat_arr = []
        x = x_.view(x_.shape[0], -1)
        y = y_.view(y_.shape[0], -1)
        y_n = y_with_noise.view(y_with_noise.shape[0], -1)
        eps = y - y_n

        # epsilon norm = |t - t_noise|
        eps_norm = torch.norm(eps, p=2, dim=-1)
        stat_arr.append(eps_norm.cpu().numpy())

        # mse = eps_norm**2 / N
        mse = (eps_norm**2) / y.shape[-1]
        stat_arr.append(mse.cpu().numpy())

        # cosine similarity
        t_norm = torch.norm(y, p=2, dim=-1)
        t_n_norm = torch.norm(y_n, p=2, dim=-1)
        cos = torch.sum(y * y_n, dim=-1) / (t_norm * t_n_norm)
        cos_numpy = cos.cpu().numpy()
        stat_arr.append(cos_numpy)

        # angular distance = arccos(cosine sim) / pi
        ang_dist = np.nan_to_num(np.arccos(cos_numpy)) / np.pi
        stat_arr.append(ang_dist)

        # eps_mean
        eps_mean = eps.mean(-1)
        stat_arr.append(eps_mean.cpu().numpy())

        # eps_var
        eps_var = torch.mean((eps - eps_mean.unsqueeze(-1))**2, dim=-1)
        stat_arr.append(eps_var.cpu().numpy())

        # w_mean
        w_mean = w.mean()
        stat_arr.append(np.array([w_mean.cpu().numpy()]*x.shape[0]))

        # w_var
        w_var = torch.mean((w - w_mean) ** 2)
        stat_arr.append(np.array([w_var.cpu().numpy()] * x.shape[0]))

        # w_norm
        w_norm = torch.norm(w, p=2)
        stat_arr.append(np.array([w_norm.cpu().numpy()] * x.shape[0]))

        # w_size
        stat_arr.append(np.array([int(w.numel())] * x.shape[0]))

        # x_mean
        x_mean = x.mean(-1)
        stat_arr.append(x_mean.cpu().numpy())

        # x_var
        x_var = torch.mean((x - x_mean.unsqueeze(-1)) ** 2, dim=-1)
        stat_arr.append(x_var.cpu().numpy())

        # x_norm
        x_norm = torch.norm(x, p=2, dim=-1)
        stat_arr.append(x_norm.cpu().numpy())

        # x_size
        stat_arr.append(np.array([int(x.shape[-1])] * x.shape[0]))

        # y_mean
        y_mean = y.mean(-1)
        stat_arr.append(y_mean.cpu().numpy())

        # y_var
        y_var = torch.mean((y - y_mean.unsqueeze(-1)) ** 2, dim=-1)
        stat_arr.append(y_var.cpu().numpy())

        # y_norm
        y_norm = torch.norm(y, p=2, dim=-1)
        stat_arr.append(y_norm.cpu().numpy())

        # y_size
        stat_arr.append(np.array([int(y.shape[-1])] * y.shape[0]))

        # C_out
        stat_arr.append(np.array([int(w.shape[0])] * y.shape[0]))


        # Add to stats dictionary
        if id in self.stats:
            stat_arr = np.vstack(stat_arr).transpose()
            s = np.concatenate([self.stats[id], stat_arr])
            self.stats[id] = s
        else:
            self.stats[id] = np.vstack(stat_arr).transpose()

    def __enter__(self):
        self.enabled = True
        self.stats.clear()
        return self

    def __exit__(self, *args):
        self.enabled = False
        # Save measures
        if self.folder is not None and self.subfolder is not None:
            location = os.path.join(self.folder, self.subfolder)
            if os.path.exists(location):
                shutil.rmtree(location)
            if not os.path.exists(location):
                os.makedirs(location)
            for s_id in self.stats:
                path = os.path.join(location, '%s.csv' % s_id)
                df = pd.DataFrame(columns=self.stats_names, data=self.stats[s_id])
                df.to_csv(path, index=False)
