from utils.misc import Singleton
import numpy as np
import pandas as pd
import os
import shutil
from utils.misc import sorted_nicely
import torch
from pathlib import Path


home = str(Path.home())
base_dir = os.path.join(home, 'mxt-sim')


class MeasureStatistics(metaclass=Singleton):
    def __init__(self, folder):
        self.enabled = False
        self.folder = os.path.join(base_dir, 'distance', folder)
        self.stats = {}
        self.stats_names = ['dist']

    def save_measure(self, tensor, id):
        if id not in self.stats:
            self.stats[id] = np.array([])

        # Assume dimensions of [N,C,H,W]
        t = tensor.view(tensor.shape[0], -1)
        d = torch.sum(t**2, dim=-1)
        stat_arr = d.cpu().numpy()

        # Add to stats dictionary
        s = np.concatenate([self.stats[id], stat_arr])
        self.stats[id] = s

    def __enter__(self):
        self.enabled = True
        self.stats.clear()
        return self

    def __exit__(self, *args):
        if self.enabled and len(self.stats) > 0:
            self.enabled = False
            # Save measures
            if os.path.exists(self.folder):
                shutil.rmtree(self.folder)
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)

            path = os.path.join(self.folder, 'distance.csv')
            pairs = [i for i in self.stats.items()]
            cols = [i[0] for i in pairs]
            data = np.array([i[1] for i in pairs]).transpose()
            df = pd.DataFrame(data=data, columns=cols)
            df.to_csv(path, index=False)
