from utils.misc import Singleton
import numpy as np
import pandas as pd
import os
import shutil
from utils.misc import sorted_nicely, cos_sim
import torch
from pathlib import Path
import pickle
from tqdm import tqdm


home = str(Path.home())
base_dir = os.path.join(home, 'mxt-sim')


def angle(x, y):
    cos = cos_sim(x, y)
    return torch.acos(cos)


class MeasureStatistics(metaclass=Singleton):
    def __init__(self, folder):
        self.enabled = False
        self.folder = os.path.join(base_dir, 'angle', folder)
        self.stats = {}
        self.targets = []

    def save_measure(self, tensor, id):
        # Assume dimensions of [N,C,H,W]
        t = tensor.view(tensor.shape[0], -1)
        ang_matrix = np.zeros(shape=(t.shape[0], t.shape[0]))
        for i in tqdm(range(t.shape[0])):
            for j in range(t.shape[0]):
                if j > i:
                    ang = angle(t[i], t[j])
                    ang_matrix[i][j] = float(ang)

        # Add to stats dictionary
        if id not in self.stats:
            self.stats[id] = ang_matrix
        else:
            s = np.vstack([self.stats[id], ang_matrix])
            self.stats[id] = s

    def save_target(self, target):
        self.targets = np.concatenate([self.targets, target.cpu().numpy()])
        pass

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

            stats_dict = {}
            for l in self.stats:
                df = pd.DataFrame(data=self.stats[l])
                stats_dict[l] = df

            stats_dict['target'] = self.targets
            path = os.path.join(self.folder, 'angle.pkl')
            f = open(path, "wb")
            pickle.dump(stats_dict, f)
            f.close()
            #
            # path = os.path.join(self.folder, 'distance.csv')
            # pairs = [i for i in self.stats.items()]
            # cols = [i[0] for i in pairs]
            # data = np.array([i[1] for i in pairs]).transpose()
            # df = pd.DataFrame(data=data, columns=cols)
            # df.to_csv(path, index=False)
