import torch
from abc import ABC, abstractmethod
import numpy as np
from utils.misc import Singleton
import os
import shutil
import uuid

class DumpManager(metaclass=Singleton):
    def __init__(self, dump_dir=None):
        if dump_dir is None:
            raise Exception('dump_dir must be provided')

        self.dump_dir = dump_dir
        if os.path.exists(dump_dir):
            shutil.rmtree(dump_dir)
        os.makedirs(dump_dir)
        self.enabled = False
        self.tag = ''

    def __enter__(self):
        self.enabled = True
        return self

    def __exit__(self, *args):
        self.enabled = False

    def set_tag(self, tag):
        self.tag = tag

    def dump(self, tensor, name):
        if self.enabled:
            f = os.path.join(self.dump_dir, name + '_' + self.tag)
            print("dumping: %s" % f)
            np.save(f, tensor.cpu().numpy())
