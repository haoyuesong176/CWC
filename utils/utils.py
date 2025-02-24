import os
import torch
import random
import logging
import numpy as np
import matplotlib.pyplot as plt


def _show_class_distribution(partition, n_classes):
    classes = [0 for i in range(n_classes)]
    targets = partition.data.targets
    index = partition.index
    for ind in index:
        classes[targets[ind]] += 1
    return classes


def plot_distribution(client_data_loaders, save_path, filename, n_classes=10):
    n_clients = len(client_data_loaders)
    classes_list = []
    for k in range(n_clients):
        classes = _show_class_distribution(client_data_loaders[k].dataset, n_classes)
        classes_list.append(classes)
    logging.info(classes_list)
    plt.clf()
    plt.imshow(classes_list, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.savefig(os.path.join(save_path, filename))
    

def logging_setup(savepath, lfile='log', to_console=False):

    """
    Description: 
    python logging setup

    Params: 
    savepath -- string, path in which the log file will be saved
    lfile -- string, name of the log file
    to_console -- bool, controlling whether to direct a copy of the log to the console
    """

    ldir = savepath
    if not os.path.exists(ldir):
        os.makedirs(ldir)
    lfile = os.path.join(ldir, lfile)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename=lfile)

    if to_console:
      console = logging.StreamHandler()
      console.setLevel(logging.INFO)
      console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
      logging.getLogger('').addHandler(console)


def seed_setup(seed):
    
    """
    Description:
    setting the seed for torch, numpy and random.
    """
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


class AverageMeter(object):

    """
    Description:
    a simple meter for tracking a variable's average.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

