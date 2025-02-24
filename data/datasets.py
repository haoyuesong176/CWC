import os
import math 
import pickle 
import functools
import numpy as np
import pandas as pd
from skimage import io
import skimage.transform as sktf
from PIL import Image

import torch
import torch.distributed as dist 
from torch.utils.data import Dataset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
from torchvision.transforms import ToTensor



class Partition(object):
    def __init__(self, data, index, permute=None):
        self.data = data 
        self.index = index 
        self.permute = permute # permute idx

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, sub_index):
        data_idx = self.index[sub_index]
        image, label = self.data[data_idx]
        if self.permute is not None:
            image = image.view(-1)[self.permute].view(image.size())
        return image, label      


# ----------- MNIST ------------


def load_mnist_permute(datasets_path, permute_list, train=True):
    n_clients = len(permute_list)
    root = os.path.join(datasets_path, "MNIST")
    mnist = datasets.MNIST(root, train=train, download=True, transform=ToTensor())
    partitions = []
    raw_index = np.array([i for i in range(len(mnist))])
    np.random.shuffle(raw_index) # np seed
    assert len(mnist) % n_clients == 0 # dividable
    size = int(len(mnist) / n_clients) 
    for i in range(n_clients):
        index = raw_index[i*size:(i+1)*size]
        partitions.append(Partition(mnist, index, permute=permute_list[i]))
    return partitions


def load_mnist_split(label_dists, datasets_path, train=True, resize=False):
    root = os.path.join(datasets_path, "MNIST")
    if resize:
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    else:
        transform = ToTensor()
    mnist = datasets.MNIST(root, train=train, download=True, transform=transform)
    partitions = []
    raw_index = np.array([i for i in range(len(mnist))])
    for ld in label_dists: 
        index_mask = (mnist.targets == 10) # all False
        for label in ld:
            index_mask |= (mnist.targets == label) 
        index = raw_index[index_mask]
        partitions.append(Partition(mnist, index))
    return partitions


def get_mnist_dataset(datasets_path, train=True, download=True):

    root = os.path.join(datasets_path, "MNIST")
    mnist = datasets.MNIST(root, train=train, download=download, transform=ToTensor())
    return mnist


def get_dataloaders(datasets, batch_size, shuffle, num_workers, drop_last):
    dataloaders = []
    for dataset in datasets:
        dataloaders.append(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last))
    return dataloaders


# ---------- CIFAR10 ------------

def load_cifar10_split(label_dists, datasets_path, train=True, transform=None, target_transform=None, resize=False):
    root = os.path.join(datasets_path, "CIFAR10")
    split = 'train' if train else 'test'
    cifar = _get_cifar('cifar10', root, split, transform, target_transform, download=True, resize=resize)
    partitions = []
    raw_index = np.array([i for i in range(len(cifar))])
    targets = np.array(cifar.targets)
    for ld in label_dists: 
        index_mask = (targets == 1234) # all False
        for label in ld:
            index_mask |= (targets == label) 
        index = raw_index[index_mask]
        partitions.append(Partition(cifar, index))
    return partitions


def _get_cifar(name, root, split, transform, target_transform, download, pn_normalize=True, apply_transform=False, resize=False):
    """Args:
    conf: the configuration class 
    name: str, cifar10/cifar100 
    root: the location to save/load the dataset 
    split: "train" / "test" 
    transform: the data augmentation for training  
    target_transform: the data augmentation for testing 
    download: bool variable
    """
    is_train = True if "train" in split else False

    # decide normalize parameter.
    if name == "cifar10":
        dataset_loader = datasets.CIFAR10
        normalize = (
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        )
    elif name == "cifar100":
        dataset_loader = datasets.CIFAR100
        normalize = (
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        )
    
    normalize = normalize if pn_normalize else None
    size = 384 if resize else 32

    # decide data type.
    if is_train:
        if apply_transform:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop((32, 32), 4),
                    transforms.Resize(size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
                + ([normalize] if normalize is not None else [])
            )
        else:
            transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()] + ([normalize] if normalize is not None else []))
    else:
        transform = transforms.Compose(
            [transforms.Resize(size), transforms.ToTensor()] + ([normalize] if normalize is not None else [])
        )
    return dataset_loader(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def get_cifar_dataset(name, datasets_path, split="train", transform=None, target_transform=None, 
                download=True):

    root = os.path.join(datasets_path, name.upper()[:-1])
    if name == "cifar10d" or name == "cifar100d":
        return _get_cifar(name[:-1], root, split, transform, target_transform, download)        
    else:
        raise RuntimeError("get_cifar_dataset: dataset not supported!")


# ---------- ISIC 2018 ------------


def load_isic_partitions(datasets_path, minor_ratio, test_ratio):
    # 4 clients: major + minor == 1500
    minor = int(1500 * minor_ratio)
    major = 1500 - minor
    n_test = int(1500 * test_ratio)
    n_test_minor = int(n_test * minor_ratio)
    n_test_major = n_test - n_test_minor
    root = os.path.join(datasets_path, "ISIC2018")
    dataset = ISIC2018(root)
    index0 = np.array([i for i in range(3000)])
    index1 = np.array([i+3000 for i in range(3000)])
    np.random.shuffle(index0)
    np.random.shuffle(index1)
    index0_list = [index0[0:minor], index0[minor:1500], index0[1500:1500+minor], index0[1500+minor:3000]]
    index1_list = [index1[0:major], index1[major:1500], index1[1500:1500+major], index1[1500+major:3000]]
    index0_list = [l.tolist() for l in index0_list]
    index1_list = [l.tolist() for l in index1_list]
    index_list = [np.array(ind0+ind1) for ind0, ind1 in zip(index0_list, index1_list)]

    partitions = []
    for client_index in index_list:
        client_test_index = client_index[0:n_test].tolist()
        client_train_index = client_index[n_test:1500].tolist()
        partitions.append((Partition(dataset, client_train_index), Partition(dataset, client_test_index)))
    return partitions
    
    


def get_isic_dataset(datasets_path):
    root = os.path.join(datasets_path, "ISIC2018")
    return ISIC2018(root)


class ISIC2018(Dataset):
    def __init__(self, root):
        self.label_path = os.path.join(root, "labels.csv") 
        self.img_dir = os.path.join(root, "images")
        self.names = os.listdir(self.img_dir)

        self.classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        self.class2idx = {}
        self.idx2class = {}
        for idx, cl in enumerate(self.classes):
            self.class2idx[cl] = idx
            self.idx2class[idx] = cl

        # build mapping
        self.mapping_path = os.path.join(root, 'mapping.pkl')
        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, 'rb') as f:
                self.mapping = pickle.load(f)
        else:
            df = pd.read_csv(self.label_path)
            X = list(df['image'])
            y = [self._get_cat(df, img) for img in X]
            mapping = list()
            for img, lbl in zip(X, y):
                img_path = os.path.join(self.img_dir, f'{img}.jpg')
                tup = (img_path, lbl)
                mapping.append(tup)
                with open(self.mapping_path, 'wb') as f:
                    pickle.dump(mapping, f)
            self.mapping = mapping

        # transform
        inp_size = 224
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.transform = transforms.Compose([transforms.Resize(360), 
                                             transforms.CenterCrop(inp_size), 
                                             transforms.ToTensor(), 
                                             transforms.Normalize(mean, std)])

        # targets
        self.targets = [lbl for path, lbl in self.mapping]

        # cropping to (3000 + 3000)
        self.index0 = []
        self.index1 = []
        for i, target in enumerate(self.targets):
            if target == 0:
                self.index0.append(i)
            else:
                self.index1.append(i)
        self.index = self.index0[:3000] + self.index1[:3000]

        self.targets = [0 for i in range(3000)] + [1 for i in range(3000)]
                        
    def __len__(self):
        return 6000

    def _get_cat(self, df, img):
        df_img = df[df['image'] == img]
        df_img.reset_index(drop=True, inplace=True)
        for cat in self.classes:
            v = int(df_img.loc[0, cat])
            if v == 1:  # where in the column is 1 is the cat
                lbl = self.class2idx[cat]
        lbl = 0 if lbl != 1 else 1
        return lbl
        
    def __getitem__(self, idx):
        idx = self.index[idx]
        img_path, label = self.mapping[idx]
        image = io.imread(img_path)
        image = Image.fromarray(image)
        image = self.transform(image)
        
        return image, label


# ----------- Dirichlet ------------


def client_noniid_dataloaders(train_set, test_set, n_clients, test_ratio, non_iid_alpha, seed_use, random_state,
                              batch_size, shuffle, num_workers, drop_last):

    if test_set is None: # for ISIC
        merged_dataset = train_set
    else:
        # Warmings: make sure that train_set, test_set are with the same tranforms
        merged_dataset = ConcatDataset([train_set, test_set])
        if type(train_set.targets) is not list:
            merged_dataset.targets = train_set.targets.tolist() + test_set.targets.tolist()
        else:
            merged_dataset.targets = train_set.targets + test_set.targets

    partition_size = [1.0 / n_clients for _ in range(n_clients)]

    data_partitioner = DataPartitioner(merged_dataset, partition_sizes=partition_size,
                                       partition_type="non_iid", non_iid_alpha=non_iid_alpha,
                                       seed_use=seed_use, random_state=random_state, partition_obj=False)

    dataloaders = []
    for k in range(n_clients):
        client_data = data_partitioner.use(k)
        split_ratio = [1 - test_ratio, test_ratio]
        spliter = DataPartitioner(client_data, split_ratio, "random", non_iid_alpha=None, seed_use=None, random_state=random_state)
        client_train_data = spliter.use(0)
        client_test_data = spliter.use(1)
        client_train_loader = torch.utils.data.DataLoader(client_train_data, 
                                            batch_size=batch_size, 
                                            shuffle=shuffle, 
                                            num_workers=num_workers, 
                                            pin_memory=True,
                                            drop_last=drop_last)
        client_test_loader = torch.utils.data.DataLoader(client_test_data, 
                                            batch_size=batch_size, 
                                            shuffle=shuffle, 
                                            num_workers=num_workers, 
                                            pin_memory=True,
                                            drop_last=drop_last)
        dataloaders.append([client_train_loader, client_test_loader])

    return dataloaders
        


# ------- Private -------


def _partition_merge(partitions):
    index_list = [p.index for p in partitions]
    index = np.concatenate(index_list)
    return Partition(partitions[0].data, index)
        

class DataPartitioner(object):
    def __init__(self, data, partition_sizes, partition_type, non_iid_alpha, seed_use, random_state,
                 consistent_indices=True, partition_obj=True,
                 ):
        """Args:
        data: Partition object or data array
        partition_sizes: number of data per device, [Number clients]
        partition_type: str
        consistent_indices: bool. If True, the indices are broadcast to all the devices        
        Majority of this function is from https://github.com/epfml/federated-learning-public-code/tree/7e002ef5ff0d683dba3db48e2d088165499eb0b9/codes/FedDF-code
        """
        self.partition_sizes = partition_sizes
        self.partition_type = partition_type
        self.consistent_indices = consistent_indices

        self.non_iid_alpha = non_iid_alpha
        self.seed_use = seed_use
        self.random_state = random_state
        
        self.partitions = []
        
        if partition_obj == False:  
            self.data_size = len(data.targets)
            self.data = data 
            indices = np.array([x for x in range(self.data_size)])
        else:        
            self.data_size = len(data.index)
            self.data = data.data 
            indices = data.index 
        self.partition_indices(indices)
            
    def partition_indices(self, indices):
        indices = self._create_indices(indices) 
        if self.consistent_indices:
            indices = self._get_consistent_indices(indices)
        from_index = 0 
        for partition_size in self.partition_sizes:
            to_index = from_index + int(partition_size * self.data_size)
            self.partitions.append(indices[from_index:to_index])
            from_index = to_index         
            
    def _create_indices(self, indices):

        if self.partition_type == "original":
            pass 

        elif self.partition_type == "random":
            self.random_state.shuffle(indices)

        elif self.partition_type == "sort":
            indices2targets = np.array([(idx, s_target) for idx, s_target in enumerate(self.data.targets) if idx in indices])
            indices = indices2targets[np.argsort(indices2targets[:, 1]), 0]
            
        elif self.partition_type == "non_iid":

            num_class = len(np.unique(self.data.targets))
            num_indices = len(indices)
            num_workers = len(self.partition_sizes)
            indices2targets = np.array([(idx, s_target) for idx, s_target in enumerate(self.data.targets) if idx in indices])

            list_of_indices = _build_non_iid_by_dirichlet(random_state=self.random_state, 
                                                            indices2targets=indices2targets, 
                                                            non_iid_alpha=self.non_iid_alpha, 
                                                            num_classes=num_class, num_indices=num_indices,
                                                            n_workers=num_workers)
            indices = functools.reduce(lambda a, b: a+b, list_of_indices)  # concatenate over the list of indices 

        else:
            raise NotImplementedError("The partition type %s is not implemented yet" % self.partition_type)
        return indices 
    
    def _get_consistent_indices(self, indices):
        if dist.is_initialized():
            indices = torch.IntTensor(indices)
            dist.broadcast(indices, src=0)
            return list(indices)
        else:
            return indices
            
    def use(self, partition_id):
        return Partition(self.data, self.partitions[partition_id])
            
    
def _build_non_iid_by_dirichlet(random_state, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers):
    """Args:
    random_state: numpy random state for sampling from dirichlet distribution
    indices2targets: [N, 2] [indices, class_labels]
    non_iid_alpha: a float value, the smaller the value is, the more non-iid the data is distributed 
    num_classes: the number of unique class labels, i.e., 10 for CIFAR10 
    num_indices: the length of the indices2targets 
    n_workers: number of clients
    """
    if n_workers == 40 and non_iid_alpha == 0.1:
        n_auxi_worker = 20 
    elif n_workers == 40 and non_iid_alpha == 0.01:
        n_auxi_worker = 10 
    elif n_workers == 100 and num_classes == 10:
        n_auxi_worker = 10 
    elif n_workers == 100 and num_classes == 100 and non_iid_alpha == 0.01:
        n_auxi_worker = 20
    else:
        n_auxi_worker = n_workers 
    random_state.shuffle(indices2targets)  # shuffle along the row-dimension
    
    from_index = 0 
    splitted_targets = []
    num_split = math.ceil(n_workers / n_auxi_worker)
    
    split_n_workers = [n_auxi_worker if idx < num_split - 1 else n_workers - n_auxi_worker * (num_split - 1) for idx in range(num_split)]
    split_ratios = [v / n_workers for v in split_n_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(ratio * num_indices)
        splitted_targets.append(np.array(indices2targets[from_index:to_index]))
        from_index = to_index
    
    idx_batch = []
    
    for idx, s_target in enumerate(splitted_targets):
        s_target_size = len(s_target)
        s_n_worker = split_n_workers[idx]
        min_size = 0 
        while min_size < int(0.5 * s_target_size / s_n_worker):
            _idx_batch = [[] for _ in range(s_n_worker)]
            for s_class in range(num_classes):
                map_orig_class_index = np.where(s_target[:, 1] == s_class)[0]
                idx_class = s_target[map_orig_class_index, 0]
                try:
                    proportions = random_state.dirichlet(np.repeat(non_iid_alpha, s_n_worker))
                    # proportions = [1 / s_n_worker for _ in range(s_n_worker)]  # perfect class imbalance situation
                    q = 0
                    for p, idx_j in zip(proportions, _idx_batch):
                        if len(idx_j) >= s_target_size / s_n_worker:
                            proportions[q] = 0 
                        q+=1                            
                    proportions = proportions / np.sum(proportions)
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[:-1] 
                    split_baseon_proportion = np.split(idx_class, proportions)
                    for batch_idx, s_batch in enumerate(_idx_batch):
                        s_batch += split_baseon_proportion[batch_idx].tolist()
                        _idx_batch[batch_idx] = s_batch 
                    sizes = [len(s_batch) for s_batch in _idx_batch]
                    min_size = np.min(sizes)
                    # print("class label ", s_class, " minimum size", min_size)
                except ZeroDivisionError:
                    pass
        idx_batch += _idx_batch
    return idx_batch
 
