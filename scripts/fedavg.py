import os
import sys
import copy
import shutil 
import logging
import numpy as np 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter 

sys.path.insert(0, os.path.dirname(__file__) + '/..')
import configs.conf as const 
import data.datasets as data 
import fed_model.mlp as mlp
import fed_model.cnn as cnn
import fed_model.lenet as lenet
import fed_model.trans as trans
import fed_model.resnet as resnet
import utils.utils as utils
import core.train as train


def weights_aggregate(models):

    weights_list = [copy.deepcopy(model.state_dict()) for model in models]

    updated_weights = copy.deepcopy(weights_list[0])

    for weight_layer_name in updated_weights:
        
        if 'num_batches_tracked' in weight_layer_name:
            continue

        updated_weights[weight_layer_name] = torch.zeros_like(updated_weights[weight_layer_name])

        for client_weights in weights_list:
            updated_weights[weight_layer_name] += client_weights[weight_layer_name]
        updated_weights[weight_layer_name] /= len(weights_list)

    for model in models:
        model.load_state_dict(updated_weights)


def run(conf):

    # ------ Setup --------
    np.random.seed(conf.seed_use)
    conf.random_state = np.random.RandomState(conf.seed_use)
    utils.logging_setup(conf.save_path, lfile="train_log.txt", to_console=True)
    writer = SummaryWriter(os.path.join(conf.save_path, "tensorboard"))
    logging.info(conf)


    # -------- Prepare Datasets --------- 
    if conf.dataset == 'mnist':
        assert conf.n_clients == 5
        label_dists = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        train_partitions = data.load_mnist_split(label_dists, conf.image_path, train=True)
        test_partitions = data.load_mnist_split(label_dists, conf.image_path, train=False)
        train_dataloaders = data.get_dataloaders(train_partitions, conf.batch_size, conf.shuffle, conf.num_workers, conf.drop_last)
        test_dataloaders = data.get_dataloaders(test_partitions, conf.batch_size, conf.shuffle, conf.num_workers, conf.drop_last)
    elif conf.dataset == 'cifar10':
        assert conf.n_clients == 5
        label_dists = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        resize_on = True if conf.arch == 'trans' and conf.trans_pretrained else False
        train_partitions = data.load_cifar10_split(label_dists, conf.image_path, train=True, resize=resize_on)
        test_partitions = data.load_cifar10_split(label_dists, conf.image_path, train=False, resize=resize_on)
        train_dataloaders = data.get_dataloaders(train_partitions, conf.batch_size, conf.shuffle, conf.num_workers, conf.drop_last)
        test_dataloaders = data.get_dataloaders(test_partitions, conf.batch_size, conf.shuffle, conf.num_workers, conf.drop_last)
    elif conf.dataset == 'pmnist':
        permute_list = [np.random.permutation(28*28) for i in range(conf.n_clients)]
        train_partitions = data.load_mnist_permute(conf.image_path, permute_list, train=True)
        test_partitions = data.load_mnist_permute(conf.image_path, permute_list, train=False)
        train_dataloaders = data.get_dataloaders(train_partitions, conf.batch_size, conf.shuffle, conf.num_workers, conf.drop_last)
        test_dataloaders = data.get_dataloaders(test_partitions, conf.batch_size, conf.shuffle, conf.num_workers, conf.drop_last)
    elif conf.dataset == 'mnistd':
        train_dataset = data.get_mnist_dataset(conf.image_path, train=True)
        test_dataset = data.get_mnist_dataset(conf.image_path, train=False)
        loaders = data.client_noniid_dataloaders(train_dataset, test_dataset, conf.n_clients, conf.test_ratio, conf.non_iid_alpha, 
                                                 conf.seed_use, conf.random_state,
                                                 conf.batch_size, conf.shuffle, conf.num_workers, conf.drop_last)
        train_dataloaders = [pair[0] for pair in loaders]
        test_dataloaders = [pair[1] for pair in loaders]
        utils.plot_distribution(train_dataloaders, conf.save_path, "train_dist.jpg")
        utils.plot_distribution(test_dataloaders, conf.save_path, "test_dist.jpg")
    elif conf.dataset == 'cifar10d' or conf.dataset == 'cifar100d':
        train_dataset = data.get_cifar_dataset(conf.dataset, conf.image_path, split="train")
        test_dataset = data.get_cifar_dataset(conf.dataset, conf.image_path, split="test")
        loaders = data.client_noniid_dataloaders(train_dataset, test_dataset, conf.n_clients, conf.test_ratio, conf.non_iid_alpha, 
                                                 conf.seed_use, conf.random_state,
                                                 conf.batch_size, conf.shuffle, conf.num_workers, conf.drop_last)
        train_dataloaders = [pair[0] for pair in loaders]
        test_dataloaders = [pair[1] for pair in loaders]
        utils.plot_distribution(train_dataloaders, conf.save_path, "train_dist.jpg")
        utils.plot_distribution(test_dataloaders, conf.save_path, "test_dist.jpg")
    elif conf.dataset == 'isicd':
        train_dataset = data.get_isic_dataset(conf.image_path)
        loaders = data.client_noniid_dataloaders(train_dataset, None, conf.n_clients, conf.test_ratio, conf.non_iid_alpha, 
                                                 conf.seed_use, conf.random_state,
                                                 conf.batch_size, conf.shuffle, conf.num_workers, conf.drop_last)
        train_dataloaders = [pair[0] for pair in loaders]
        test_dataloaders = [pair[1] for pair in loaders]
        utils.plot_distribution(train_dataloaders, conf.save_path, "train_dist.jpg", n_classes=2)
        utils.plot_distribution(test_dataloaders, conf.save_path, "test_dist.jpg", n_classes=2)
    else: 
        raise RuntimeError("Dataset not supported!")



    # ------------- Prepare Model -------------

    models = []
    optimizers = []
    for k in range(conf.n_clients):
        if conf.arch == 'mlp':
            model = mlp.MLP(conf.dataset, hs=conf.mlp_hs).cuda()
        elif conf.arch == 'lenet':
            model = lenet.LeNet(conf.dataset).cuda()
        elif conf.arch == 'cnn':
            model = cnn.CNN().cuda()
        elif conf.arch == 'resnet':
            model = resnet.ResNet().cuda()
        elif conf.arch == 'trans':
            model = trans.Trans(pretrained=conf.trans_pretrained).cuda()
        else:
            raise RuntimeError("Arch not supported!")
        models.append(model)
        optimizers.append(optim.Adam(models[k].parameters(), lr=conf.lr))
    loss_fn = nn.CrossEntropyLoss(reduction='sum')


    """
    # ------------ Init Model -----------
    
    init_weights = models[0].state_dict()
    for model in models:
        model.load_state_dict(init_weights)
    """


    # ------------ Train ------------    

    for t in range(conf.n_rounds):
        
        logging.info("--------- round-{} ---------".format(t))

        for k in range(conf.n_clients):

            logging.info("------ client-{} ------".format(k))

            client_train_loader = train_dataloaders[k]
            client_model = models[k]
            client_optimizer = optimizers[k]

            for e in range(conf.n_epochs):
        
                logging.info("--- local epoch-{} ---".format(e))

                train_loss = train.train_loop(client_train_loader, client_model, loss_fn, client_optimizer)
                
        logging.info("--- weight aggregation ---".format(e))

        weights_aggregate(models)


        if conf.bacc: 
            # Recording (bacc == True) 
            val_records = []
            overall_val_loss, overall_val_acc = 0, 0
            overall_pred_list, overall_label_list = [], []
            for c in range(conf.n_clients):
                val_loss, val_acc, bacc_results = train.val_loop(test_dataloaders[c], models[c], loss_fn, bacc=True)
                bacc, pred_list, label_list = bacc_results
                val_records.append([val_loss, val_acc, bacc])
                overall_val_loss += val_loss
                overall_val_acc += val_acc
                overall_pred_list += pred_list
                overall_label_list += label_list
            overall_val_loss /= conf.n_clients
            overall_val_acc /= conf.n_clients
            overall_bacc = train.compute_bacc(overall_pred_list, overall_label_list, n_classes=max(overall_label_list)+1)
            train.tensorboard_record_fedavg_bacc(val_records, overall_val_loss, overall_val_acc, overall_bacc, writer, t)
        else:
            # Recording (bacc == False) 
            val_records = []
            overall_val_loss, overall_val_acc = 0, 0 
            for c in range(conf.n_clients):
                val_loss, val_acc = train.val_loop(test_dataloaders[c], models[c], loss_fn)
                val_records.append([val_loss, val_acc])
                overall_val_loss += val_loss
                overall_val_acc += val_acc
            overall_val_loss /= conf.n_clients
            overall_val_acc /= conf.n_clients
            train.tensorboard_record_fedavg(val_records, overall_val_loss, overall_val_acc, writer, t)

 
    
if __name__ == "__main__":
    conf = const.give_fed_args() 
    run(conf)

    
