import os
import sys
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
import fed_model.resnet as resnet
import fed_model.lenet as lenet
import fed_model.trans as trans
import utils.utils as utils
import core.train as train


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
        resize_on = True if conf.arch == 'lenet' else False
        train_partitions = data.load_mnist_split(label_dists, conf.image_path, train=True, resize=resize_on)
        test_partitions = data.load_mnist_split(label_dists, conf.image_path, train=False, resize=resize_on)
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
    elif conf.dataset == 'cifar10d':
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
    if conf.arch == 'mlp': # for both MNIST & CIFAR10
        model = mlp.MLP_SI(conf.dataset, mode=conf.mode, n_clients=conf.n_clients, alpha=conf.alpha, epsilon=conf.epsilon, hs=conf.mlp_hs).cuda()
    elif conf.arch == 'lenet': # for CIFAR10
        model = lenet.LeNet_SI(conf.dataset, mode=conf.mode, n_clients=conf.n_clients, alpha=conf.alpha, epsilon=conf.epsilon).cuda()
    elif conf.arch == 'cnn': # for CIFAR10
        model = cnn.CNN_SI(mode=conf.mode, n_clients=conf.n_clients, alpha=conf.alpha, epsilon=conf.epsilon).cuda()
    elif conf.arch == 'resnet':
        model = resnet.ResNet_SI(mode=conf.mode, n_clients=conf.n_clients, alpha=conf.alpha, epsilon=conf.epsilon).cuda()
    elif conf.arch == 'trans': # for CIFAR10
        model = trans.Trans_SI(pretrained=conf.trans_pretrained, mode=conf.mode, n_clients=conf.n_clients, alpha=conf.alpha, epsilon=conf.epsilon).cuda()
    else:
        raise RuntimeError("Arch not supported!")
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    optimizers = []
    for k in range(conf.n_clients):
        optimizers.append(optim.Adam(model.parameters(), lr=conf.lr))


    # ------------ Train ------------    
    
    # init SI_prev_task for task_0
    for n, p in model.named_parameters():
        if p.requires_grad:
            n = n.replace('.', '__')
            model.register_buffer('{}_SI_prev_task'.format(n), p.detach().clone())


    # Training Loops
    step = 0

    for t in range(conf.n_rounds):
        
        logging.info("--------- round-{} ---------".format(t))


        # For Each Task
        for k in range(conf.n_clients):

            logging.info("------ client-{} ------".format(k))

            # init W & p_old for current task(client)
            W, p_old = {},{}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone() 

            client_train_loader = train_dataloaders[k]
            optimizer = optimizers[k]

            for e in range(conf.n_epochs):
        
                logging.info("--- local epoch-{} ---".format(e))
                
                # ----- Train Loop -----
 
                model.train()

                pred_train_loss = 0
                surr_train_loss = 0
                train_loss = 0

                for i, (X, y) in enumerate(client_train_loader): 

                    # Data
                    X = X.cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True)

                    # Calculate L_seg
                    if conf.lseg:
                        pred = model(X)
                        pred_loss = loss_fn(pred, y) / len(X)
                        optimizer.zero_grad()
                        pred_loss.backward()

                        for n, p in model.named_parameters():
                            if p.requires_grad:
                                n = n.replace('.', '__')
                                if p.grad is not None:
                                    model.register_buffer('{}_grad'.format(n), p.grad)
                        
                    # Model
                    pred = model(X)
                    pred_loss = loss_fn(pred, y) / len(X)
                    surr_loss = model.surrogate_loss()
                    loss = pred_loss + conf.si_c * surr_loss

                    # GD
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Updating W & p_old
                    for n, p in model.named_parameters():
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            if p.grad is not None:
                                if conf.lseg:
                                    seg_grad = getattr(model, '{}_grad'.format(n))
                                    W[n].add_(-seg_grad*(p.detach()-p_old[n]))
                                else:
                                    W[n].add_(-p.grad*(p.detach()-p_old[n]))
                            p_old[n] = p.detach().clone()

                    # Logging
                    msg1 = 'Iter {}/{}, Pred Loss {:.4f}, '.format((i+1), len(client_train_loader), pred_loss.item())
                    msg2 = 'Iter {}/{}, Surr Loss {:.4f}, '.format((i+1), len(client_train_loader), surr_loss.item())
                    msg3 = 'Iter {}/{}, Overall Loss {:.4f}, '.format((i+1), len(client_train_loader), loss.item())
                    logging.info(msg1)        
                    logging.info(msg2)        
                    logging.info(msg3)        
                    
                    pred_train_loss += pred_loss.item()
                    surr_train_loss += surr_loss.item()
                    train_loss += loss.item()

                pred_train_loss /= len(client_train_loader)
                surr_train_loss /= len(client_train_loader)
                train_loss /= len(client_train_loader)
             
               
                if conf.bacc: 
                    # Recording (bacc == True) 
                    val_records = []
                    overall_val_loss, overall_val_acc = 0, 0
                    overall_pred_list, overall_label_list = [], []
                    for c in range(conf.n_clients):
                        val_loss, val_acc, bacc_results = train.val_loop(test_dataloaders[c], model, loss_fn, bacc=True)
                        bacc, pred_list, label_list = bacc_results
                        val_records.append([val_loss, val_acc, bacc])
                        overall_val_loss += val_loss
                        overall_val_acc += val_acc
                        overall_pred_list += pred_list
                        overall_label_list += label_list
                    overall_val_loss /= conf.n_clients
                    overall_val_acc /= conf.n_clients
                    overall_bacc = train.compute_bacc(overall_pred_list, overall_label_list, n_classes=max(overall_label_list)+1)
                    train.tensorboard_record_bacc(pred_train_loss, surr_train_loss, train_loss, val_records, 
                                                  overall_val_loss, overall_val_acc, overall_bacc, writer, step, k)
                else:
                    # Recording (bacc == False) 
                    val_records = []
                    overall_val_loss, overall_val_acc = 0, 0
                    for c in range(conf.n_clients):
                        val_loss, val_acc = train.val_loop(test_dataloaders[c], model, loss_fn, bacc=False)
                        val_records.append([val_loss, val_acc])
                        overall_val_loss += val_loss
                        overall_val_acc += val_acc
                    overall_val_loss /= conf.n_clients
                    overall_val_acc /= conf.n_clients
                    train.tensorboard_record_si(pred_train_loss, surr_train_loss, train_loss, val_records, 
                                                overall_val_loss, overall_val_acc, writer, step, k)


                step += 1

            # Update Omega
            model.update_omega(W)
    

    
if __name__ == "__main__":
    conf = const.give_si_args() 
    run(conf)

    
