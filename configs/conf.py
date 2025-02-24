
import argparse
import numpy as np 

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def give_si_args():
    parser = argparse.ArgumentParser(description='VAE-Reconstruction')
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--arch", type=str, default="mlp")
    parser.add_argument("--mode", type=str, default="queue")
    parser.add_argument("--mlp_hs", default=256, type=int)    
    parser.add_argument("--trans_pretrained", default=False, type=str2bool)
    parser.add_argument("--lseg", default=False, type=str2bool)
    parser.add_argument("--bacc", default=False, type=str2bool)
    parser.add_argument("--n_rounds", default=100, type=int)    
    parser.add_argument("--n_epochs", default=4, type=int)    
    parser.add_argument('--save_path', type=str, default="../output/tmp/")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--si_c", default=0.1, type=float)
    parser.add_argument("--epsilon", default=0.001, type=float)
    parser.add_argument("--alpha", default=0.0, type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num_workers", default=4, type=int)    
    parser.add_argument("--seed_use", default=1234, type=int)    
    parser.add_argument("--n_clients", default=4, type=int)
    parser.add_argument("--shuffle", default=True, type=str2bool)
    parser.add_argument("--drop_last", default=True, type=str2bool)
    parser.add_argument("--image_path", type=str, default="../image_dataset/")
    parser.add_argument("--test_ratio", default=0.2, type=float)
    parser.add_argument("--non_iid_alpha", default=100, type=float)

    return parser.parse_args()


def give_fed_args():
    parser = argparse.ArgumentParser(description='VAE-Reconstruction')
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--arch", type=str, default="mlp")
    parser.add_argument("--mlp_hs", default=256, type=int)    
    parser.add_argument("--n_rounds", default=100, type=int)    
    parser.add_argument("--n_epochs", default=4, type=int)    
    parser.add_argument("--non_iid_alpha", default=100, type=float)
    parser.add_argument("--trans_pretrained", default=False, type=str2bool)
    parser.add_argument("--test_ratio", default=0.2, type=float)
    parser.add_argument('--save_path', type=str, default="../output/tmp/")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num_workers", default=4, type=int)    
    parser.add_argument("--bacc", default=False, type=str2bool)
    parser.add_argument("--seed_use", default=1234, type=int)    
    parser.add_argument("--n_clients", default=4, type=int)
    parser.add_argument("--shuffle", default=True, type=str2bool)
    parser.add_argument("--drop_last", default=True, type=str2bool)
    parser.add_argument("--image_path", type=str, default="../image_dataset/")

    return parser.parse_args()


def give_args():
    parser = argparse.ArgumentParser(description='VAE-Reconstruction')
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument("--n_rounds", default=100, type=int)    
    parser.add_argument("--n_epochs", default=4, type=int)    
    parser.add_argument("--mlp_hs", default=256, type=int)    
    parser.add_argument('--save_path', type=str, default="../output/tmp/")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--test_ratio", default=0.2, type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num_workers", default=4, type=int)    
    parser.add_argument("--seed_use", default=1234, type=int)    
    parser.add_argument("--n_clients", default=4, type=int)
    parser.add_argument("--non_iid_alpha", default=100, type=float)
    parser.add_argument("--pn_normalize", default=True, type=str2bool)
    parser.add_argument("--apply_transform", default=True, type=str2bool)
    parser.add_argument("--shuffle", default=True, type=str2bool)
    parser.add_argument("--drop_last", default=True, type=str2bool)
    parser.add_argument("--image_path", type=str, default="../image_dataset/")

    return parser.parse_args()


