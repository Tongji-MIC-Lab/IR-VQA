import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import sys
import os
import time

from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import train
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=-1)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='block')
    parser.add_argument('--cp_name', type=str, default='bu')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--ls_neg', type=float, default=1.0)
    parser.add_argument('--ls_flat', type=float, default=1.0)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = True
    dictionary = Dictionary.load_from_file('root_path/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary, args)
    val_dset = VQAFeatureDataset('val', dictionary, args)
    
    if 'online' in args.cp_name: 
            trainval_dset = ConcatDataset([train_dset, val_dset])
            train_loader = DataLoader(trainval_dset, args.batch_size, shuffle=True, num_workers=0)
            eval_loader = None
    else:
        train_loader = DataLoader(train_dset, args.batch_size, shuffle=True, num_workers=0)
        eval_loader = DataLoader(val_dset, args.batch_size, shuffle=False, num_workers=0)

    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args).cuda()
    model.nn_w_emb.init_embedding('root_path/glove6b_init_300d.npy')
    model = model.cuda()
    train(model, train_loader, eval_loader, args)
