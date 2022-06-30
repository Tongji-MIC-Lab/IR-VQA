from __future__ import print_function

import errno
import os
import numpy as np
import json
import time
from PIL import Image
import torch
import torch.nn as nn


EPS = 1e-7


def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)


def assert_array_eq(real, expected):
    assert (np.abs(real-expected) < EPS).all(), \
        '%s (true) vs %s (expected)' % (real, expected)


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_imageid(folder):
    images = load_folder(folder, 'jpg')
    img_ids = set()
    for img in images:
        img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        img_ids.add(img_id)
    return img_ids


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def weights_init(m):
    """custom weights initialization."""
    cname = m.__class__
    if cname == nn.Linear or cname == nn.Conv2d or cname == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
    elif cname == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print('%s is not initialized.' % cname)


def init_net(net, net_file):
    if net_file:
        net.load_state_dict(torch.load(net_file))
    else:
        net.apply(weights_init)


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


class Logger_ini(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        #print(msg)

class Logger(object):
    def __init__(self, args):
        self.cp_name = args.cp_name
        self.exp_list_path = args.exp_list_path
        self.cp = {}
        self.cp['start_time'] = time.time()
        self.cp['args'] = args.__dict__
        self.cp['train_loss_list'] = []
        self.cp['train_ac_list'] = []
        self.cp['eval_loss_list'] = []
        self.cp['eval_ac_list'] = []

    def log(self, l_list):
        self.cp['train_loss_list'].append(l_list[1])
        self.cp['train_ac_list'].append(l_list[2])
        self.cp['eval_loss_list'].append(l_list[3])
        self.cp['eval_ac_list'].append(l_list[4])
        self.cp['last_time'] = time.time()
        self.cp['time_curEpoch'] = l_list[5]
        print(self.cp_name, '\t time consuming: %.2f' % (l_list[6]))
        print(('\t train_loss: %.2f, train_ac: %.2f' % (l_list[1], l_list[2])))
        print('\t val_loss: %.2f,eval_ac: %.2f (%.2f)' % (l_list[3], l_list[4], l_list[5]))
        json.dump(self.cp, open('./save/' + self.cp_name + '.json', 'w'))
        json.dump(self.cp, open(self.exp_list_path + "/" + self.cp_name + '.json', 'w'))
        
