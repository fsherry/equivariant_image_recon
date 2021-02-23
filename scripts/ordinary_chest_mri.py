from equivariant_ip.mri.forward_operators import SingleCoilMRIForwardOperator
from equivariant_ip.data_terms import ls_data_term_grad
from equivariant_ip.proximal_blocks.primal_ordinary import PrimalProximalOrdinary
from equivariant_ip.reconstruction_methods.lfb import LearnedForwardBackward
from equivariant_ip.utils.datasets import DatasetFromMemmap, ProcessDataset
from equivariant_ip.utils.train_validate import train_full
from equivariant_ip.utils.rotate import rotate_ims

import argparse
import astra
from math import sqrt, pi, inf, ceil, floor
import numpy as np
import os
import pickle
import sys
import torch
import uuid

parser = argparse.ArgumentParser(description='Train ordinary learned forward-backward method')
parser.add_argument('--save_path', type=str)
parser.add_argument('--name', type=str, default='lfb')
parser.add_argument('--N_epochs', type=int, default=100)
parser.add_argument('--N_train', type=int, default=100)
parser.add_argument('--depth', type=int, default=5)
parser.add_argument('--n_memory', type=int, default=5)
parser.add_argument('--filter_size', type=int, default=3)
parser.add_argument('--n_res_blocks', type=int, default=1)
parser.add_argument('--channels', type=int, default=96)
parser.add_argument('--checkpoint', type=str)
args = parser.parse_args()


checkpoint = torch.load(args.checkpoint) if args.checkpoint is not None else None
name = '-'.join(args.checkpoint.split('/')[-1].split('-')[:-1]) + '-cont' if args.checkpoint is not None else args.name + '-' + str(uuid.uuid4())
device = 0 if torch.cuda.is_available() else 'cpu'
N_epochs = args.N_epochs

mask = torch.tensor(np.load('/nfs/st01/hpc-cmih-cbs31/fs436/FastMRI/memmap_files/line_mask512.npy'), device=device)
forward_operator = SingleCoilMRIForwardOperator(mask)

def data_term_grad(x, y):
    return ls_data_term_grad(x, y, forward_operator)

import json

with open('/nfs/st01/hpc-cmih-cbs31/fs436/LIDC-IDRI/train_params.json', 'rb') as in_file:
    train_params = json.load(in_file)
    train_shape = (train_params['N'], *train_params['x_shape'])
with open('/nfs/st01/hpc-cmih-cbs31/fs436/LIDC-IDRI/val_params.json', 'rb') as in_file:
    val_params = json.load(in_file)
    val_shape = (val_params['N'], *val_params['x_shape'])


def process_fn(i, x):
    angle = torch.rand(1) * 360 
    im = rotate_ims(x.view(1, *x.shape), angle)
    im = torch.cat((im, torch.zeros_like(im)), dim=1)
    y = mask.view(1, *mask.shape) * forward_operator.simulate_kspace(im, noise_level=0.05)
    return (im[0, ...], y[0, ...])

train_data = ProcessDataset(DatasetFromMemmap('/nfs/st01/hpc-cmih-cbs31/fs436/LIDC-IDRI/train_gt.dat', train_shape,
                               N_train=args.N_train, device=device), process_fn)
validation_data = ProcessDataset(DatasetFromMemmap('/nfs/st01/hpc-cmih-cbs31/fs436/LIDC-IDRI/val_gt.dat', val_shape,
                                    device=device), process_fn) 

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=5)

conv_params = {'kernel_size': 3, 'padding': 1, 'padding_mode': 'replicate'}
proximal_constructor = lambda : PrimalProximalOrdinary(conv_params, in_channels=2, init_as_id=False)
class Model(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.lfb = LearnedForwardBackward(data_term_grad, proximal_constructor, args.depth)
    def forward(self, y):
        return self.lfb(torch.zeros(y.shape[0], 2, 512, 512, device=y.device), y)

model = Model().to(device)
for block in model.lfb.prox_blocks:
    for res_block in block.middle_part:
        res_block._set_equal_to_identity()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)

def save_func(str, res):
    filename = os.path.join(args.save_path, '{}-{}'.format(name, str))
    torch.save(res, filename)
    
val_interval = ceil(5000 / args.N_train)
train_full(model, N_epochs, optimiser, train_dataloader, val_dataloader, val_interval, save_func=save_func, checkpoint=checkpoint)
