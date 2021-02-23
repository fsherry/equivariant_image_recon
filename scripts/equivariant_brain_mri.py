from equivariant_ip.mri.forward_operators import SingleCoilMRIForwardOperator
from equivariant_ip.data_terms import ls_data_term_grad
from equivariant_ip.proximal_blocks.primal_equivariant import PrimalProximalEquivariant
from equivariant_ip.reconstruction_methods.lfb import LearnedForwardBackward
from equivariant_ip.utils.datasets import DatasetFromMemmap, ProcessDataset, CartesianProductDataset
from equivariant_ip.utils.train_validate import train_full, validate
from equivariant_ip.utils.rotate import rotate_ims

import argparse
import e2cnn
from math import ceil
import numpy as np
import os
import json
import torch
import uuid

parser = argparse.ArgumentParser(
    description='Train equivariant learned forward-backward method')
parser.add_argument('--name', type=str, default='lfb')
parser.add_argument('--N_epochs', type=int, default=100)
parser.add_argument('--N_train', type=int, default=100)
parser.add_argument('--N_rot', type=int, default=4)
parser.add_argument('--depth', type=int, default=8)
parser.add_argument('--n_memory', type=int, default=5)
parser.add_argument('--filter_size', type=int, default=3)
parser.add_argument('--n_res_blocks', type=int, default=1)
parser.add_argument('--channels', type=int, default=24)
parser.add_argument('--data_path', type=str)
parser.add_argument('--save_path', type=str)
parser.add_argument('--checkpoint', type=str)
args = parser.parse_args()

checkpoint = torch.load(
    args.checkpoint) if args.checkpoint is not None else None
name = '-'.join(
    args.checkpoint.split('/')[-1].split('-')
    [:-1]) + '-cont' if args.checkpoint is not None else args.name + '-' + str(
        uuid.uuid4())
device = 0 if torch.cuda.is_available() else 'cpu'
N_epochs = args.N_epochs

base_space = e2cnn.gspaces.Rot2dOnR2(args.N_rot)
channels = 96 // args.N_rot
feat_type = e2cnn.nn.FieldType(base_space,
                               channels * [base_space.regular_repr])

mask = torch.tensor(np.load(os.path.join(args.data_path, 'line_mask.npy')),
                    device=device)
forward_operator = SingleCoilMRIForwardOperator(mask)

with open(os.path.join(args.data_path, 'params.json'), 'rb') as in_file:
    params = json.load(in_file)


def data_term_grad(x, y):
    return ls_data_term_grad(x, y, forward_operator)


train_ims = DatasetFromMemmap(os.path.join(args.data_path, 'selected_ims.dat'),
                              shape=params['x_shape'],
                              N_samples=args.N_train,
                              device=device)
train_ys = DatasetFromMemmap(os.path.join(args.data_path, 'measurements.dat'),
                             shape=params['y_shape'],
                             N_samples=args.N_train,
                             device=device)
train_data = CartesianProductDataset(train_ims, train_ys)

validation_ims = DatasetFromMemmap(os.path.join(args.data_path,
                                                'selected_ims.dat'),
                                   shape=params['x_shape'],
                                   N_samples=200,
                                   offset=10000,
                                   device=device)
validation_ys = DatasetFromMemmap(os.path.join(args.data_path,
                                               'measurements.dat'),
                                  shape=params['y_shape'],
                                  N_samples=200,
                                  offset=10000,
                                  device=device)
validation_data = CartesianProductDataset(validation_ims, validation_ys)

train_dataloader = torch.utils.data.DataLoader(train_data,
                                               batch_size=1,
                                               shuffle=True)
val_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=5)

conv_params = {'kernel_size': 3, 'padding': 1, 'padding_mode': 'replicate'}


def proximal_constructor():
    return PrimalProximalEquivariant(conv_params,
                                     in_channels=2,
                                     n_memory=args.n_memory,
                                     feat_type_intermed=feat_type,
                                     init_as_id=False)


class Model(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.lfb = LearnedForwardBackward(data_term_grad, proximal_constructor,
                                          args.depth)

    def forward(self, y):
        return self.lfb(
            torch.zeros(y.shape[0], *params['x_shape'][1:4], device=y.device),
            y)


model = Model().to(device)
for block in model.lfb.prox_blocks:
    for res_block in block.middle_part:
        res_block._set_equal_to_identity()

optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)


def save_func(str, res):
    filename = os.path.join(args.save_path, '{}-{}'.format(name, str))
    torch.save(res, filename)


val_interval = ceil(5000 / args.N_train)
train_full(model,
           N_epochs,
           optimiser,
           train_dataloader,
           val_dataloader,
           val_interval,
           save_func=save_func,
           checkpoint=checkpoint)
