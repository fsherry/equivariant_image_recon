from math import prod

import json
import numpy as np
import pickle
import torch


class DatasetFromNumpyFile(torch.utils.data.Dataset):
    def __init__(self, data_pkl, device=0):
        with open(data_pkl, "rb") as in_file:
            data = pickle.load(in_file)
        self.xs = data["xs"]
        self.ys = data["ys"]
        self.N = self.xs.shape[0]
        self.device = device

    def __getitem__(self, i):
        return (
            torch.tensor(self.xs[i, ...], device=self.device),
            torch.tensor(self.ys[i, ...], device=self.device),
        )

    def __len__(self):
        return self.N


class DatasetFromMemmap(torch.utils.data.Dataset):
    def __init__(self, memmap_file, shape, N_samples=None, offset=0, device=0):
        self.N = shape[0]
        self.N_samples = self.N if N_samples is None else min(
            N_samples, self.N)
        self.N_samples = shape[0] - offset if offset + self.N_samples > shape[
            0] else self.N_samples
        byte_offset = 4 * offset * prod(shape[1:])
        self.xs = np.memmap(memmap_file,
                            mode="r",
                            dtype=np.float32,
                            offset=byte_offset,
                            shape=(self.N_samples, *shape[1:]))
        self.device = device

    def __getitem__(self, i):
        if i >= self.N_samples:
            raise IndexError(
                "Index {} is out of range for data of size {}".format(
                    i, self.N_samples))
        return torch.tensor(self.xs[i, ...], device=self.device)

    def __len__(self):
        return self.N_samples


class CartesianProductDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(dataset[i] for dataset in self.datasets)

    def __len__(self):
        return min(len(dataset) for dataset in self.datasets)


class ProcessDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, process_fn):
        self.dataset = dataset
        self.process_fn = process_fn

    def __getitem__(self, i):
        point = self.dataset[i]
        return self.process_fn(i, point)

    def __len__(self):
        return len(self.dataset)
