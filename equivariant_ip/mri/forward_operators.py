from ..forward_operators import ForwardOperator

import torch


class SingleCoilMRIForwardOperator(ForwardOperator):
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, x, masks=None):
        if masks is None:
            y = self.mask.view(1, *self.mask.shape, 1) * torch.fft(
                x.permute(0, 2, 3, 1), signal_ndim=2, normalized=True)
        else:
            y = masks.view(*masks.shape, 1) * torch.fft(
                x.permute(0, 2, 3, 1), signal_ndim=2, normalized=True)
        return y.permute(0, 3, 1, 2)

    def vec_jac_prod(self, x, z, masks=None):
        if masks is None:
            w = torch.ifft(self.mask.view(1, *self.mask.shape, 1) *
                           z.permute(0, 2, 3, 1),
                           signal_ndim=2,
                           normalized=True)
        else:
            w = torch.ifft(masks.view(*masks.shape, 1) * z.permute(0, 2, 3, 1),
                           signal_ndim=2,
                           normalized=True)
        return w.permute(0, 3, 1, 2)

    def pseudoinverse(self, y, masks=None):
        if masks is None:
            w = torch.ifft(self.mask.view(1, *self.mask.shape, 1) *
                           y.permute(0, 2, 3, 1),
                           signal_ndim=2,
                           normalized=True)
        else:
            w = torch.ifft(masks.view(*masks.shape, 1) * y.permute(0, 2, 3, 1),
                           signal_ndim=2,
                           normalized=True)
        return w.permute(0, 3, 1, 2)

    def simulate_kspace(self, x, noise_level=0.05, masks=None):
        y = self(x, masks)
        if masks is None:
            y = self.mask.view(1, 1, *self.mask.shape) * (
                y + noise_level * torch.randn_like(y))
        else:
            y = masks.view(masks.shape[0], 1, *masks.shape[1:]) * (
                y + noise_level * torch.randn_like(y))
        return y
