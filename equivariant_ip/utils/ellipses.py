import torch

from math import pi


def ellipses(N, N_ellipses=10, N_pix=200, device=0):
    background_angles = 0.25 * pi * torch.randn(N, 1,
                                                device=device) - 0.125 * pi
    foreground_angles = 2 * pi * torch.randn(N, N_ellipses, device=device)
    angles = torch.cat((background_angles, foreground_angles), dim=1)

    background_intensities = 0.1 * torch.ones(N, 1, device=device)
    foreground_intensities = torch.rand(N, N_ellipses, device=device).view(
        N, N_ellipses) / 5
    intensities = torch.cat((background_intensities, foreground_intensities),
                            dim=1).view(N, N_ellipses + 1, 1,
                                        1).expand(N, N_ellipses + 1, N_pix,
                                                  N_pix)

    background_axes = 0.38 + 0.1 * torch.rand(N, 1, 1, 1, 2, device=device)
    foreground_axes = 0.05 + 0.15 * torch.rand(
        N, N_ellipses, 1, 1, 2, device=device)
    ellipse_axes = torch.cat((background_axes, foreground_axes),
                             dim=1).expand(N, N_ellipses + 1, N_pix, N_pix, 2)

    background_offsets = 0.02 * torch.rand(N, 1, 1, 1, 2, device=device)
    foreground_offsets = 0.4 * torch.rand(
        N, N_ellipses, 1, 1, 2, device=device)
    offsets = torch.cat((background_offsets, foreground_offsets),
                        dim=1).expand(N, N_ellipses + 1, N_pix, N_pix, 2)

    meshgrid_x, meshgrid_y = torch.meshgrid((torch.linspace(-0.5,
                                                            0.5,
                                                            N_pix,
                                                            device=device),
                                             torch.linspace(-0.5,
                                                            0.5,
                                                            N_pix,
                                                            device=device)))
    meshgrid_x, meshgrid_y = meshgrid_x.view(1, 1, N_pix, N_pix).expand(
        N, N_ellipses + 1, N_pix,
        N_pix), meshgrid_y.view(1, 1, N_pix,
                                N_pix).expand(N, N_ellipses + 1, N_pix, N_pix)
    meshgrid_xy = torch.stack((meshgrid_x, meshgrid_y), dim=-1)
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    rot_matrices = torch.stack(
        (torch.stack((cos_angles, -sin_angles),
                     dim=2), torch.stack((sin_angles, cos_angles), dim=2)),
        dim=3)

    rot_grid = torch.einsum('hijkl, hinl -> hijkn', meshgrid_xy, rot_matrices)
    background_cutoff = (torch.sum(
        ((offsets[:, 0, :, :] + rot_grid[:, 0, :, :]) /
         ellipse_axes[:, 0, :, :])**2,
        dim=3) <= 1).view(offsets.shape[0], 1, *offsets.shape[2:4])
    im = torch.sum(intensities * background_cutoff * (torch.sum(
        ((offsets + rot_grid) / ellipse_axes)**2, dim=4) <= 1.),
                   dim=1)
    return im


class RandomEllipsesPhotonCountsDataset(torch.utils.data.Dataset):
    def __init__(self,
                 N,
                 forward_operator,
                 N_ellipses=15,
                 N_pix=200,
                 device=0):
        self.N = N
        self.N_ellipses = N_ellipses
        self.N_pix = N_pix
        self.device = device
        self.forward_operator = forward_operator

    def __getitem__(self, i):
        x = ellipses(1, self.N_ellipses, self.N_pix, self.device)
        y = self.forward_operator.simulate_counts(x)
        return (x.view(*x.shape[1:]), y.view(*y.shape[1:]))

    def __len__(self):
        return self.N


class RandomEllipsesSinogramsDataset(torch.utils.data.Dataset):
    def __init__(self,
                 N,
                 forward_operator,
                 N_ellipses=15,
                 N_pix=200,
                 device=0):
        self.N = N
        self.N_ellipses = N_ellipses
        self.N_pix = N_pix
        self.device = device
        self.forward_operator = forward_operator

    def __getitem__(self, i):
        x = ellipses(1, self.N_ellipses, self.N_pix, self.device)
        y = self.forward_operator.simulate_sinograms(x)
        return (x.view(*x.shape[1:]), y.view(*y.shape[1:]))

    def __len__(self):
        return self.N
