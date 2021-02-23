from skimage.transform import rotate
import torch


def rotate_ims(ims, angles, **kwargs):
    dev = ims.device
    rot_ims = [[
        torch.tensor(
            rotate(ims[i, c, ...].detach().cpu(), angles[i].detach().cpu(),
                   **kwargs)) for c in range(ims.shape[1])
    ] for i in range(ims.shape[0])]
    return torch.stack([torch.stack(im_channels)
                        for im_channels in rot_ims]).to(dev)
