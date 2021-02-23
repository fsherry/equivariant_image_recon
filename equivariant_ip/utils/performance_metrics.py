import torch
from skimage.metrics import structural_similarity


def _abs(ims):
    return torch.sqrt(torch.sum(ims.detach().cpu()**2, dim=1))


def psnr(refs, ims):
    abs_refs, abs_ims = _abs(refs), _abs(ims)
    mse = torch.sum((abs_refs - abs_ims)**2,
                    dim=(1, 2)) / (abs_refs.shape[1] * abs_refs.shape[2])
    ps = torch.max(abs_refs.view(abs_refs.shape[0], -1)**2, dim=1).values
    return 10 * torch.log10(ps / mse)


def ssim(refs, ims, full=False):
    abs_refs, abs_ims = _abs(refs), _abs(ims)
    ssims = [
        structural_similarity(abs_refs[i, ...].numpy(),
                              abs_ims[i, ...].numpy(),
                              full=full) for i in range(refs.shape[0])
    ]
    if full:
        ssims, ssim_ims = list(zip(*ssims))
        return torch.tensor(ssims), torch.stack(
            [torch.tensor(im) for im in ssim_ims], dim=0)
    else:
        return torch.tensor(ssims)
