from equivariant_ip.utils.performance_metrics import psnr, ssim

import torch


def validate(model, val_dataloader):
    model.train()
    with torch.no_grad():
        psnrs = []
        ssims = []
        for x, y in val_dataloader:
            recon = model(y).cpu()
            psnrs.append(psnr(x.cpu(), recon))
            ssims.append(ssim(x.cpu(), recon))
    model.eval()
    psnrs = torch.cat(psnrs, dim=0)
    ssims = torch.cat(ssims, dim=0)
    return {'psnr': psnrs, 'ssim': ssims}
