from equivariant_ip.utils.performance_metrics import psnr, ssim

from math import inf
import torch


def validate(model, val_dataloader):
    model.train()
    with torch.no_grad():
        recons = []
        psnrs = []
        ssims = []
        for x, y in val_dataloader:
            recon = model(y).cpu()
            recons.append(recon)
            psnrs.append(psnr(x.cpu(), recon))
            ssims.append(ssim(x.cpu(), recon))
    model.eval()
    recons = torch.cat(recons, dim=0)
    psnrs = torch.cat(psnrs, dim=0)
    ssims = torch.cat(ssims, dim=0)
    return ({"psnrs": psnrs, "ssims": ssims}, recons)


def train_one_epoch(model, optimiser, train_dataloader, callback=None):
    epoch_losses = []
    model.train()
    for i, (x, y) in enumerate(train_dataloader):
        optimiser.zero_grad()
        recon = model(y)
        loss = 0.5 * torch.mean((recon - x)**2)
        loss.backward()
        optimiser.step()
        epoch_losses.append(loss.item())
        print(loss.item())
        if callback is not None:
            callback(i, loss.item())
    return epoch_losses


def train_full(
    model,
    N_epochs,
    optimiser,
    train_dataloader,
    val_dataloader,
    val_interval=1,
    save_func=None,
    scheduler=None,
    checkpoint=None,
):
    images = None
    if checkpoint is not None:
        losses = checkpoint["history"]["losses"]
        validation_numbers = checkpoint["history"]["validation_numbers"]
        best_ssim = (torch.stack(
            checkpoint["history"]["validation_numbers"]["ssims"]).median(
                dim=1).values.max())
        best_model = checkpoint[
            "best_model"] if "best_model" in checkpoint else None
        model.load_state_dict(checkpoint["model"])
        optimiser.load_state_dict(checkpoint["optimiser"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        losses = []
        validation_numbers = {"psnrs": [], "ssims": []}
        best_ssim = -inf
        best_model = None
    for i in range(N_epochs):
        epoch_losses = train_one_epoch(model, optimiser, train_dataloader)
        losses += epoch_losses
        if (i + 1) % val_interval == 0:
            current_numbers, images = validate(model, val_dataloader)
            validation_numbers = {
                metric: validation_numbers[metric] + [current_numbers[metric]]
                for metric in ("psnrs", "ssims")
            }
            if validation_numbers["ssims"][-1].median() > best_ssim:
                best_model = model.state_dict()

            if scheduler is not None:
                if isinstance(scheduler,
                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(validation_numbers["ssim"]["mean"][-1])
                else:
                    scheduler.step()

            checkpoint = {
                "history": {
                    "losses": losses,
                    "validation_numbers": validation_numbers,
                    "images": images,
                },
                "model":
                model.state_dict(),
                "best_model":
                best_model,
                "optimiser":
                optimiser.state_dict(),
                "scheduler":
                scheduler.state_dict() if scheduler is not None else None,
            }
            if save_func is not None:
                device = next(model.parameters()).device
                save_func("checkpoint", checkpoint)

    res = {
        "history": {
            "losses": losses,
            "validation_numbers": validation_numbers,
            "images": images,
        },
        "model": model.state_dict(),
        "best_model": best_model,
        "optimiser": optimiser.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    if save_func is not None:
        save_func("final", res)
    return res
