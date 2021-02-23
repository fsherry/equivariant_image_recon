import torch


def grad(x):
    grad_x = torch.cat(
        (x[:, :, 1:, :] - x[:, :, :-1, :],
         torch.zeros(*x.shape[:2], 1, x.shape[3], device=x.device)),
        dim=2)
    grad_y = torch.cat((x[:, :, :, 1:] - x[:, :, :, :-1],
                        torch.zeros(*x.shape[:3], 1, device=x.device)),
                       dim=3)
    return grad_x, grad_y


def total_variation(x, eps=1e-8):
    grad_x, grad_y = grad(x)
    return torch.mean(torch.sqrt(eps +
                                 torch.sum(grad_x**2 + grad_y**2, dim=1)))


def tv_recon(y, forward_op, data_term, alpha=1e-2, x_init=None, N_it=1000):
    recon = forward_op.pseudoinverse(
        y).detach().clone() if x_init is None else x_init.detach().clone()
    recon.requires_grad = True
    optimiser = torch.optim.Adam([recon], lr=1e-3)
    for i in range(N_it):
        optimiser.zero_grad()
        loss = data_term(recon, y, forward_op) + alpha * total_variation(recon)
        loss.backward()
        optimiser.step()
    recon.requires_grad = False
    return recon.detach()
