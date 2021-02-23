import torch

class LearnedForwardBackward(torch.nn.Module):
    def __init__(self, grad_data_term, prox_block_constructor, depth=8):
        super().__init__()
        self.grad_data_term = grad_data_term
        self.prox_blocks = torch.nn.ModuleList([prox_block_constructor() for _ in range(depth)])
        self.n_memory = self.prox_blocks[0].n_memory

    def forward(self, x_init, y):
        x = x_init.clone()
        s = torch.zeros(x_init.shape[0], self.n_memory, *x_init.shape[2:], device=x.device)
        for prox_block in self.prox_blocks:
            (x, s) = prox_block(x, self.grad_data_term(x, y), s)
        return x
