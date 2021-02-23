from .ct.forward_operators import PoissonForwardOperator

import torch


def ls_data_term(x, y, forward_operator):
    return 0.5 * torch.mean((forward_operator(x) - y)**2)


def ls_data_term_grad(x, y, forward_operator):
    return forward_operator.vec_jac_prod(x, forward_operator(x) - y)


def poisson_data_term(x, y, forward_operator):
    #dropping the additive constant corresponding to mean(log(y!))
    predicted_counts = forward_operator(x)
    return torch.mean(predicted_counts - y * torch.log(predicted_counts))


def poisson_data_term_grad(x, y, forward_operator):
    predicted_counts = forward_operator(x)
    #    return forward_operator.vec_jac_prod(x, 1. - y / predicted_counts, predicted_counts)
    # equivalently, and avoiding potential divide by zero errors
    grad = forward_operator.vec_jac_prod(x, predicted_counts - y, 1.)
    # divide by N_in and mu_max to bring the dynamic range into a useable setting for a neural network with the "usual" sized weights
    if isinstance(forward_operator, PoissonForwardOperator):
        grad /= forward_operator.N_in * forward_operator.mu_max
    return grad
