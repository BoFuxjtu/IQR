import numpy as np
import torch
import math
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from data.utils import MultiEnvDataset
from methods.modules import *

# kernel loss
def uniform_kernel_loss(u, h, tau):
    """
    Uniform kernel with quantile (τ) smoothing
    """
    U_u = (((u/h) ** 2) / 2 + 1 / 2) * (torch.abs(u) <= h).float() + torch.abs(u) * (torch.abs(u) > h).float()
    return (h / 2) * U_u + (tau - 1 / 2) * u

# Gaussian kernel
def gaussian_kernel_loss(u, h, tau):
    """
    Gaussian kernel with quantile (τ) smoothing
    """
    # expectation of folded normal distribution
    Phi = 0.5 * (1 + torch.erf(u / (h * math.sqrt(2))))
    expected_abs_gu = (
            math.sqrt(2 / math.pi) * h * torch.exp(-u ** 2 / (2 * h ** 2))
            + u * (2 * Phi - 1)
    )

    return (1 / 2) * expected_abs_gu + (tau - 1 / 2) * u

# Laplacian kernel
def laplacian_kernel_loss(u, h, tau):
    """
    Laplacian kernel with quantile (τ) smoothing
    """
    rho_tau = u * (tau - (u < 0).float())

    return rho_tau + h * torch.exp(-torch.abs(u) / h) / 2

# Logistic kernel
def logistic_kernel_loss(u, h, tau):
    """
    Logistic kernel with quantile (τ) smoothing
    """
    return (tau * u + h * torch.log1p(torch.exp(-u / h)))

# Epanechnikov kernel
def epanechnikov_kernel_loss(u, h, tau):
    """
    Epanechnikov kernel with quantile (τ) smoothing

    """
    E_u = ((3 * (u/h) ** 2) / 4 - (u/h) ** 4 / 8 + 3 / 8) * (torch.abs((u/h)) <= 1) + torch.abs((u/h)) * (torch.abs((u/h)) > 1)
    return (h / 2) * E_u + (tau - 1 / 2) * u

# kernel gradient

# uniform grad
def uniform_grad(u, h):
    uh = u / h
    return torch.where(
        uh <= -1, torch.zeros_like(u),
        torch.where(uh >= 1, torch.ones_like(u),
                    0.5 * (1 + uh))
    )

# gaussian grad
def gaussian_grad(u, h):
    return 0.5 * (1 + torch.erf(u / (h * math.sqrt(2))))

# Laplacian kernel
def laplacian_grad(u, h):
    return torch.where(
        u < 0,
        0.5 * torch.exp(u / h),
        1 - 0.5 * torch.exp(-u / h)
    )

# Logistic grad
def logistic_grad(u, h):
    return torch.sigmoid(u / h)

# Epanechnikov grad
def epanechnikov_grad(u, h):
    uh = u / h
    return torch.where(
        uh <= -1, torch.zeros_like(u),
        torch.where(
            uh >= 1, torch.ones_like(u),
            0.5 + 0.75 * uh - 0.25 * uh ** 3
        )
    )

def ker_grad(u, h=0.1, kernel_type="gaussian"):
    if kernel_type == 'gaussian':
        return gaussian_grad(u, h)
    elif kernel_type == 'logistic':
        return logistic_grad(u, h)
    elif kernel_type == 'uniform':
        return uniform_grad(u, h)
    elif kernel_type == 'epanechnikov':
        return epanechnikov_grad(u, h)
    elif kernel_type == 'laplacian':
        return laplacian_grad(u, h)
    else:
        raise ValueError("No such kernel!")

# Other kernels

# Choosing kernels
def smoothed_quantile_loss(u, kernel_type="gaussian", h=0.1, tau=0.5):
    """
    Compute the smoothed loss using the selected kernel type and quantile (τ)
    u: residual (ys - outs), kernel_type: type of smoothing kernel
    """
    if kernel_type == "uniform":
        return uniform_kernel_loss(u, h, tau)
    elif kernel_type == "gaussian":
        return gaussian_kernel_loss(u, h, tau)
    elif kernel_type == "laplacian":
        return laplacian_kernel_loss(u, h, tau)
    elif kernel_type == "logistic":
        return logistic_kernel_loss(u, h, tau)
    elif kernel_type == "epanechnikov":
        return epanechnikov_kernel_loss(u, h, tau)
    else:
        raise ValueError(
            "Invalid kernel type. Choose from 'uniform', 'gaussian', 'laplacian', 'logistic', 'epanechnikov'.")


# SIQR model
class SIQRLinearModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(SIQRLinearModel, self).__init__()
        self.linear = torch.nn.Linear(in_features=input_dim, out_features=1, bias=True)
        self.x_mean = torch.tensor(np.zeros((1, input_dim))).float()
        self.x_std = torch.tensor(np.ones((1, input_dim))).float()

    def standardize(self, train_x):
        self.x_mean = torch.tensor(np.mean(train_x, 0, keepdims=True)).float()
        self.x_std = torch.tensor(np.std(train_x, 0, keepdims=True)).float()

    def forward(self, x):
        x = (x - self.x_mean) / self.x_std
        y = self.linear(x)
        return y


# SIQR
def siqr_sgd(features, responses, kernel_type="gaussian", tau_Gumbel=0.5, tau_quantile=0.5, hyper_gamma=20, learning_rate=1e-3, niters=50000,
             niters_d=2, niters_g=1, offset=-2, batch_size=32, mask=None, final_temp=0.05, iter_save=100, log=False,h=None,intercept=False):
    """
    Implementation of Kernel-Smoothed Invariant Quantile Regression (KS-IQR) with Gumbel Approximation

    Parameters
    ----------
    features : list
        List of numpy arrays (n_k, p) representing explanatory variables
    responses : list
        List of numpy arrays (n_k, 1) representing response variables
    kernel_type : str
        The type of kernel to use for smoothing ('uniform', 'gaussian', 'laplacian', 'logistic', 'epanechnikov')
    tau_quantile : float
        The quantile (τ) for quantile regression
    tau_Gumbel : float
        The Gumbel distribution parameter (for the Gumbel gate)
    ...

    Returns
    -------
    A dictionary containing the weights, gates, and loss history during training
    """
    num_envs = len(features)
    dim_x = np.shape(features[0])[1]
    n0 = np.shape(features[0])[0]
    if h is None:
        h = np.sqrt(tau_quantile * (1 - tau_quantile) * dim_x * hyper_gamma / (n0 * num_envs))
        #h = np.maximum(0.05, np.minimum(np.sqrt(tau_quantile * (1 - tau_quantile) * dim_x * hyper_gamma / (n0 * num_envs)), 0.2))

    model = SIQRLinearModel(dim_x)
    model_var = GumbelGate(dim_x, init_offset=offset, device='cpu')
    optimizer_var = optim.Adam(model_var.parameters(), lr=learning_rate)
    optimizer_g = optim.Adam(model.linear.parameters(), lr=learning_rate)

    dataset = MultiEnvDataset(features, responses)

    gate_rec, weight_rec, loss_rec = [], [], []

    # Start training loop
    for it in tqdm(range(niters)):
        tau_Gumbel = max(final_temp, tau_Gumbel * 0.993) if it % 100 == 0 else tau_Gumbel

        optimizer_var.zero_grad()
        optimizer_g.zero_grad()

        xs, ys = dataset.next_batch(batch_size)
        gate = model_var.generate_mask((1, tau_Gumbel))

        outs = [model(gate * x) for x in xs]

        # Use smoothed quantile loss based on the selected kernel and quantile (tau)
        loss_J = 0
        loss_R = 0
        for e in range(num_envs):
            loss_R_e = torch.mean(
                smoothed_quantile_loss(
                     ys[e] - outs[e],
                      kernel_type=kernel_type,
                    h=h,
                     tau=tau_quantile
                )
            )
            loss_R += loss_R_e
            grad_beta_e = torch.autograd.grad(
                loss_R_e,
                #model.linear.weight,
                [model.linear.weight, model.linear.bias],
                create_graph=True
            )[0].view(-1)
            loss_J += torch.sum((grad_beta_e ** 2) * gate)
        #loss_R = loss_R/ num_envs

        # Apply smoothed quantile loss using selected kernel and quantile tau

        loss = loss_R + hyper_gamma * loss_J

        loss.backward()

        optimizer_g.step()
        optimizer_var.step()

        # Save weights and logits
        if it % iter_save == 0:
            with torch.no_grad():
                weight = model.linear.weight.detach().cpu()
                logits = model_var.get_logits_numpy()
                gate_rec.append(sigmoid(logits))
                weight_rec.append(np.squeeze(weight.numpy() + 0.0))
            loss_rec.append(loss.item())

    ret = {'weight': weight_rec[-1] * sigmoid(logits),
           'weight_rec': np.array(weight_rec),
           'gate_rec': np.array(gate_rec),
           'loss_rec': np.array(loss_rec)}

    beta = weight_rec[-1] * sigmoid(logits)

    return beta