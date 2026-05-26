import numpy as np
import torch
import math
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from data.utils import MultiEnvDataset
from methods.modules import *
from scipy.optimize import minimize
from methods.brute_force import pooled_least_squares
from regularization_path import hyper_gamma


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
        self.linear = torch.nn.Linear(in_features=input_dim, out_features=1, bias=False)
        self.x_mean = torch.tensor(np.zeros((1, input_dim))).float()
        self.x_std = torch.tensor(np.ones((1, input_dim))).float()

    def standardize(self, train_x):
        #self.x_mean = torch.tensor(np.mean(train_x, 0, keepdims=True)).float()
        #self.x_std = torch.tensor(np.std(train_x, 0, keepdims=True)).float()
        mean = np.mean(train_x, 0, keepdims=True)
        std = np.std(train_x, 0, keepdims=True)
        mean[0, 0] = 0.0
        std[0, 0] = 1.0
        self.x_mean = torch.tensor(mean).float()
        self.x_std = torch.tensor(std).float()

    def forward(self, x):
        x = (x - self.x_mean) / self.x_std
        y = self.linear(x)
        return y


# KS_IQR_Gumbel
def KS_IQR_Gumbel_sgd(features, responses, kernel_type="gaussian", tau_Gumbel=0.5, tau_quantile=0.5, hyper_gamma=20, learning_rate=1e-3, niters=50000,
             offset=-2, batch_size=32, mask=None, final_temp=0.05, iter_save=100, log=False,h=None,min_lr=5e-5):
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

    X = [
        np.hstack([np.ones((X.shape[0], 1)), X])
        for X in features
    ]

    model = SIQRLinearModel(dim_x+1)
    #model_var = GumbelGate(dim_x+1, init_offset=offset, device='cpu')
    model_var = GumbelGate(dim_x, init_offset=offset, device='cpu')


    optimizer_var = optim.Adam(model_var.parameters(), lr=learning_rate)
    optimizer_g = optim.Adam(model.linear.parameters(), lr=learning_rate)
    #optimizer_var = optim.Adam(model_var.parameters(), lr=learning_rate, betas=(0.6, 0.8), eps=1e-8)
    #optimizer_g = optim.Adam(model.linear.parameters(), lr=learning_rate, betas=(0.6, 0.8), eps=1e-8)

    dataset = MultiEnvDataset(X, responses)

    gate_rec, weight_rec, loss_rec = [], [], []

    # Start training loop
    for it in tqdm(range(niters)):
        #tau_Gumbel = max(final_temp, tau_Gumbel * 0.995) if it % 50 == 0 else tau_Gumbel
        if it % 50 == 0:
            tau_Gumbel = max(final_temp, tau_Gumbel * 0.9995)

            for param_group in optimizer_g.param_groups:
               param_group['lr'] = max(param_group['lr'] * 0.9995, min_lr)
               #param_group['lr'] *= 1


            for param_group in optimizer_var.param_groups:
                #['lr'] *= 1
                param_group['lr'] = max(param_group['lr'] * 0.9995, min_lr)

        optimizer_var.zero_grad()
        optimizer_g.zero_grad()

        xs, ys = dataset.next_batch(batch_size)

        #gate = model_var.generate_mask((1, tau_Gumbel))
        #outs = [model(gate * x) for x in xs]

        #logits0 = model_var.get_logits()
        #gate = torch.sigmoid(logits0)  # deterministic
        #outs = [model(gate * x) for x in xs]

        gate0 = model_var.generate_mask((1, tau_Gumbel)).squeeze(0)
        gate = torch.cat([torch.ones(1), gate0], dim=0)
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
            #grad_beta_e = torch.autograd.grad(
            #    loss_R_e,
            #    model.linear.weight,
            #    #[model.linear.weight, model.linear.bias],
            #    create_graph=True
            #)[0].view(-1)
            #loss_J += torch.sum((grad_beta_e ** 2) * gate)

            residual = ys[e] - outs[e]
            psi = ker_grad(-residual, h=h, kernel_type=kernel_type) - tau_quantile
            grad_beta_e = (psi.T @ xs[e]) / len(ys[e])
            grad_beta_e = grad_beta_e.view(-1)
            loss_J += torch.sum((grad_beta_e ** 2) * gate)
            #loss_J += torch.sum((grad_beta_e[1:] ** 2) * gate0)

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

    #beta_full = weight_rec[-1] * sigmoid(logits)
    gate_final = sigmoid(logits)
    gate_full = np.concatenate([[1.0], gate_final])
    beta_full = weight_rec[-1] * gate_full

    ret = {'beta_full': beta_full,
           'beta': beta_full[1:],
           'weight_rec': np.array(weight_rec),
           'gate_rec': np.array(gate_rec),
           'loss_rec': np.array(loss_rec)}

    #if intercept:
    #    return beta_full
    #else:
    #    return beta_full[1:]

    return ret

def KS_IQR_Gumbel_sgd2(features, responses, kernel_type="gaussian", tau_Gumbel=0.5, tau_quantile=0.5, hyper_gamma=20,
                      learning_rate=1e-3, niters=50000,
                      offset=-1, batch_size=32, mask=None, final_temp=0.05, iter_save=100, log=False, h=None,
                      min_lr=5e-5,
                      warmup_ratio=0.5, sparsity_weight=0,Gumbel_ratio=0.4):
    """
    Implementation of Kernel-Smoothed Invariant Quantile Regression (KS-IQR) with Gumbel Approximation

    Parameters
    ----------
    ...
    warmup_ratio : float
        warmup (default: 50%)
    sparsity_weight : float
        parameter (default: 0.01)
    """
    num_envs = len(features)
    dim_x = np.shape(features[0])[1]
    n0 = np.shape(features[0])[0]
    if h is None:
        h = np.sqrt(tau_quantile * (1 - tau_quantile) * dim_x * hyper_gamma / (n0 * num_envs))

    X = [
        np.hstack([np.ones((X.shape[0], 1)), X])
        for X in features
    ]

    model = SIQRLinearModel(dim_x + 1)

    # offset=-2 -> sigmoid(-2)≈0.12
    # offset=-5 -> sigmoid(-5)≈0.007 (more sparse initialization)

    model_var = GumbelGate(dim_x, init_offset=offset, device='cpu')
    #model_var = GumbelGate(dim_x + 1, init_offset=offset, device='cpu')

    optimizer_var = optim.Adam(model_var.parameters(), lr=learning_rate)
    optimizer_g = optim.Adam(model.linear.parameters(), lr=learning_rate)

    dataset = MultiEnvDataset(X, responses)
    gate_rec, weight_rec, loss_rec = [], [], []

    warmup_iters = int(niters * warmup_ratio)

    # Start training loop
    for it in tqdm(range(niters)):
        if it % 50 == 0:
            tau_Gumbel = max(final_temp, tau_Gumbel * 0.997)

            for param_group in optimizer_g.param_groups:
                param_group['lr'] = max(param_group['lr'] * 0.997, min_lr)

            for param_group in optimizer_var.param_groups:
                param_group['lr'] = max(param_group['lr'] * 0.997, min_lr)

        optimizer_var.zero_grad()
        optimizer_g.zero_grad()

        xs, ys = dataset.next_batch(batch_size)

        logits0 = model_var.get_logits()

        if it < warmup_iters:
            # warmup: deterministic gate
            gate0 = torch.sigmoid(logits0 / tau_Gumbel)
            #if log and it % 1000 == 0:
                #print(f"[Warmup {it}/{warmup_iters}] Using deterministic gate")
        else:
            if np.random.rand() < Gumbel_ratio:  # 40% Gumbel
                gate0 = model_var.generate_mask((1, tau_Gumbel)).squeeze(0)
            else:  # 60% deterministic
                gate0 = torch.sigmoid(logits0 / tau_Gumbel)

            #if log and it == warmup_iters:
                #print(f"[Phase 2 starts] Introducing Gumbel exploration")

        gate = torch.cat([torch.ones(1), gate0], dim=0)
        #gate = gate0

        outs = [model(gate * x) for x in xs]

        # Use smoothed quantile loss
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

            residual = ys[e] - outs[e]
            psi = ker_grad(-residual, h=h, kernel_type=kernel_type) - tau_quantile
            grad_beta_e = (psi.T @ xs[e]) / len(ys[e])
            grad_beta_e = grad_beta_e.view(-1)
            grad_beta_e = torch.clamp(grad_beta_e, -5, 5)

            loss_J += torch.sum((grad_beta_e ** 2) * gate)
        sparsity_penalty = sparsity_weight * torch.sum(gate0)

        loss = loss_R + hyper_gamma * loss_J + sparsity_penalty

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

            if log and it % 5000 == 0:
                active_gates = np.sum(sigmoid(logits) > 0.5)
                print(f"Iter {it}: Loss={loss.item():.4f}, Active gates={active_gates}/{dim_x}")

    # Final output
    gate_final = sigmoid(logits)
    gate_full = np.concatenate([[1.0], gate_final])
    beta_full = weight_rec[-1] * gate_full
    #beta_full = weight_rec[-1] * sigmoid(logits)

    ret = {'beta_full': beta_full,
           'beta': beta_full[1:],
           'weight_rec': np.array(weight_rec),
           'gate_rec': np.array(gate_rec),
           'loss_rec': np.array(loss_rec),
           #'final_gate': gate_final,
           #'n_active': np.sum(gate_final > 0.5)
    }

    return ret
