import numpy as np
import torch
import math
import scipy
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from data.utils import MultiEnvDataset
from methods.modules import *
from scipy.optimize import minimize
from scipy.special import erf
import os
os.environ["R_HOME"] = r"E:\R\R-4.5.2"
os.environ["PATH"]   = r"E:\R\R-4.5.2\bin\x64;" + os.environ["PATH"]
import numpy as np
from rpy2 import robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.conversion import localconverter
clime_r = importr('clime')

# kernel
def uniform_kernel(u, h):
    uh = u / h
    return np.where(
        np.abs(uh) <= 1, 1/(2*h),
        0)

# Gaussian kernel
def gaussian_kernel(u, h):
    return np.exp(-u ** 2 / (2 * h ** 2)) / (h * math.sqrt( 2 * math.pi))

# Laplacian kernel
def laplacian_kernel(u, h):
    return np.exp(-np.abs(u) / h) / (2 * h)

# Logistic kernel
def logistic_kernel(u, h):
    return np.exp(-u / h) / ((1 + np.exp(-u / h)) ** 2 * h)

# Epanechnikov kernel
def epanechnikov_kernel(u, h):
    return (3 / 4 * (1 - u/h) ** 2) * (np.abs((u/h)) <= 1) / h

# kernel gradient

# uniform grad
def uniform_grad(u, h):
    uh = u / h
    return np.where(
        uh <= -1, 0,
        np.where(uh >= 1, 1,
                    0.5 * (1 + uh))
    )

# gaussian grad
def gaussian_grad(u, h):
    return 0.5 * (1 + erf(u / (h * math.sqrt(2))))

# Laplacian kernel
def laplacian_grad(u, h):
    return np.where(
        u < 0,
        0.5 * np.exp(u / h),
        1 - 0.5 * np.exp(-u / h)
    )

# Logistic grad
def logistic_grad(u, h):
    return np.sigmoid(u / h)

# Epanechnikov grad
def epanechnikov_grad(u, h):
    uh = u / h
    return np.where(
        uh <= -1, 0,
        np.where(
            uh >= 1, 1,
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
def ker(u, kernel_type="gaussian", h=0.1, tau=0.5):
    """
    Compute the smoothed loss using the selected kernel type and quantile (τ)
    u: residual (ys - outs), kernel_type: type of smoothing kernel
    """
    if kernel_type == "uniform":
        return uniform_kernel(u, h)
    elif kernel_type == "gaussian":
        return gaussian_kernel(u, h)
    elif kernel_type == "laplacian":
        return laplacian_kernel(u, h)
    elif kernel_type == "logistic":
        return logistic_kernel(u, h)
    elif kernel_type == "epanechnikov":
        return epanechnikov_kernel(u, h)
    else:
        raise ValueError(
            "Invalid kernel type. Choose from 'uniform', 'gaussian', 'laplacian', 'logistic', 'epanechnikov'.")

# kernel loss
def uniform_kernel_loss(u, h, tau):
    """
    Uniform kernel with quantile (τ) smoothing
    """
    U_u = (((u/h) ** 2) / 2 + 1 / 2) * (np.abs(u) <= h).float() + np.abs(u) * (np.abs(u) > h).float()
    return (h / 2) * U_u + (tau - 1 / 2) * u

# Gaussian kernel
def gaussian_kernel_loss(u, h, tau):
    """
    Gaussian kernel with quantile (τ) smoothing
    """
    # expectation of folded normal distribution
    Phi = 0.5 * (1 + erf(u / (h * math.sqrt(2))))
    expected_abs_gu = (
            math.sqrt(2 / math.pi) * h * np.exp(-u ** 2 / (2 * h ** 2))
            + u * (2 * Phi - 1)
    )

    return (1 / 2) * expected_abs_gu + (tau - 1 / 2) * u

# Laplacian kernel
def laplacian_kernel_loss(u, h, tau):
    """
    Laplacian kernel with quantile (τ) smoothing
    """
    rho_tau = u * (tau - (u < 0).float())

    return rho_tau + h * np.exp(-np.abs(u) / h) / 2

# Logistic kernel
def logistic_kernel_loss(u, h, tau):
    """
    Logistic kernel with quantile (τ) smoothing
    """
    return (tau * u + h * np.log1p(np.exp(-u / h)))

# Epanechnikov kernel
def epanechnikov_kernel_loss(u, h, tau):
    """
    Epanechnikov kernel with quantile (τ) smoothing

    """
    E_u = ((3 * (u/h) ** 2) / 4 - (u/h) ** 4 / 8 + 3 / 8) * (np.abs((u/h)) <= 1) + np.abs((u/h)) * (np.abs((u/h)) > 1)
    return (h / 2) * E_u + (tau - 1 / 2) * u

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

def calc_siqr_loss(var_set, x_list, y_list, kernel_type="gaussian", tau=0.5, hyper_gamma=10, h=0.1):
    num_envs = len(x_list)
    Ys_list = [y.reshape(-1, 1) for y in y_list]
    #if len(var_set) == 0:
    #    return np.zeros(0)
    if len(var_set) == 0:
        Xs_int_list = [
            np.ones((y.shape[0], 1))
            for y in y_list
        ]
    else:
        Xs_list = [x[:, var_set] for x in x_list]
        Xs_int_list = [
            np.hstack([np.ones((X.shape[0], 1)), X])
            for X in Xs_list
        ]

    #Xs_list = [x[:, var_set] for x in x_list]
    #Ys_list = [y.reshape(-1, 1) for y in y_list]
    #num_envs = len(x_list)
    s = len(var_set)

    #Xs_int_list = [
    #    np.hstack([np.ones((X.shape[0], 1)), X])
    #    for X in Xs_list
    #]

    def objective(beta):
        beta = beta.reshape(-1, 1)
        loss = 0.0
        for i in range(num_envs):
            X = Xs_int_list[i]
            y = Ys_list[i]
            r = y - X @ beta
            loss_Re = np.mean(smoothed_quantile_loss(r, kernel_type=kernel_type, h=h, tau=tau))
            psi = ker_grad(-r, h=h, kernel_type=kernel_type) - tau
            grad_Re = X.T @ psi / len(y)
            loss_Je = hyper_gamma * np.sum(grad_Re ** 2)
            loss += loss_Re + loss_Je
        loss /= num_envs
        return loss
    beta0 = np.zeros(s+1)
    res = minimize(
        objective,
        beta0,
        method="L-BFGS-B",
        options={
            "maxiter": 5000,
            "maxfun": 50000,
            "ftol": 1e-12,
            "gtol": 1e-10
        }
    )
    if not res.success:
        return None, np.inf
    cur_beta = res.x.reshape(-1, 1)
    cur_loss = objective(cur_beta)
    #cur_beta = cur_beta[1:]
    #return cur_beta, cur_loss

    #if len(var_set) == 0:
    #    return np.zeros(0), cur_loss
    #else:
    #    return cur_beta[1:], cur_loss
    return cur_beta, cur_loss


def brute_force_siqr(x_list, y_list, tau=0.5, kernel_type='gaussian', hyper_gamma=10,h=None,return_intercept=None):
    if return_intercept is None:
        return_intercept = False
    dim_x = x_list[0].shape[1]
    num_envs = len(x_list)
    n0 = np.shape(x_list[0])[0]
    if h is None:
        h = np.sqrt(tau * (1 - tau) * dim_x * hyper_gamma / (n0 * num_envs))
        #h = np.maximum(0.05, np.minimum(np.sqrt(tau * (1 - tau) * dim_x * hyper_gamma / (n0 * num_envs)), 0.2))

    best_loss = np.inf
    best_beta = np.zeros(dim_x)
    best_set = []

    for sel in range(2 ** dim_x):
        var_set = [j for j in range(dim_x) if (sel & (1 << j))]
        beta_S, loss_S = calc_siqr_loss(var_set, x_list, y_list, kernel_type=kernel_type,
                                        tau=tau, hyper_gamma=hyper_gamma, h=h)
        if beta_S is None:
            continue

        cur_loss = loss_S
        beta_full = np.zeros(dim_x+1)
        beta_full[0] = beta_S[0]
        for i, idx in enumerate(var_set):
            beta_full[idx+1] = beta_S[i+1]

        if cur_loss < best_loss:
            best_loss = cur_loss
            best_set = var_set
            best_beta = beta_full

    print(f"SIQR brute force: var_set={best_set}, loss={best_loss}")
    if return_intercept == False:
        return best_beta[1:]
    else:
        return best_beta


def pooled_SQR(x_list, y_list, var_set=None, tau=0.5, kernel_type="gaussian", h=None):
    dim_x = x_list[0].shape[1]
    if var_set is None:
        var_set = np.arange(dim_x)
    if len(var_set) == 0:
        return np.zeros(dim_x)
    num_envs = len(x_list)
    n0 = x_list[0].shape[0]

    if h is None:
        h = np.sqrt(tau * (1 - tau) * dim_x / (n0 * num_envs))
    Xs_list = [x[:, var_set] for x in x_list]
    Xs_int_list = [
        np.hstack([np.ones((X.shape[0], 1)), X])
        for X in Xs_list
    ]
    Ys_list = [y.reshape(-1, 1) for y in y_list]

    def objective(beta):
        beta = beta.reshape(-1, 1)
        loss = 0.0
        for i in range(num_envs):
            X = Xs_int_list[i]
            y = Ys_list[i]
            r = y - X @ beta
            loss += np.mean(
                smoothed_quantile_loss(
                    r,
                    kernel_type=kernel_type,
                    h=h,
                    tau=tau
                )
            )

        return loss / num_envs

    beta0 = np.zeros(len(var_set) + 1)
    res = minimize(
        objective,
        beta0,
        method="L-BFGS-B",
        options={
            "maxiter": 5000,
            "ftol": 1e-12,
            "gtol": 1e-10
        }
    )

    if not res.success:
        print("Warning:", res.message)

    beta = res.x.reshape(-1)

    intercept = beta[0]
    coef = beta[1:]

    return coef
