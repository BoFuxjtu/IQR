from data.model import *
from methods.brute_force import greedy_search, brute_force, pooled_least_squares, support_set
from methods.predessors import *
from methods.eills_gumbel import *
from methods.iqr_brute import pooled_SQR
from demo_wrapper import *
import numpy as np
import time
from utils import get_linear_SCM, get_SCM, get_nonlinear_SCM
import argparse
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="mode", type=int, default=1)
args = parser.parse_args()

#mode = args.mode
mode=1

import matplotlib.pyplot as plt
from matplotlib import rc

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=16)
rc('text', usetex=True)

if mode == 0:
    #candidate_n = [200, 500, 1000, 2000, 5000]
    candidate_n = [200, 500, 1000, 2500, 5000]

    num_repeats = 60

    np.random.seed(0)

    methods = [
        eills_sgd_gumbel,
		lse_s_star,
        erm,
        KS_IQR_Gumbel_s_star,
        pooled_sqr
    ]
    method_name = ["EILLS-GB", "EILLS-RF", "KS-IQR-GB", "KS-IQR-RF", "Oracle-LS", "Pool-LS", "Oracle-SQR", "Pool-SQR"]

    #result = np.zeros((len(candidate_n), num_repeats, len(methods) + 4, 12))
    result = np.zeros((len(candidate_n), num_repeats, len(methods) + 4, 70))

    def run_single_case(n, t, methods):
        print(f'Running Case: n = {n}, t = {t}')

        np.random.seed(t)
        torch.manual_seed(t)

        # generate random graph with 20 nodes
        #models, true_coeff, _, _, _ = get_linear_SCM(
        #    num_vars=71, num_envs=2, y_index=35,
        #    min_child=5, min_parent=5, nonlinear_id=5,
        #    bias_greater_than=0.5, same_var=False, log=False, randtype='gaussian'
        #)
        # t3 distribution
        models, true_coeff, _, _, _ = get_linear_SCM(
            num_vars=71, num_envs=2, y_index=35,
            min_child=5, min_parent=5, nonlinear_id=5,
            bias_greater_than=0.5, same_var=False, log=False, randtype='mix'
        )
        #models = [StructuralCausalModel1(13), StructuralCausalModel2(13)]
        #true_coeff = np.array([3, 2, -0.5] + [0] * (13 - 4))

        # generate data
        xs, ys = [], []
        for i in range(2):
            x, y, _ = models[i].sample(n)
            xs.append(x)
            ys.append(y)

        betas = []

        for mid, method in enumerate(methods):

            if mid == 0:  # EILLS-RF
                packs = eills_sgd_gumbel(
                    xs, ys,
                    hyper_gamma=10,
                    learning_rate=1e-3,
                    niters=50000,
                    batch_size=64,
                    init_temp=5,
                    final_temp=0.1,
                    log=False
                )
                beta1 = packs['weight']
                betas.append(beta1)
                #mask = packs['gate_rec'][-1] > 0.8
                #var_set = np.arange(70)[mask].tolist()

                #beta_EILLS_RF = broadcast(
                #    pooled_least_squares([x[:, var_set] for x in xs], ys),
                #    var_set,
                #    70
                #)
                #betas.append(beta_EILLS_RF)


                #packs_IQR = KS_IQR_Gumbel_sgd(
                #    xs, ys,
                #    hyper_gamma=10,
                #    tau_quantile=0.5,
                #    learning_rate=2e-2,
                #    niters=500000,
                #    batch_size=n,
                #    tau_Gumbel=2.5,
                #    final_temp=0.05,
                #    h=0.10,
                #    min_lr=5e-3,
                #    kernel_type='gaussian'
                #)

                packs_IQR = KS_IQR_Gumbel_sgd2(
                    xs, ys,
                    hyper_gamma=10,
                    tau_quantile=0.5,
                    learning_rate=1e-2,
                    niters=100000,
                    batch_size=max(n//3,200),
                    tau_Gumbel=2.5,
                    final_temp=0.1,
                    h=0.20,
                    min_lr=1e-2,
                    kernel_type='gaussian',
                    offset=-1,
                    warmup_ratio = 0.3, Gumbel_ratio = 0.4
                )
                betas.append(packs_IQR['beta'])

            else:
                beta = method(xs, ys, true_coeff)
                betas.append(beta)
        return t, true_coeff, np.stack(betas, axis=0)


    if __name__ == "__main__":

        for (ni, n) in enumerate(candidate_n):

            results_t = Parallel(
                n_jobs=20,
                backend="loky",
                verbose=20
            )(
                delayed(run_single_case)(
                    n, t, methods
                )
                for t in range(num_repeats)
            )
            for t, true_coeff, betas in results_t:

                result[ni, t, 0, :] = true_coeff

                #for mid in range(len(methods)+3):
                for mid in range(len(methods) + 1):
                    result[ni, t, mid + 1, :] = betas[mid]

    #np.save('IQR_Gumbel_test2.npy', result)
    #np.save('IQR_Gumbel_test2_t.npy', result)
    np.save('IQR_Gumbel_test2_mix.npy', result)

else:
    #results = np.load('IQR_Gumbel_test2.npy')
    #results = np.load('IQR_Gumbel_test2_t.npy')
    results = np.load('IQR_Gumbel_test2_mix.npy')
    plt.close('all')
    num_n = results.shape[0]
    num_sml = results.shape[1]

    #vec_n = [200, 500, 1000, 2000, 5000]
    vec_n = [200, 500, 1000, 2500, 5000]
    #method_name = ["EILLS-GB", "EILLS-RF", "KS-IQR-GB",  "KS-IQR-RF", "Oracle-LS", "Pool-LS", "Oracle-SQR", "Pool-SQR"]
    #method_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    #method_idx = [0, 2, 4, 5, 6, 7]
    method_idx = [0, 1, 2, 3, 4, 5]
    method_name = ["EILLS-GB", "KS-IQR-GB", "Oracle-LS", "Pool-LS", "Oracle-SQR", "Pool-SQR"]

    lines = [
        'solid',
        'solid',
        'dashed',
        'dashed',
        'solid',
        'solid',
        'dashed',
        'dashed'
    ]

    # Marker styles: different shapes of markers
    markers = [
    'D',  # First line for Method 1
    'o',  # Second line for Method 1
    'P',  # First line for Method 2
    's',  # Second line for Method 2
    'x',  # First line for Method 3
    '<',  # Second line for Method 3
    '>',  # First line for Method 4
    '^',  # Second line for Method 4
    'v',  # First line for Method 5
    'H',   # Second line for Method 5
    '*',  # Method 6 - line 1
    'd'   # Method 6 - line 2
    ]

    # Colors: choosing bright and distinguishable colors for each method
    colors = [
        '#E41A1C',
        '#377EB8',
        '#4DAF4A',
        '#FF7F00',
        '#984EA3',
        '#A65628',
        '#9acdc4',
        '#05348b',
        '#ae1908',
        '#ec813b',
        '#e5a84b',
    ]

    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111)
    plt.subplots_adjust(top=0.98, bottom=0.12, left=0.17, right=0.98)
    ax1.set_ylabel(r"$\|\hat{\beta} - \beta^\star\|_2^2$", fontsize=22)

    for (j, mid) in enumerate(method_idx):
        metric = []
        for i in range(len(vec_n)):
            measures = []
            for k in range(num_sml):
                #error = np.sum(np.square(results[i, k, mid + 1, :] - results[i, k, 0, :]))
                #if error > 0.2 and mid != 3:
                #    print(f'method = {mid}, n = {vec_n[i]}, seed = {k}, error = {error}')
                measures.append(np.sum(np.square(results[i, k, mid + 1, :] - results[i, k, 0, :])))
            metric.append(np.median(measures))
        ax1.plot(vec_n, metric, linestyle=lines[j], marker=markers[j], label=method_name[mid], color=colors[j])
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax1.set_xlabel('$n$', fontsize=22)

    ax1.legend(loc='best')
    #plt.savefig("gumbel_test.pdf")
    #plt.savefig("gumbel_test_t.pdf")
    plt.savefig("gumbel_test_mix.pdf")
    plt.show()