from data.model import *
from demo_wrapper import EILLS_Gumbel
from methods.brute_force import greedy_search, brute_force, pooled_least_squares, support_set
from methods.predessors import *
import numpy as np
from demo_wrapper import *
from joblib import Parallel, delayed

##############################################
#
#                Batch Tests
#
##############################################


dim_z = dim_x + 1
models = [StructuralCausalModel1(dim_z), StructuralCausalModel2(dim_z)]
#models = [StructuralCausalModel1(dim_z, exogenous_dist='t'), StructuralCausalModel2(dim_z, exogenous_dist='t')]
true_coeff = np.array([3, 2, -0.5] + [0] * (dim_z - 4))

candidate_n = [100, 200, 300, 400, 500, 700, 1000, 1500, 2000]
#candidate_n = [100, 200, 400]
set_s_star = [0, 1, 2]
set_g = [6, 7, 8]
set_lse = [6, 7, 8, 9]

sets_interested = [
    set_s_star,
    set_g,
    set_lse
]

num_repeats = 200

np.random.seed(0)

methods = [
    oracle_irm,
    oracle_icp,
    oracle_anchor,
    eills,
    KS_IQR_brute_force
]

result = np.zeros((len(candidate_n), num_repeats, len(methods), dim_x))

def run_single_case(n, t, models, methods, true_coeff):
    print(f'Running Case: n = {n}, t = {t}')

    # generate data
    xs, ys = [], []
    oracle_var = 0
    for i in range(2):
        x, y, _ = models[i].sample(n)
        xs.append(x)
        ys.append(y)

    betas = []
    for method in methods:
        beta = method(xs, ys, true_coeff)
        betas.append(beta)

    #  beta for all t and methods
    return t, np.stack(betas, axis=0)

if __name__ == "__main__":

    for (ni, n) in enumerate(candidate_n):

        results_t = Parallel(
            n_jobs=20,
            backend="loky",
            verbose=10
        )(
            delayed(run_single_case)(
                n, t, models, methods, true_coeff
            )
            for t in range(num_repeats)
        )
        for t, betas in results_t:
            result[ni, t, :, :] = betas

np.save('iqr_demo.npy', result)
#np.save("iqr_demo_heavytail", result)