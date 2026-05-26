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
import os
import re
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
from joblib import Parallel, delayed
import multiprocessing

mode = 1
# import pickle
if mode == 0:
    def load_sachs_environments(folder_path, target_name, k=9, standardize=True):
        """
        读取文件夹中的多个 xls，每个 xls 作为一个 environment

        Parameters
        ----------
        folder_path : str
            xls file path
        target_name : str
            target outcome name
        standardize : bool

        Returns
        -------
        features : list of np.ndarray
        responses : list of np.ndarray
        var_names : list of str
        """

        features = []
        responses = []

        files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.xls', '.xlsx')) and not f.startswith('~')])

        def extract_index(filename):
            match = re.match(r"(\d+)\.", filename)
            if match:
                return int(match.group(1))
            else:
                return float('inf')

        files_sorted = sorted(files, key=extract_index)
        selected_files = files_sorted[:k]
        selected_files = [f for i, f in enumerate(selected_files) if i != 1]
        # print("Using environments:")
        # for f in selected_files:
        # print(f)
        for file in selected_files:
            path = os.path.join(folder_path, file)
            df = pd.read_excel(path, engine='xlrd')
            # df = np.log(df + 1)
            # df = np.arcsinh(df)

            if target_name not in df.columns:
                raise ValueError(f"{target_name} not found in {file}")

            y = df[target_name].values.reshape(-1, 1)
            X = df.drop(columns=[target_name]).values

            if standardize:
                X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
                y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-8)

            features.append(X.astype(np.float32))
            responses.append(y.astype(np.float32))

        df0 = pd.read_excel(os.path.join(folder_path, selected_files[0]), engine='xlrd')
        var_names = list(df0.drop(columns=[target_name]).columns)

        return features, responses, var_names


    def process_single_target(target_var, folder_path, tau_list):
        """
        处理单个 target variable 的函数

        Parameters
        ----------
        target_var : str
            目标变量名
        folder_path : str
            数据文件夹路径
        tau_list : list
            tau值列表

        Returns
        -------
        tuple : (target_var, result_dict)
            返回目标变量名和结果字典
        """
        print(f"\nTarget: {target_var}")

        try:
            features, responses, var_names = load_sachs_environments(
                folder_path, target_name=target_var, k=9, standardize=True
            )

            if features is None:
                print(f"  Failed to load data for {target_var}")
                return target_var, None

            result = {
                "KS_IQR_full_env": {},
                "EILLS_full_env": None,
                "KS_IQR_pairwise": {tau: {} for tau in tau_list},
                "EILLS_pairwise": {}
            }

            results_pairwise_KS_IQR = {}
            results_pairwise_KS_IQR[target_var] = {tau: {} for tau in tau_list}

            # KS-IQR
            for tau in tau_list:
                beta_KS_IQR = brute_force_siqr(features, responses, tau=tau, kernel_type='gaussian', hyper_gamma=20,
                                                h=0.1)
                result["KS_IQR_full_env"][tau] = beta_KS_IQR.copy()
                selected_KS_IQR = [(var_names[i], beta_KS_IQR[i])
                                   for i in range(len(beta_KS_IQR)) if beta_KS_IQR[i] != 0]

                print(f"KS-IQR (tau = {tau})  Selected: {len(selected_KS_IQR)} variables")
                for var, coef in selected_KS_IQR:
                    print(f"    {var}: {coef:.4f}")

            # EILLS
            beta_EILLS = brute_force(features, responses, 20, loss_type='eills')
            result["EILLS_full_env"] = beta_EILLS.copy()
            selected_EILLS = [(var_names[i], beta_EILLS[i])
                              for i in range(len(beta_EILLS)) if beta_EILLS[i] != 0]

            print(f"EILLS  Selected: {len(selected_EILLS)} variables")
            for var, coef in selected_EILLS:
                print(f"    {var}: {coef:.4f}")

            # Pairwise
            edge_count_KS_IQR = {tau: {} for tau in tau_list}
            edge_count_EILLS = defaultdict(int)
            total = 0
            num_envs = len(features)

            for i, j in combinations(range(num_envs), 2):
                feats = [features[i], features[j]]
                resps = [responses[i], responses[j]]
                beta2_EILLS = brute_force(feats, resps, 20, loss_type='eills')
                result["EILLS_pairwise"][(i, j)] = beta2_EILLS.copy()
                selected2_EILLS = np.where(np.abs(beta2_EILLS) > 1e-2)[0]
                for k in selected2_EILLS:
                    edge_count_EILLS[k] += 1

                for tau in tau_list:
                    beta2_KS_IQR = brute_force_siqr(feats, resps, tau=tau, kernel_type='gaussian', hyper_gamma=20,
                                                     h=0.1)
                    result["KS_IQR_pairwise"][tau][(i, j)] = beta2_KS_IQR.copy()
                    selected2_KS_IQR = np.where(np.abs(beta2_KS_IQR) > 1e-2)[0]
                    for k in selected2_KS_IQR:
                        edge_count_KS_IQR[tau][k] = edge_count_KS_IQR[tau].get(k, 0) + 1

                total += 1

            # 计算频率
            edge_freq_KS_IQR = {}
            for tau in tau_list:
                edge_freq_KS_IQR[tau] = {
                    k: edge_count_KS_IQR[tau][k] / total
                    for k in edge_count_KS_IQR[tau]
                }
            edge_EILLS_freq = {k: edge_count_EILLS[k] / total for k in edge_count_EILLS}
            threshold = 0

            selected_stable_EILLS = [
                (var_names[k], edge_EILLS_freq[k])
                for k in edge_EILLS_freq if edge_EILLS_freq[k] >= threshold
            ]

            selected_stable_EILLS = sorted(selected_stable_EILLS, key=lambda x: -x[1])

            print(f"[EILLS ] Stable variables among 28 pairwise (freq >= {threshold}): {len(selected_stable_EILLS)}")
            for var, freq in selected_stable_EILLS:
                print(f"    {var}: {freq:.2f}")

            for tau in tau_list:
                print(f"\n[KS-IQR tau={tau}]")

                selected_stable = sorted(
                    [(var_names[k], edge_freq_KS_IQR[tau][k])
                     for k in edge_freq_KS_IQR[tau]
                     if edge_freq_KS_IQR[tau][k] >= threshold],
                    key=lambda x: -x[1]
                )

                for var, freq in selected_stable:
                    print(f"    {var}: {freq:.2f}")

            return target_var, result

        except Exception as e:
            print(f"Error processing {target_var}: {str(e)}")
            import traceback
            traceback.print_exc()
            return target_var, None


    if __name__ == '__main__':
        folder_path = "sachs.som.datasets/Data Files"

        first_file = [f for f in os.listdir(folder_path) if f.endswith('.xls')][0]
        df_test = pd.read_excel(os.path.join(folder_path, first_file))
        all_vars = df_test.columns.tolist()

        tau_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        print(f"Analyzing {len(all_vars)} variables: {all_vars}\n")
        print(f"Using {multiprocessing.cpu_count()} CPU cores for parallel processing\n")

        results_list = Parallel(n_jobs=20, verbose=10)(
            delayed(process_single_target)(target_var, folder_path, tau_list)
            for target_var in all_vars
        )

        results = {target_var: result for target_var, result in results_list if result is not None}

    np.save("results_real_data_Sachs.npy", results, allow_pickle=True)
    print("\n" + "=" * 60)
    print("All processing completed!")
    print(f"Processed {len(results)} variables successfully")
    print("Results saved to results.npy")
    print("=" * 60)
if mode == 1:
    results = np.load('results_real_data_Sachs.npy', allow_pickle=True).item()
    folder_path = "sachs.som.datasets/Data Files"
    first_file = [f for f in os.listdir(folder_path) if f.endswith('.xls')][0]
    df_test = pd.read_excel(os.path.join(folder_path, first_file))
    all_vars = df_test.columns.tolist()
    first_target = list(results.keys())[0]
    tau_list = sorted(results[first_target]["KS_IQR_full_env"].keys())

    for target_var in all_vars:
        if target_var not in results or results[target_var] is None:
            continue

        print(f"\n{'=' * 100}")
        print(f"target variable: {target_var}")
        print(f"{'=' * 100}")

        predictor_vars = [v for v in all_vars if v != target_var]

        # ============ KS-IQR (Full) ============
        print(f"\n[ KS-IQR (Full Environment)]")
        print("-" * 100)

        for tau in tau_list:
            beta = results[target_var]["KS_IQR_full_env"][tau]
            selected = [(predictor_vars[i], beta[i]) for i in range(len(beta)) if abs(beta[i]) > 1e-2]

            if selected:
                print(f"\nTau = {tau}: select {len(selected)} predictors")
                for var, coef in sorted(selected, key=lambda x: abs(x[1]), reverse=True)[:5]:
                    print(f"  {var:15s} -> {target_var:15s}  coef: {coef:8.4f}")
            else:
                print(f"Tau = {tau}: none selected")

        # ============ EILLS (Full) ============
        print(f"\n[ EILLS (Full Environment)]")
        print("-" * 100)

        beta = results[target_var]["EILLS_full_env"]
        selected = [(predictor_vars[i], beta[i]) for i in range(len(beta)) if abs(beta[i]) > 1e-2]

        if selected:
            print(f"select {len(selected)} predictors")
            for var, coef in sorted(selected, key=lambda x: abs(x[1]), reverse=True)[:5]:
                print(f"  {var:15s} -> {target_var:15s}  coef: {coef:8.4f}")
        else:
            print("none selected")

        # ============ KS-IQR (Pairwise) ============
        print(f"\n[ KS-IQR (Pairwise Stability)]")
        print("-" * 100)

        for tau in tau_list:
            pairwise_results = results[target_var]["KS_IQR_pairwise"][tau]

            edge_count = {}
            total_pairs = len(pairwise_results)

            for pair_key, beta in pairwise_results.items():
                for i, pred_var in enumerate(predictor_vars):
                    if abs(beta[i]) > 1e-2:
                        edge_count[pred_var] = edge_count.get(pred_var, 0) + 1

            stable_edges = [(var, count / total_pairs) for var, count in edge_count.items()]
            stable_edges = sorted(stable_edges, key=lambda x: x[1], reverse=True)

            if stable_edges:
                print(f"\nTau = {tau}: {len(stable_edges)} predictors selected by pairwise stability")
                #for var, freq in stable_edges[:5]:
                for var, freq in stable_edges:
                    print(
                        f"  {var:15s} -> {target_var:15s}  {freq:.2f} ({int(freq * total_pairs)}/{total_pairs} pairs)")
            else:
                print(f"Tau = {tau}: none selected")

        # ============ EILLS (Pairwise) ============
        print(f"\n[ EILLS (Pairwise Stability)]")
        print("-" * 100)

        pairwise_results = results[target_var]["EILLS_pairwise"]

        edge_count = {}
        total_pairs = len(pairwise_results)

        for pair_key, beta in pairwise_results.items():
            for i, pred_var in enumerate(predictor_vars):
                if abs(beta[i]) > 1e-2:
                    edge_count[pred_var] = edge_count.get(pred_var, 0) + 1

        stable_edges = [(var, count / total_pairs) for var, count in edge_count.items()]
        stable_edges = sorted(stable_edges, key=lambda x: x[1], reverse=True)

        if stable_edges:
            print(f"{len(stable_edges)} predictors selected by pairwise stability")
            for var, freq in stable_edges:
                print(
                    f"  {var:15s} -> {target_var:15s}  {freq:.2f} ({int(freq * total_pairs)}/{total_pairs} pairs)")
        else:
            print("none selected")


