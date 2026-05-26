import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import genfromtxt
from data.model import *
from demo_wrapper import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=20)
rc('text', usetex=True)

results = np.load('iqr_demo2_h_test.npy')

dim_x = 12

env1_model = StructuralCausalModel1(dim_x + 1)
env2_model = StructuralCausalModel2(dim_x + 1)
X1_test, _1, _2 = env1_model.sample(10000)
X2_test, _1, _2 = env2_model.sample(10000)
X_cov = np.matmul(X1_test.T, X1_test) / 20000 + np.matmul(X2_test.T, X2_test) / 20000

num_n = results.shape[0]
num_sml = results.shape[1]

#vec_n = [100, 300, 700, 1000, 2000]
vec_n = [100, 200, 300, 400, 500, 700, 1000, 1500, 2000]
method_name = ["EILLS", "KS-IQR (default)"]
method_idx = [0,1,2,3,4,5,6,7]
h_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.3]
for h in h_grid:
   method_name.append(f"KS-IQR $S^*$ (h={h})")

lines = [
    'solid',       # First line for Method 1
    'solid',       # Second line for Method 1
    'dotted',      # First line for Method 2
    'dotted',      # Second line for Method 2
    'dashed',      # First line for Method 3
    'dashed',      # Second line for Method 3
    'dashdot',     # First line for Method 4
    'dashdot',     # Second line for Method 4
    'solid',       # First line for Method 5
    'solid',        # Second line for Method 5
    'dotted',      # First line for Method 6
    'dotted',      # Second line for Method 6
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
    '#F781BF',
    '#999999',
]
# '#FF7F00',
#'#377EB8'
fig = plt.figure(figsize=(5, 6))
ax1 = fig.add_subplot(111)
plt.subplots_adjust(top=0.98, bottom=0.1, left=0.17, right=0.98)
ax1.set_ylabel(r"$\|\bar{\Sigma}^{1/2}(\hat{\beta} - \beta^*)\|_2^2$")

dim_x = 12
true_coeff = np.array([3, 2, -0.5] + [0] * 9)
#true_coeff = np.reshape(true_coeff, (1, 1, dim_x))

for (j, mid) in enumerate(method_idx):
	metric = []
	for i in range(len(vec_n)):
		measures = []
		for k in range(num_sml):
			measures.append(mydist(X_cov, results[i, k, mid, :] - true_coeff))
			#measures.append(np.linalg.norm(results[i, k, mid, :] - true_coeff))
		metric.append(np.mean(measures))
	ax1.plot(vec_n, metric, linestyle=lines[j], marker=markers[j], label=method_name[j], color=colors[j])

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax1.set_xlabel('$n$')
ax1.set_yscale("log")
ax1.set_xscale("log")

ax1.legend(loc='best',fontsize=12)
plt.savefig("fig_loss_h.pdf")