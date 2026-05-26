import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import genfromtxt
from data.model import *
from demo_wrapper import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=20)
rc('text', usetex=True)

color_tuple = [
	'#ae1908',  # red
	'#ec813b',  # orange
	'#05348b',  # dark blue
	'#9acdc4',  # pain blue
	'#6bb392',  # green
	'#e5a84b',   # yellow
]
#results = np.load('iqr_demo.npy')
#results = np.load('iqr_demo_heavytail.npy')
results = np.load('iqr_demo_mix.npy')

dim_x = 12

#env1_model = StructuralCausalModel1(dim_x + 1)
#env2_model = StructuralCausalModel2(dim_x + 1)

#env1_model = StructuralCausalModel1(dim_x + 1, exogenous_dist='mix')
#env2_model = StructuralCausalModel2(dim_x + 1, exogenous_dist='mix')
#X1_test, _1, _2 = env1_model.sample(10000)
#X2_test, _1, _2 = env2_model.sample(10000)
#X_cov = np.matmul(X1_test.T, X1_test) / 20000 + np.matmul(X2_test.T, X2_test) / 20000

num_n = results.shape[0]
num_sml = results.shape[1]

#vec_n = [100, 300, 700, 1000, 2000]
vec_n = [100, 200, 300, 400, 500, 700, 1000, 1500, 2000]
method_name = [ "IRM", "ICP", "Anchor", "EILLS", "KS-IQR"]
method_idx = [0,1,2,3,4]

lines = [
	'solid',
	'solid',
	'solid',
	'solid',
	'solid'
]

markers = [
	'D',
	'o',
	's',
	'P',
	'x',
	'<'
]

colors = [
	'#E41A1C',
    '#377EB8',
    '#4DAF4A',
	'#FF7F00',
	'#984EA3'
]
# '#FF7F00',
#'#377EB8'
fig = plt.figure(figsize=(5, 6))
ax1 = fig.add_subplot(111)
plt.subplots_adjust(top=0.98, bottom=0.1, left=0.17, right=0.98)
#ax1.set_ylabel(r"$\|\bar{\Sigma}^{1/2}(\hat{\beta} - \beta^*)\|_2^2$")
ax1.set_ylabel(r"$\|\hat{\beta} - \beta^*\|_2^2$")

dim_x = 12
true_coeff = np.array([3, 2, -0.5] + [0] * 9)
#true_coeff = np.reshape(true_coeff, (1, 1, dim_x))

for (j, mid) in enumerate(method_idx):
	metric = []
	for i in range(len(vec_n)):
		measures = []
		for k in range(num_sml):
			#measures.append(mydist(X_cov, results[i, k, mid, :] - true_coeff))
			measures.append(np.linalg.norm(results[i, k, mid, :] - true_coeff))
		metric.append(np.mean(measures))
	ax1.plot(vec_n, metric, linestyle=lines[j], marker=markers[j], label=method_name[j], color=colors[j])

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax1.set_xlabel('$n$')
ax1.set_yscale("log")
ax1.set_xscale("log")

ax1.legend(loc='best',fontsize=18)
#plt.show()
#plt.savefig("fig_loss.pdf")
#plt.savefig("fig_loss_heavytail.pdf")
plt.savefig("fig_loss_mix.pdf")