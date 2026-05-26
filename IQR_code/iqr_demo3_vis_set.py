import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import genfromtxt
from data.model import *
from demo_wrapper3 import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=20)
rc('text', usetex=True)

color_tuple = [
	'#E41A1C',  # Method 1 - 红色
    '#E41A1C',
    '#377EB8',  # Method 2 - 蓝色
    '#377EB8',
    '#4DAF4A',  # Method 3 - 绿色
    '#4DAF4A',
    '#FF7F00',  # Method 4 - 橙色
    '#FF7F00',
    '#984EA3',  # Method 5 - 紫色
    '#984EA3',
    '#A65628',  # Method 6 - 棕色
    '#A65628',
    '#F781BF',  # Method 7 - 粉色
    '#F781BF',
    '#999999',
    '#999999',
]
#results = np.load('iqr_demo.npy')
#results = np.load('iqr_demo_heavytail.npy')
#results = np.load('iqr_demo_tnoise.npy')
#results = np.load('iqr_demo_Model2.npy')
results = np.load('iqr_demo_Model3.npy')
dim_x = 4

vec_n = [100, 200, 300, 400, 500, 700, 1000, 1500, 2000]
#vec_n = [100, 300, 500,700,1000]
method_idx = [0, 1, 2, 3, 4, 5]
#method_idx = [0, 1]
#method_name = [r"IRM $S^*$", r"IRM $(S^*)^c$", r"ICP $S^*$", r"ICP $(S^*)^c$", r"Anchor $S^*$", r"Anchor $(S^*)^c$", r"EILLS $S^*$", r"EILLS $(S^*)^c$", r"KSFIQR $S^*$", r"KSFIQR $(S^*)^c$"]
#method_name = [r"EILLS $S^*$", r"EILLS $(S^*)^c$", f"KSFIQR $S^*$ (tau={0.3})",  f"KSFIQR $(S^*)^c$ (tau={0.3})"]
method_name = [r"EILLS $S^*$", r"EILLS $(S^*)^c$"]
tau_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
for tau in tau_grid:
   method_name.append(f"KS-IQR $S^*$ (tau={tau})")
   method_name.append(f"KS-IQR $G$ (tau={tau})")

# Line styles: solid, dotted, dashed, dashdot, etc.
lines = [
    'solid',
    'dotted',
    'solid',
    'dotted',
    'solid',
    'dotted',
    'solid',
    'dotted',
    'solid',
    'dotted',
    'solid',
    'dotted',
    'solid',
    'dotted',
    'solid',
    'dotted',
]

# Marker styles: different shapes of markers
markers = [
    'o',
    'x',
    'o',
    'x',
    'o',
    'x',
    'o',
    'x',
    'o',
    'x',
    'o',
    'x',
    'o',
    'x',
    'o',
    'x',
]

# Colors: choosing bright and distinguishable colors for each method
colors = [
    '#E41A1C',
    '#377EB8',
    '#4DAF4A',
    '#FF7F00',
    '#984EA3',
	'#E41A1C',
    '#377EB8',
    '#4DAF4A',
    '#FF7F00',
    '#984EA3',
    '#984EA3',
    '#A65628'
]

colors = [
	'#E41A1C',  # Method 1 - 红色
    '#E41A1C',
    '#377EB8',  # Method 2 - 蓝色
    '#377EB8',
    '#4DAF4A',  # Method 3 - 绿色
    '#4DAF4A',
    '#FF7F00',  # Method 4 - 橙色
    '#FF7F00',
    '#984EA3',  # Method 5 - 紫色
    '#984EA3',
    '#A65628',  # Method 6 - 棕色
    '#A65628',
    '#F781BF',  # Method 7 - 粉色
    '#F781BF',
    '#999999',
    '#999999',
]

fig = plt.figure(figsize=(5, 6))
ax1 = fig.add_subplot(111)
plt.subplots_adjust(top=0.98, bottom=0.1, left=0.17, right=0.98)
ax1.set_ylabel(r"$\|\bar{\Sigma}^{1/2}(\hat{\beta} - \beta^*)\|_2^2$")

true_coeff = np.array([3, 2, -0.5] + [0] * 1)

true_coeff = np.reshape(true_coeff, (1, 1, dim_x))
for i in method_idx:
    ax1.plot(vec_n, np.mean(np.abs(results[:, :, i, 0:3]) > 0, axis=(1,2)) * 3,
			linestyle=lines[2 * i], marker=markers[2 * i], label=method_name[2 * i], color=colors[2 * i])
    ax1.plot(vec_n, np.mean(np.abs(results[:, :, i, [3]]) > 0, axis=(1,2)) * 1,
			linestyle=lines[2 * i + 1], marker=markers[2 * i + 1], label=method_name[2 * i + 1], color=colors[2 * i + 1])

#ax1.plot(vec_n, np.mean(np.abs(results[:, :, 0, 0:3]) > 0, axis=(1,2)) * 3,
#			linestyle=lines[6], marker=markers[6], label=method_name[0], color=colors[6])
#ax1.plot(vec_n, np.mean(np.abs(results[:, :, 0, [3]]) > 0, axis=(1,2)) * 1,
#			linestyle=lines[7], marker=markers[7], label=method_name[1], color=colors[7])
#
#ax1.plot(vec_n, np.mean(np.abs(results[:, :, 1, 0:3]) > 0, axis=(1,2)) * 3,
#			linestyle=lines[8], marker=markers[8], label=method_name[2], color=colors[8])
#ax1.plot(vec_n, np.mean(np.abs(results[:, :, 1, [3]]) > 0, axis=(1,2)) * 1,
#			linestyle=lines[9], marker=markers[9], label=method_name[3], color=colors[9])

#[5, 6, 7, 8, 10]

#ax1.plot(vec_n, np.mean(np.abs(results[:, :, 0, 0:3]) > 0, axis=(1,2)) * 3,
#			linestyle=lines[6], marker=markers[6], label=method_name[6], color=colors[6])
#.plot(vec_n, np.mean(np.abs(results[:, :, 0, [5, 6, 7, 8, 10]]) > 0, axis=(1,2)) * 5,
#			linestyle=lines[7], marker=markers[7], label=method_name[7], color=colors[7])

#ax1.plot(vec_n, np.mean(np.abs(results[:, :, 2, 0:3]) > 0, axis=(1,2)) * 3,
#			linestyle=lines[8], marker=markers[8], label=method_name[8], color=colors[8])
#ax1.plot(vec_n, np.mean(np.abs(results[:, :, 2, [5, 6, 7, 8, 10]]) > 0, axis=(1,2)) * 5,
#			linestyle=lines[9], marker=markers[9], label=method_name[9], color=colors[9])

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax1.set_xlabel('$n$')
plt.ylim((-0.2, 3.2))
ax1.set_ylabel('number of selected variables')

ax1.legend(loc='best')
ax1.legend(fontsize=10)
plt.savefig("fig_set_S_Sc_Model3.pdf")
