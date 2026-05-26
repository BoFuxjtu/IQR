import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import genfromtxt
from data.model import *
from demo_wrapper5 import *
from methods.tools import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=20)
rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')

color_tuple = [
	'#ae1908',  # red
	'#ec813b',  # orange
	'#05348b',  # dark blue
	'#9acdc4',  # pain blue
	'#6bb392',  # green
	'#e5a84b',   # yellow
]
results = np.load('iqr_demo5_0.9.npy')
#results = np.load('iqr_demo5_0.65_0.75.npy')
dim_x = 3

#vec_n = [100, 300, 500,700,1000]
vec_n = [100, 200, 300, 400, 500, 700, 1000, 1500, 2000]
method_idx = [0, 1, 2, 3, 4, 5]
#method_name = [r"EILLS $S^*$", r"EILLS $G$", r"KS-IQR $S^*$", r"KS-IQR $G$"]
method_name = [r"EILLS $S^*$"]
tau_grid = [0.5, 0.6, 0.7, 0.8, 0.9]
for tau in tau_grid:
   method_name.append(f"KS-IQR $S^*$ (tau={tau})")

lines = [
    'solid',       # First line for Method 1
    'dashed',       # Second line for Method 1
    'dashed',      # First line for Method 2
    'dashed',      # Second line for Method 2
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
    '#A65628'
]

dim_x = 3
fig = plt.figure(figsize=(5, 6))
ax1 = fig.add_subplot(111)
plt.subplots_adjust(top=0.98, bottom=0.1, left=0.17, right=0.98)

#true_coeff = np.reshape(true_coeff, (1, 1, dim_x))

#ax1.plot(vec_n, np.mean(np.abs(results[:, :, 0, 0:3]) > 0, axis=(1,2)) * 3,
#			linestyle=lines[0], marker=markers[0], label=method_name[0], color=colors[0])
#ax1.plot(vec_n, np.mean(np.abs(results[:, :, 0, [3]]) > 0, axis=(1,2)) * 1,
#			linestyle=lines[1], marker=markers[1], label=method_name[1], color=colors[1])

#ax1.plot(vec_n, np.mean(np.abs(results[:, :, 1, 0:3]) > 0, axis=(1,2)) * 3,
#			linestyle=lines[2], marker=markers[2], label=method_name[2], color=colors[2])
#ax1.plot(vec_n, np.mean(np.abs(results[:, :, 1, [3]]) > 0, axis=(1,2)) * 1,
#			linestyle=lines[3], marker=markers[3], label=method_name[3], color=colors[3])


for i in method_idx:
    ax1.plot(vec_n, np.mean(np.abs(results[:, :, i, 0:3]) > 0, axis=(1,2)) * 3,
			linestyle=lines[i], marker=markers[i], label=method_name[i], color=colors[i])

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax1.set_xlabel('$n$')
plt.ylim((-0.2, 3.2))
ax1.set_ylabel('number of selected variables')

ax1.legend(loc='best')
ax1.legend(fontsize=14)
plt.savefig("fig_set_demo5_0.9.pdf")
#plt.savefig("fig_set_demo5_0.65_0.75.pdf")