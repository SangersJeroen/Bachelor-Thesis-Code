import mreels
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

mpl.rcParams['figure.dpi'] = 180
mpl.rcParams["legend.loc"] = "best"
mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ["Computer Modern Roman"]
})

data = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4', 25)

data.remove_neg_val()
data.rem_neg_el()

qmap_m, qaxis_m = mreels.get_qeels_slice(data, (924, 1198))
qmap_k, qaxis_k = mreels.get_qeels_slice(data, (825, 1117), starting_point=(924, 1198))
eax = data.axis0

qmap = np.append(qmap_m, qmap_k[1:], axis=0)
qaxis = np.append(qaxis_m, qaxis_k[1:])

qmap = mreels.sigmoid(qmap, (50,150))

fig, ax = plt.subplots(1,1)

ax.pcolormesh(qaxis, eax, qmap.T, shading='nearest')
plt.gcf().set_size_inches(6.4,4)
ax.set_xlabel(r'$q_{\perp}$ [$nm^{-1}$]', fontsize=12)
ax.set_ylabel(r'$\Delta E$ [$eV$]', fontsize=12)

ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))

ax.xaxis.set_major_locator(MultipleLocator(0.5))

plt.vlines(qaxis_m[-1], ymin=0, ymax=eax[-1], linestyle='--', linewidth=2, color='white')

plt.text(qaxis[0], 37, r"$\Gamma$", fontsize=12)
plt.text(qaxis_m[-5], 37, r"$M$", fontsize=12)
plt.text(qaxis[-10], 37, r"$K$", fontsize=12)

plt.savefig('qmap-example.eps', format='eps')
plt.show()
