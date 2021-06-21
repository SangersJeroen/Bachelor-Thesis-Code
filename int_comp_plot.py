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
data.build_axes()
eax = data.axis0

data = None

qmap_rad = np.load('qmap_rad.npy')
qmap_line = np.load('qmap_line.npy')
qmap_slice = np.load('qmap_slice.npy')

for i in np.argwhere(
	np.isnan(qmap_line[:,0]) ):
	qmap_line[i,:] = qmap_line[i-1,:]

qaxis_rad = np.linspace(0, 1.43, len(qmap_rad[:,0]))

qaxis_line = np.linspace(0, 1.43, len(qmap_line[:,0]))

qaxis_slice = np.linspace(0, 1.43, len(qmap_slice[:,0]))

print(qmap_rad.shape, qaxis_rad.shape)
print(qmap_line.shape, qaxis_line.shape)
print(qmap_slice.shape, qaxis_slice.shape)

fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
plt.gcf().set_size_inches(6.4, 1.6*6.4)

ax[0].pcolormesh(qaxis_rad, eax,
	mreels.sigmoid(qmap_rad.T, (100, 120)), shading='nearest')
ax[0].set_xticks([0, 1.43])
ax[0].title.set_text("radial integration")

ax[1].pcolormesh(qaxis_line, eax,
	mreels.sigmoid(qmap_line.T, (50,100)), shading='nearest')
ax[1].set_xticks([0, 1.43])
ax[1].title.set_text('line integration')
ax[1].set_ylabel(r'energy loss $\Delta E$ [$eV$]')

ax[2].pcolormesh(qaxis_slice, eax,
	mreels.sigmoid(qmap_slice.T, (100, 120)), shading='nearest')
ax[2].set_xticks([0, 1.43])
ax[2].set_xticklabels([r"$\Gamma$", r"$M$"])
ax[2].title.set_text('stack slicing')
ax[2].set_xlabel(r"momentum transfer $q_{\perp}$ [$nm^{-1}$]")

plt.savefig('tech_comp.eps', format='eps')
plt.show()
