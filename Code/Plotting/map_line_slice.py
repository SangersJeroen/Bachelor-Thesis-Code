import mreels
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

mpl.rcParams['figure.dpi'] = 180
mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ["Computer Modern Roman"]
})

#Data obj
mreels_stack = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4', 25)
mreels_stack.build_axes()

#Loading the maps
qmap0, qaxis0 = mreels.get_qeels_slice(mreels_stack, (777,1373))

qmap0 = qmap0[1:,:]
qaxis0 = qaxis0[1:]

mask = np.where((2 <= mreels_stack.axis0) & (mreels_stack.axis0 <= 28), True, False)
eax = mreels_stack.axis0

upper = 25
lower = 0
step = 1


fig, ax = plt.subplots(1,1)
plt.gcf().set_size_inches(6.4,4)
for i in range(lower, upper, step):
    ax.plot(mreels_stack.axis0[mask], qmap0[i,mask]-500*i,
               color=str((i-lower)/(upper-lower+5)), label=str(qaxis0[i]), linewidth=1)

plt.axhline(qmap0[0,mask].max(), linestyle="--", color='red')
plt.text(20, qmap0[0,mask].max()-20000, r"$q_{\perp}=$"+"{:.2f}".format(qaxis0[0]), fontsize=12, color='red')
plt.axhline(qmap0[30].min()-500*30, linestyle="--", color='red')
plt.text(20, qmap0[30].min()-35000, r"$q_{\perp}=$"+"{:.2f}".format(qaxis0[30]), fontsize=12, color='red')

ax.vlines(3.5, ymin=-500*30, ymax=qmap0[0,np.argwhere(eax == 3.5)[0][0]], linestyle='-.', color='blue')
ax.vlines(7.25, ymin=-500*30, ymax=qmap0[0,np.argwhere(eax == 7.25)[0][0]], linestyle="-.", color='blue')
ax.vlines(22.25, ymin=-500*30, ymax=qmap0[0, np.argwhere(eax == 22.25)[0][0]], linestyle="-.", color='blue')

ax.set_ylabel(r'scattering Intensity')
ax.set_yticks([])
ax.set_xticks([0,5,10,15,20,25])
ax.xaxis.set_minor_locator(MultipleLocator(1))

ax.set_xlabel(r'energy loss $\Delta E$ [$eV$]')
ax.set_title(r"q-EELS spectrum")
plt.xlim([2,25])
plt.ylim([-40000,250000])
plt.savefig('track_peak.eps', format='eps')
plt.show()