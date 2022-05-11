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
data.build_axes()

eax = data.axis0

yc, xc = data.get_centre()

fig, ax = plt.subplots(1,1)

plt.gcf().set_size_inches(6.4,4)

ax.plot(eax, data.stack[:, yc, xc])
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.set_yticks([])
ax.set_xlim([-1, 36])
ax.set_xlabel(r"energy loss $\Delta E$ [$eV$]")
ax.set_ylabel(r"intensity [arb. units]")
plt.savefig('big-spectrum.eps', format='eps')
plt.show()