import mreels
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 100

string0 = '000-+K-+M'

#Data obj
mreels_stack = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4', 25)
mreels_stack.build_axes()

#Loading the maps
qmap0 = np.load(string0 + " qmap.npy")
qaxis0 = np.load(string0 + " qaxis.npy")

mask = np.where((0 <= mreels_stack.axis0) & (mreels_stack.axis0 <= 2), True, False)


upper = len(qaxis0)-1
lower = 270
step = 1


fig, ax = plt.subplots(1,1)
plt.gcf().set_size_inches(9,7)
for i in range(lower, upper, step):
    ax.plot(mreels_stack.axis0[mask], qmap0[i,mask]-10*i,
               color=str((i-lower)/(upper-lower)), label=str(qaxis0[i]))

    ax.text(mreels_stack.axis0[mask][-2]-i/upper*0.5, qmap0[i,mask].min()-10*i,
               r'{:.2f}'.format(qaxis0[i]), fontsize=10, color='black')

ax.set_ylabel(r'Scattering Intensity')
ax.set_yticks([])

ax.set_xlabel(r'Energy loss [$\Delta eV$]')
ax.set_title(string0 + r" $1.48 \leq q \leq 1.65$")

plt.savefig('line_trace '+string0)
plt.show()