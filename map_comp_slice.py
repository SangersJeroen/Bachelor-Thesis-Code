import mreels
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 100

string0 = '000-+1a12'
string1 = '000-+01a1'

#Data obj
mreels_stack = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4', 25)
mreels_stack.build_axes()

#Loading the maps
qmap0 = np.load(string0 + " qmap.npy")
qaxis0 = np.load(string0 + " qaxis.npy")
qmap1 = np.load(string1 + " qmap.npy")
qaxis1 = np.load(string1 + " qaxis.npy")

mask = np.where((0 <= mreels_stack.axis0) & (mreels_stack.axis0 <= 2), True, False)


upper = 50
lower = 5
step = 5


fig, ax = plt.subplots(1,2, sharey=True, sharex=True)
plt.gcf().set_size_inches(9,7)
for i in range(lower, upper, step):
    ax[0].plot(mreels_stack.axis0[mask], qmap0[i,mask]-1000*i,
               color=str((i+5)/upper), label=str(qaxis0[i]))
    ax[1].plot(mreels_stack.axis0[mask], qmap1[i,mask]-1000*i,
               color=str((i+5)/upper), label=str(qaxis0[i]))

    ax[0].text(mreels_stack.axis0[mask][-2]-i/upper*0.5, qmap0[i,mask][-2]-1000*i,
               r'{:.2f}'.format(qaxis0[i]), fontsize=10, color='black')
    ax[1].text(mreels_stack.axis0[mask][-2]-i/upper*0.5, qmap1[i,mask][-2]-1000*i,
               r'{:.2f}'.format(qaxis0[i]), fontsize=10)

ax[0].set_ylabel(r'Scattering Intensity')
ax[0].set_yticks([])

ax[0].set_xlabel(r'Energy loss [$\Delta eV$]')
ax[0].set_title(string0)
ax[1].set_xlabel(r'Energy loss [$\Delta eV$]')
ax[1].set_title(string1)

plt.savefig('comparing '+string0 + ' '+string1)
plt.show()