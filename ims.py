import numpy as np
import matplotlib.pyplot as plt
import mreels

data_obj = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4', 25)
data_obj.build_axes()

qmap_kk = np.load('qmap_kk.npy')
qaxis = np.load('qaxis_line_int_for_kk.npy')
eaxis = data_obj.axis0[data_obj.axis0>0][:-1]

lower = 50
upper = 750

fig, ax = plt.subplots(1,2)
plt.gcf().set_size_inches(12,6)
plt.subplots_adjust(wspace=0.5)
for i in range(lower, upper, 50):
    ax[0].plot(eaxis, np.real(qmap_kk[i,:])+qaxis[i])
    ax[0].text(eaxis[-1], np.real(qmap_kk[i,-1]+qaxis[i]), r'$q=${:.2f}'.format(np.real(qmap_kk[i,-1])))
ax[0].set_xlabel(r'Energy loss [$eV$]')
ax[0].set_ylabel(r'$\Re{(\epsilon)}$ Offset')

for i in range(lower, upper, 50):
    ax[1].plot(eaxis, np.imag(qmap_kk[i,:])+qaxis[i])
    ax[1].text(eaxis[-1], np.imag(qmap_kk[i,-1])+qaxis[i], r'$q=${:.2f}'.format(np.imag(qmap_kk[i,-1])))
ax[1].set_xlabel(r'Energy loss [$eV$]')
ax[1].set_ylabel(r'$\Im{(\epsilon)}$ Offset')

plt.savefig('slices.png')
plt.show()