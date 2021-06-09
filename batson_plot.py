import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mreels

string = '000-+0a11 '

data_obj = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4', 25)
data_obj.build_axes()

bat_map = np.load(string+'batson.npy')
qmap = np.load(string+'qmap.npy')
qaxis = np.load(string+'qaxis.npy')

mpl.rcParams['figure.dpi'] = 100
fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
plt.gcf().set_size_inches(7,7)


upper = 80
lower = 25
step = 8

mask = np.where( (data_obj.axis0 > -1) & (data_obj.axis0 < 35), True, False )

xax = data_obj.axis0[mask]

for i in range(lower, upper, step):
    ax[0].plot(xax, qmap[i,mask], color=str((i-lower)/(upper-lower)))
    ax[1].plot(xax, bat_map[i,mask], color=str((i-lower)/(upper-lower)))

    ax[0].text(xax[-1]-i/upper*6, qmap[i,mask].max(), "q={:.2f}".format(qaxis[i]))

ax[0].set_ylabel(r"Intensity")
ax[0].set_xlabel(r'Energy loss [$\Delta eV$]')
ax[0].set_title('uncorrected qmap')
ax[1].set_title('batson corrected qmap')
plt.show()