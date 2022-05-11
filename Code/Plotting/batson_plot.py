import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mreels

string = '000-+01a1 '

data_obj = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4', 25)
data_obj.build_axes()

bat_map = np.load(string+'batson.npy')
qmap = np.load(string+'qmap.npy')
qaxis = np.load(string+'qaxis.npy')

mreels.plot_qeels_data(data_obj, mreels.sigmoid(bat_map, (50,100)), qaxis, prefix=' ')





mpl.rcParams['figure.dpi'] = 100
fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
plt.gcf().set_size_inches(14,7)


upper = 200
lower = 30
step =  10
spread = 10000

mask = np.where( (data_obj.axis0 > 12) & (data_obj.axis0 < 16), True, False )

xax = data_obj.axis0[mask]

ax[0].axhline(qmap[lower,mask].min(), linestyle='--', color='red')
ax[1].axhline(qmap[lower,mask].min(), linestyle='--', color='red')

for i in range(lower, upper, step):
    ax[0].plot(xax, qmap[i,mask]-(i-lower)/(upper-lower)*spread, color=str((i-lower)/(upper-lower)*0.8))
    ax[1].plot(xax, bat_map[i,mask]-(i-lower)/(upper-lower)*spread, color=str((i-lower)/(upper-lower)*0.8))


ax[0].text(xax[-1], qmap[lower,mask].min(), r"$q=${:.2f}".format(qaxis[lower]), color='red')
ax[0].axhline(qmap[upper,mask].min()-spread, linestyle='--', color='red')
ax[0].text(xax[-1], qmap[upper,mask].min()-spread*0.9, r"$q=${:.2f}".format(qaxis[upper]), color='red')
ax[1].text(xax[-1], qmap[lower,mask].min(), r"$q=${:.2f}".format(qaxis[lower]), color='red')
ax[1].axhline(qmap[upper,mask].min()-spread, linestyle='--', color='red')
ax[1].text(xax[-1], qmap[upper,mask].min()-spread*0.9, r"$q=${:.2f}".format(qaxis[upper]), color='red')
ax[0].set_ylabel(r"Intensity")
ax[0].set_xlabel(r'Energy loss [$\Delta eV$]')
ax[1].set_xlabel(r'Energy loss [$\Delta eV$]')
ax[0].set_title('uncorrected qmap')
ax[1].set_title('batson corrected qmap')
plt.show()