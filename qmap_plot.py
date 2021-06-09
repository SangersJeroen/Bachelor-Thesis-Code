import mreels
import numpy as np
import matplotlib.pyplot as plt

data_obj = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4')
#data_obj.rem_neg_el()
data_obj.build_axes()
""" qmap = np.load('slice_1_GKGK_qmap_alt_1z.npy')
qaxis = np.load('slice_1_gkgk_qaxis_alt_1z.npy')
 """

qmap = np.load('qmap_GMKG.npy')
qaxis = np.load('qaxis_GMKG.npy')


mask = np.where( np.isnan(qaxis) | (qaxis == 0.0) , False, True)
qmap = qmap[mask]
qaxis = qaxis[mask]
min_q = qaxis.min()
max_q = qaxis.max()
step_q = (max_q-min_q)/len(qaxis)

data_obj.build_axes()
e = data_obj.axis0
min_e = e.min()
max_e = e.max()
step_e = (max_e-min_e)/len(e)

#Q, E = np.mgrid[min_q:max_q:step_q, min_e:max_e:step_e]
fig, ax = plt.subplots(1,2)
c = ax[0].pcolormesh(e, qaxis, mreels.sigmoid(qmap, (50,100)), shading='nearest')

ax[0].axhline(qaxis[353], color='red', linewidth=2)
ax[0].text(e[-1], qaxis[353], r"$M$", fontsize=12)

ax[0].axhline(qaxis[353+207-1], color='red', linewidth=2)
ax[0].text(e[-1], qaxis[353+207-1], r"$K$", fontsize=12)

ax[0].axhline(qaxis[353+207+447-1], color='red', linewidth=2)
ax[0].text(e[-1], qaxis[353+207+447-1], r"$\Gamma_1$", fontsize=12)

ax[0].axhline(qaxis[353+207+447+354-1], color='red', linewidth=2)
ax[0].text(e[-1], qaxis[353+207+447+354-1], r"$M_1$", fontsize=12)

ax[0].axhline(qaxis[-1], color='red', linewidth=2)
ax[0].text(e[-1], qaxis[-1], r"$K_1$", fontsize=12)

plt.gcf().set_size_inches((16,8))
cbar = fig.colorbar(c, ax=ax[0])
cbar.set_label('Arbitrary Scale', rotation=90)
ax[0].set_xlabel(r"Energy Loss [$eV$]")
ax[0].set_ylabel(r"$q$ [$nm^{-1}$]")

centre = data_obj.get_centre(25)
my, mx = (1071-471)/2+471, (1023-533)/2+533
disty, distx = my-centre[0], mx-centre[1]


ax[1].pcolorfast(mreels.sigmoid(data_obj.stack[24]).T)
ax[1].plot(1071, 1023, marker='1', markersize=10, color='yellow')
ax[1].text(1100, 1105, r'$\Gamma$', color='yellow', fontsize=15)
ax[1].plot(my, mx, marker='2', markersize=10, color='orange')
ax[1].text(my+50, mx+50, r'$M$', fontsize=15, color='orange')
ax[1].plot(914, 601, marker=1, markersize=10, color='yellow')
ax[1].text(914, 601, r'$K$', fontsize=15, color='yellow')
ax[1].set_yticks([y for y in range(0,len(data_obj.axis1),70)])
ax[1].set_xticks([x for x in range(0,len(data_obj.axis2),70)])
ax[1].set_yticklabels(["{:.2f}".format(data_obj.axis1[y]) for y in range(0, len(data_obj.axis1), 70)])
ax[1].set_xticklabels(["{:.2f}".format(data_obj.axis2[y]) for y in range(0, len(data_obj.axis2), 70)], rotation=90)
ax[1].set_xlabel(r'$q_y$ [$nm^{-1}$]')
ax[1].set_ylabel(r'$q_x$ [$nm^{-1}$]')


fig.savefig('qmap G M K G1 M1 K1.png')
plt.show()