import image_class_bs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import mreels

im = image_class_bs.Spectral_image.load_data('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4')
eels_stack = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4', 25)
eels_stack.build_axes()
eels_stack.remove_zlp((700,700))
eels_stack.remove_neg_val()
#eels_stack.correct_drift()

qmap = np.load('qmap_GMKG.npy')
qaxis = np.load('qaxis_GMKG.npy')

cy, cx = eels_stack.get_centre(25)
n = 2.07 #https://doi.org/10.1016/0040-6090(87)90320-8

n_ZLP = np.nanmax(qmap)

spectrum = eels_stack.stack[:,cy,cx]
eps, te, srfint = im.kramers_kronig_hs(spectrum, n_ZLP, 1, n, delta=0.25)


qmap_kk = np.zeros((qmap.shape[0],len(eps)), dtype='complex128')

for i in range(qmap.shape[0]):
    spectrum = qmap[i,:]
    eps, te, srfint = im.kramers_kronig_hs(spectrum, n_ZLP, 5, n, delta=0, correct_S_s=True)
    qmap_kk[i,:] = eps


mpl.use('TkAgg')

np.save("qmap_kk_GMKG.npy", qmap_kk)


fig, ax = plt.subplots(1,3)

c = ax[0].pcolormesh(eels_stack.axis0, qaxis, qmap, shading='nearest')
plt.gcf().set_size_inches((24,8))
cbar = fig.colorbar(c, ax=ax[0])
cbar.set_label('Arbitrary Scale', rotation=90)
ax[0].set_xlabel(r"Energy [$eV$]")
ax[0].set_ylabel(r"$q$ [$nm^{-1}$]")

d = ax[1].pcolormesh(eels_stack.axis0[eels_stack.axis0 > 0][:-1], qaxis, np.real(qmap_kk), shading='nearest')
ax[1].set_xlabel(r"Energy [$eV$]")
ax[1].set_ylabel(r"$q$ [$nm^{-1}$]")
dbar = fig.colorbar(d, ax=ax[1])
dbar.set_label(r"$\Re{(\epsilon)}$")

e = ax[2].pcolormesh(eels_stack.axis0[eels_stack.axis0 > 0][:-1], qaxis, np.imag(qmap_kk), shading='nearest')
ax[2].set_xlabel(r"Energy [$eV$]")
ax[2].set_ylabel(r"$q$ [$nm^{-1}$]")
ebar = fig.colorbar(e, ax=ax[2])
ebar.set_label(r"$\Im{(\epsilon)}$")

plt.savefig("di_elec_line_int.png")
plt.show()

