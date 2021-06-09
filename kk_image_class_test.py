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
eels_stack.correct_drift()

cy, cx = eels_stack.get_centre(25)

spectrum = eels_stack.stack[:,cy,cx]

n = 2.07 #https://doi.org/10.1016/0040-6090(87)90320-8

n_ZLP = eels_stack.stack.max()

eps, te, srfint = im.kramers_kronig_hs(spectrum, n_ZLP, 1, n, delta=0.25)

qmap = np.load('qmap_line_int.npy')
qaxis = np.load('qaxis_line_int.npy')
qmap_kk = np.zeros(qmap.shape)[:,1:-1]





eels_stack.rem_neg_el()
mpl.use('TkAgg')

plt.plot(eels_stack.axis0[1:-1], np.real(eps), label=r'$\Re{(\epsilon)}$')
plt.plot(eels_stack.axis0[1:-1], np.imag(eps), label=r'$\Im{(\epsilon)}$')
#plt.plot(eels_stack.axis0[1:-1], srfint, label=r'$srfint$')
plt.xlabel(r"Energy Loss $eV$")
plt.ylabel(r"$\epsilon$")
plt.title(r"$\Re{(\epsilon)}$ and $\Im{(\epsilon)}$ at $\| q \| = 0$")
plt.legend()
plt.savefig('kk_beam_centre.png')
plt.show()