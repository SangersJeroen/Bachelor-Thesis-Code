import numpy as np
from numpy.lib.type_check import imag
import mreels
import matplotlib.pyplot as plt
from mreels import sigmoid as sig

string = '000-+a101 '

if __name__ == '__main__':
    eels_stack = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4', 25)
    eels_imag = mreels.ImagingSetup('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4')
    eels_stack.build_axes()
    #eels_stack.rem_neg_el()
    #eels_stack.remove_neg_val()
    eels_stack.correct_drift()
    #eels_stack.stack = np.load('no_zlp_stack.npy')
    #qmap, qaxis = mreels.get_qeels_data(eels_stack, 750, 1, 25, threads=8)
    #qmap, qaxis = mreels.get_qeels_data(eels_stack, window00, 1, 25, (992,812), 'line', threads=12)

    yc, xc = eels_stack.get_centre()

    y110, x110 = 471, 533

    ay110, ax110 = 2*yc-y110, 2*xc-x110

    qmap_m, qaxis_m = mreels.get_qeels_slice(eels_stack, (1354, 675), starting_point=(627, 953))

    tp = np.argwhere(qaxis_m == qaxis_m.min())[0][0]

    qaxis_m[0:tp-1] *= -1

    mreels.plot_qeels_data(eels_stack, sig(qmap_m, (50,100)), qaxis_m, " ")

    pqmap, pqaxis = mreels.pool_qmap(qmap_m, qaxis_m, 4)
    ppos, perr = mreels.find_peak_in_range(pqmap, 15, 10)
    plt.errorbar(pqaxis, ppos, perr*0.25, fmt='x')
    #plt.xlim([-0.1,1.5])
    plt.ylim([0,25])
    plt.xlabel(r'$q$ [$nm^{-1}$]')
    plt.ylabel(r'$\Delta eV$ [$eV$]')
    plt.title('~15eV peak')
    plt.legend()
    plt.show()
