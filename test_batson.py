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
    eels_stack.rem_neg_el()
    eels_stack.remove_neg_val()
    #eels_stack.correct_drift()
    #eels_stack.stack = np.load('no_zlp_stack.npy')
    #qmap_m, qaxis_m = mreels.get_qeels_data(eels_stack, 300, 1, 25, threads=8)
    #qmap_m, qaxis_m = mreels.get_qeels_data(eels_stack, 750, 1, 25, (627,953), 'line', threads=12)
    qmap_m, qaxis_m = mreels.get_qeels_slice(eels_stack, (992, 812))
    #500:1000,620:900
    imspec = np.sum(eels_stack.stack[:,700:1400,800:1300], axis=(2,1))

    mreels.plot_qeels_data(eels_stack, sig(qmap_m), qaxis_m, " ")

    bat_map = mreels.batson_correct(eels_stack, 3, qmap_m, imspec=imspec)

    mreels.plot_qeels_data(eels_stack, sig(bat_map), qaxis_m, " ")



    mreels.plot_qeels_data(eels_stack, sig(qmap_m), qaxis_m, " ")

    eax = eels_stack.axis0

    peak = 15
    window = 20

    qmap_m, qaxis_m = mreels.pool_qmap(bat_map, qaxis_m, 2)

    ppos_m, perr_m = mreels.find_peak_in_range(qmap_m, np.argwhere(eax==peak)[0][0], window)
    ppos_1, perr_1 = mreels.find_peak_in_range(qmap_m, np.argwhere(eax==22)[0][0], 8)

    plt.errorbar(qaxis_m, eax[ppos_m], yerr=perr_m*0.25, fmt='x', label='15')
    plt.errorbar(qaxis_m, eax[ppos_1], yerr=perr_1*0.25, fmt='x', label='22')


    plt.xlim([-0.1,1.5])
    plt.ylim([0,25])
    plt.xlabel(r'$q$ [$nm^{-1}$]')
    plt.ylabel(r'$\Delta eV$ [$eV$]')
    plt.title('~21eV peak')
    plt.legend()
    plt.show()
