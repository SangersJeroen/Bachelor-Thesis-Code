import numpy as np
import mreels
import matplotlib.pyplot as plt

string = '000-+a101 '

if __name__ == '__main__':
    eels_stack = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4', 25)
    eels_imag = mreels.ImagingSetup('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4')
    eels_stack.build_axes()
    eels_stack.rem_neg_el()
    #eels_stack.remove_neg_val()
    #eels_stack.correct_drift()
    eels_stack.stack = np.load('no_zlp_stack.npy')
    #qmap, qaxis = mreels.get_qeels_data(eels_stack, 750, 1, 25, threads=8)
    #qmap, qaxis = mreels.get_qeels_data(eels_stack, window00, 1, 25, (992,812), 'line', threads=12)
    qmap_m, qaxis_m = mreels.get_qeels_slice(eels_stack, (992, 812))
    qmap_k, qaxis_k = mreels.get_qeels_slice(eels_stack, (1113, 766))
    qmap_p, qaxis_p = mreels.get_qeels_slice(eels_stack, (627, 953))
    qmap_r, qaxis_r = mreels.get_qeels_slice(eels_stack, (1506, 1096))
    #qmap1, qaxis1 = mreels.get_qeels_slice(eels_stack, (1113, 766), starting_point=(992, 812))
    #np.save('no_zlp_stack.npy', eels_stack.stack)
    #print(qmap0.shape)
    #print(qmap1.shape)

    #qmap = np.append(qmap0[:-1,], qmap1, axis=0)
    #qaxis = np.append(qaxis0[:-1], qaxis1)

    #qmap = qmap + np.abs(qmap.min())
    #qmaps, qaxiss = mreels.get_qeels_slice(eels_stack, (771, 778))
    #np.save(string+'qmap.npy', qmap)
    #np.save(string+'qaxis.npy', qaxis)
    eax = eels_stack.axis0
    print(eax)

    #qmap = np.load(string+'qmap.npy')
    #qaxis = np.load(string+'qaxis.npy')

    mreels.plot_qeels_data(eels_stack, mreels.sigmoid(qmap_k, (50, 150)), qaxis_k, string+" ")
    #mreels.plot_qeels_data(eels_stack, mreels.sigmoid(qmap_m, (50, 150)), qaxis_m, string+" ")

    #np.save('test fp ' +'batson.npy', batson_map)

    peak = 15
    window = 14

    qmap_k, qaxis_k = mreels.pool_qmap(qmap_k, qaxis_k, 4)
    qmap_m, qaxis_m = mreels.pool_qmap(qmap_m, qaxis_m, 4)
    qmap_p, qaxis_p = mreels.pool_qmap(qmap_p, qaxis_p, 4)
    qmap_r, qaxis_r = mreels.pool_qmap(qmap_r, qaxis_r, 4)


    ppos_k, perr_k = mreels.find_peak_in_range(qmap_k, np.argwhere(eax==peak)[0][0], window)
    ppos_m, perr_m = mreels.find_peak_in_range(qmap_m, np.argwhere(eax==peak)[0][0], window)
    ppos_p, perr_p = mreels.find_peak_in_range(qmap_p, np.argwhere(eax==peak)[0][0], window)
    ppos_r, perr_r = mreels.find_peak_in_range(qmap_r, np.argwhere(eax==peak)[0][0], window)

    plt.errorbar(qaxis_r, eax[ppos_r], yerr=perr_r*0.25, fmt='s', label='000 -> 0a11')
    plt.errorbar(qaxis_k, eax[ppos_k], yerr=perr_k*0.25, fmt='+', label='000 -> k', color='blue')
    plt.errorbar(qaxis_m, eax[ppos_m], yerr=perr_m*0.25, fmt='x', label='000 -> m')
    plt.errorbar(qaxis_p, eax[ppos_p], yerr=perr_p*0.25, fmt='.', label='000 -> 01a1')
    plt.xlim([-0.1,1.5])
    plt.ylim([0,25])
    plt.xlabel(r'$q$ [$nm^{-1}$]')
    plt.ylabel(r'$\Delta eV$ [$eV$]')
    plt.title('~21eV peak')
    plt.legend()
    plt.show()