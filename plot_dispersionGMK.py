import numpy as np
import mreels
import matplotlib.pyplot as plt

string = '000-+a101 '

if __name__ == '__main__':
    eels_stack = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4', 25)
    eels_imag = mreels.ImagingSetup('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4')
    eels_stack.build_axes()
    eels_stack.rem_neg_el()
    eels_stack.remove_neg_val()
    eels_stack.correct_drift()
    #eels_stack.stack = np.load('no_zlp_stack.npy')
    #qmap, qaxis = mreels.get_qeels_data(eels_stack, 750, 1, 25, threads=8)
    #qmap, qaxis = mreels.get_qeels_data(eels_stack, window00, 1, 25, (992,812), 'line', threads=12)
    qmap_m, qaxis_m = mreels.get_qeels_slice(eels_stack, (992, 812))
    qmap_k, qaxis_k = mreels.get_qeels_slice(eels_stack, (1113, 766), starting_point=(992, 812))
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

    qmap = np.append(qmap_m, qmap_k[1:], axis=0)
    qaxis = np.append(qaxis_m, qaxis_k[1:])

    mreels.plot_qeels_data(eels_stack, mreels.sigmoid(qmap, (100, 150)), qaxis, string+" ")
    #mreels.plot_qeels_data(eels_stack, mreels.sigmoid(qmap_m, (50, 150)), qaxis_m, string+" ")

    #np.save('test fp ' +'batson.npy', batson_map)

    peak = 15
    window = 14

    qmap, qaxis = mreels.pool_qmap(qmap, qaxis, 2)

    peak1 = 7
    window1 = 6

    peak2 = 22
    window2 = 8


    ppos, perr = mreels.find_peak_in_range(qmap, np.argwhere(eax==peak)[0][0], window)
    ppos1, perr1 = mreels.find_peak_in_range(qmap, np.argwhere(eax==peak1)[0][0], window1)
    ppos2, perr2 = mreels.find_peak_in_range(qmap, np.argwhere(eax==peak2)[0][0], window2)

    plt.errorbar(qaxis, eax[ppos], yerr=perr*0.25, fmt='+', label=r'~15$eV$', color='blue')
    plt.errorbar(qaxis, eax[ppos1], yerr=perr1*0.25, fmt='+', label=r'~7$eV$', color='red')
    plt.errorbar(qaxis, eax[ppos2], yerr=perr2*0.25, fmt='+', label=r'~22$eV$', color='green')
    plt.xlim([-0.1,qaxis.max()])
    plt.ylim([0,25])
    plt.xlabel(r'$q$ [$nm^{-1}$]')
    plt.ylabel(r'$\Delta eV$ [$eV$]')
    plt.title(r'Tracked peaks in $\Gamma \rightarrow K \rightarrow M$')
    plt.legend()
    plt.show()