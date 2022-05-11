import numpy as np
import mreels
import matplotlib.pyplot as plt
import matplotlib as mpl

string = '000-+a101 '
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams["legend.loc"] = "center"

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
    qmap, qaxis = mreels.get_qeels_slice(eels_stack, (471, 533))
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

    mreels.plot_qeels_data(eels_stack, mreels.sigmoid(qmap, (20, 150)), qaxis, string+" ")
    #mreels.plot_qeels_data(eels_stack, mreels.sigmoid(qmap_m, (50, 150)), qaxis_m, string+" ")

    #np.save('test fp ' +'batson.npy', batson_map)
    qmap, qaxis = mreels.pool_qmap(qmap, qaxis, 4)

    peak = 15
    window = 18

    peak1 = 27
    window1 = 6

    peak2 = 22
    window2 = 10

    peak3 = 25
    window3 = 6

    peak4 = 3.5
    window4 = 4

    peak5 = 6.5
    window5 = 6

    ppos, perr = mreels.find_peak_in_range(qmap, np.argwhere(eax==peak)[0][0], window)
    ppos1, perr1 = mreels.find_peak_in_range(qmap, np.argwhere(eax==peak1)[0][0], window1)
    ppos2, perr2 = mreels.find_peak_in_range(qmap, np.argwhere(eax==peak2)[0][0], window2)
    ppos3, perr3 = mreels.find_peak_in_range(qmap, np.argwhere(eax==peak3)[0][0], window3)
    ppos4, perr4 = mreels.find_peak_in_range(qmap, np.argwhere(eax==peak4)[0][0], window4)
    ppos5, perr5 = mreels.find_peak_in_range(qmap, np.argwhere(eax==peak5)[0][0], window5)

    plt.errorbar(qaxis, eax[ppos], yerr=perr*0.25, fmt='+', label=r'~15$eV$', color='blue')
    plt.errorbar(qaxis, eax[ppos1], yerr=perr1*0.25, fmt='+', label=r'~27$eV$', color='red')
    plt.errorbar(qaxis, eax[ppos2], yerr=perr2*0.25, fmt='+', label=r'~22$eV$', color='green')
    plt.errorbar(qaxis, eax[ppos3], yerr=perr3*0.25, fmt='+', label=r'~25$eV$', color='orange')
    plt.errorbar(qaxis, eax[ppos4], yerr=perr4*0.25, fmt='+', label=r'~3.5$eV$', color='gray')
    plt.errorbar(qaxis, eax[ppos5], yerr=perr5*0.25, fmt='+', label=r'~6.5$eV$', color='black')

    plt.xlim([-0.1,qaxis.max()])
    plt.ylim([0,25])
    plt.xlabel(r'$q$ [$nm^{-1}$]')
    plt.ylabel(r'$\Delta eV$ [$eV$]')
    plt.title(r'Tracked peaks in $\Gamma \rightarrow K \rightarrow M$')
    plt.legend()
    plt.show()