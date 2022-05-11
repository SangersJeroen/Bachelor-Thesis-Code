import numpy as np
import mreels
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.optimize as opt


mpl.rcParams['figure.dpi'] = 180
mpl.rcParams["legend.loc"] = "best"
mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ["Computer Modern Roman"]
})

if __name__ == '__main__':
    eels_stack = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4', 25)
    eels_imag = mreels.ImagingSetup('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4')
    eels_stack.build_axes()
    eels_stack.rem_neg_el()
    eels_stack.remove_neg_val()
    qmap0, qaxis0 = mreels.get_qeels_slice(eels_stack, (777, 1373))
    qmap1, qaxis1 = mreels.get_qeels_slice(eels_stack, (627, 953))
    qmap2, qaxis2 = mreels.get_qeels_slice(eels_stack, (914, 601))

    imspec = np.sum(eels_stack.stack[:,700:1400,800:1300], axis=(2,1))

    bat0 = mreels.batson_correct(eels_stack, 3, qmap0, imspec)
    bat1 = mreels.batson_correct(eels_stack, 3, qmap1, imspec)
    bat2 = mreels.batson_correct(eels_stack, 3, qmap2, imspec)

    eax = eels_stack.axis0
    #mreels.plot_qeels_data(eels_stack, mreels.sigmoid(qmap2, (20, 50)), qaxis2, " ")
    pool = 6

    #qmap0, qaxis0 = mreels.pool_qmap(qmap0, qaxis0, pool)

    print(np.argwhere(qaxis0 > 0.4)[0][0])

    #qmap0[58:,:] = bat0[58:,:]
    #qmap1[58:,:] = bat1[58:,:]
    #qmap2[58:,:] = bat2[58:,:]


    peak = 14.5
    window = 6

    qmap0, qaxis0 = mreels.pool_qmap(qmap0, qaxis0, pool)
    qmap1, qaxis1 = mreels.pool_qmap(qmap1, qaxis1, pool)
    qmap2, qaxis2 = mreels.pool_qmap(qmap2, qaxis2, pool)

    mask0 = np.where(qaxis0<0.7, True, False)
    mask1 = np.where(qaxis1<0.7, True, False)
    mask2 = np.where(qaxis2<0.7, True, False)


    ppos0, perr0 = mreels.find_peak_in_range(qmap0[mask0],
        np.argwhere(eax==peak)[0][0],
        window, adaptive_range=True
        )

    ppos1, perr1 = mreels.find_peak_in_range(qmap1[mask1],
        np.argwhere(eax==peak)[0][0],
        window, adaptive_range=True
        )
    ppos2, perr2 = mreels.find_peak_in_range(qmap2[mask2],
        np.argwhere(eax==peak)[0][0],
        window, adaptive_range=True
        )

    err_sc = 10

    def fitfunc(x, a, b):
        return a*np.sqrt(x) + b

    popt0, pcov0 = opt.curve_fit(fitfunc, qaxis0[mask0], ppos0 )
    popt1, pcov1 = opt.curve_fit(fitfunc, qaxis1[mask1], ppos1 )
    popt2, pcov2 = opt.curve_fit(fitfunc, qaxis2[mask2], ppos2 )

    plt.plot(qaxis0[mask0], 0.25*fitfunc(qaxis0[mask0], *popt0), linestyle="--", color="blue")
    plt.plot(qaxis1[mask1], 0.25*fitfunc(qaxis1[mask1], *popt1), linestyle="--", color="red")
    plt.plot(qaxis2[mask2], 0.25*fitfunc(qaxis2[mask2], *popt2), linestyle="--",color="green")


    plt.errorbar(qaxis0[mask0], eax[ppos0], yerr=perr0*err_sc, fmt='+',
        label=r'$\Gamma \rightarrow \overline{1} 1 \overline{2}$', color='blue')
    plt.errorbar(qaxis1[mask1], eax[ppos1], yerr=perr1*err_sc, fmt='1',
        label=r'$\Gamma \rightarrow 0 1 \overline{1}$', color='red')
    plt.errorbar(qaxis2[mask2], eax[ppos2], yerr=perr2*err_sc, fmt='2',
        label=r'$\Gamma \rightarrow 1 0 1$', color='green')

    plt.gcf().set_size_inches(6.4, 4)

    plt.xlim([0,0.7])
    plt.ylim([13,16])
    plt.xlabel(r'$\Delta q_{\perp}$ [$nm^{-1}$]')
    plt.ylabel(r'$\Delta E$ [$eV$]')
    plt.title(r'Tracked peaks left, window={}'.format(window))
    plt.legend()
    plt.savefig('plasmon_dispersion.eps', format='eps')
    plt.show()