import numpy as np
import mreels
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['figure.dpi'] = 100
mpl.rcParams["legend.loc"] = "center right"

if __name__ == '__main__':
    eels_stack = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4', 25)
    eels_imag = mreels.ImagingSetup('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4')
    eels_stack.build_axes()
    eels_stack.rem_neg_el()
    eels_stack.remove_neg_val()
    qmap0, qaxis0 = mreels.get_qeels_slice(eels_stack, (777, 1373))
    qmap1, qaxis1 = mreels.get_qeels_slice(eels_stack, (627, 953))
    qmap2, qaxis2 = mreels.get_qeels_slice(eels_stack, (914, 601))

    eax = eels_stack.axis0

    #mreels.plot_qeels_data(eels_stack, mreels.sigmoid(qmap2, (20, 50)), qaxis2, " ")
    pool = 4

    qmap0, qaxis0 = mreels.pool_qmap(qmap0, qaxis0, pool)
    qmap1, qaxis1 = mreels.pool_qmap(qmap1, qaxis1, pool)
    qmap2, qaxis2 = mreels.pool_qmap(qmap2, qaxis2, pool)


    peak = 15
    window = 18

    ppos0, perr0 = mreels.find_peak_in_range(qmap0, np.argwhere(eax==peak)[0][0], window)
    ppos1, perr1 = mreels.find_peak_in_range(qmap1, np.argwhere(eax==peak)[0][0], window)
    ppos2, perr2 = mreels.find_peak_in_range(qmap2, np.argwhere(eax==peak)[0][0], window)

    plt.errorbar(qaxis0, eax[ppos0], yerr=perr0*0.25, fmt='+', label=r'1a12$', color='blue')
    plt.errorbar(qaxis1, eax[ppos1], yerr=perr1*0.25, fmt='1', label=r'0a11$', color='red')
    plt.errorbar(qaxis2, eax[ppos2], yerr=perr2*0.25, fmt='2', label=r'a10a1', color='green')

    plt.axhline(y=(int(peak-window/2*0.25)))
    plt.axhline(y=(int(peak+window/2*0.25)))

    plt.xlim([-0.1,1.1])
    plt.ylim([12,18])
    plt.xlabel(r'$q$ [$nm^{-1}$]')
    plt.ylabel(r'$\Delta eV$ [$eV$]')
    plt.title(r'Tracked peaks in $\Gamma \rightarrow [\overline{1} 1 \overline{2}]$')
    plt.legend()
    plt.show()