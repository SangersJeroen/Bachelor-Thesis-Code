import numpy as np
import mreels
import cProfile
import matplotlib.pyplot as plt


if __name__ == '__main__':
    eels_stack = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4', 25)
    setup_obj = mreels.ImagingSetup('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4')
    eels_stack.build_axes()
    eels_stack.rem_neg_el()
    eels_stack.remove_zlp((700,700))
    eels_stack.remove_neg_val()
    eels_stack.correct_drift()
    #eels_stack.build_axes()
    #eels_stack.correct_drift()
    #qmap, qaxis = mreels.get_qeels_data(eels_stack, 1000, 1, 25, (171,288), 'line', threads=12)
    oaxis, kyaxis, kxaxis = mreels.transform_axis(eels_stack, setup_obj)
    qmap, qaxis = mreels.get_qeels_slice(eels_stack, (771, 778))
    scat_prob, kaxis, oaxis = mreels.scat_prob_differential(qmap, qaxis, oaxis)
    thick = 0.8e-9 #thickness of a layer of InSe from https://doi.org/10.1002/adpr.202000025
    e0 = 8.987551e9

    Omega, Kaxis = np.meshgrid(oaxis, kaxis)
    Omega = Omega.astype('complex128')
    Kaxis = Kaxis.astype('complex128')
    di_func = Omega*0.00001j

    kroger_terms = mreels.KrogerTerms(setup_obj, di_func, oaxis, kaxis, e0, thick)
    guess, it, errs = mreels.dif_guesser(scat_prob, kroger_terms, Omega, Kaxis, start_mag=1)

    test = 1
