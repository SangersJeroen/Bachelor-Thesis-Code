import numpy as np
import mreels
import cProfile
import matplotlib.pyplot as plt


if __name__ == '__main__':
    eels_stack = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4', 25)
    print(eels_stack.stack.shape)
    eels_stack.correct_drift()
    qmap, qaxis = mreels.get_qeels_data(eels_stack, 1000, 1, 25, (171,288), 'line', threads=3)
    #np.save("line_1_GKGK_qmap", qmap)
    #np.save("line_1_GKGK_qaxis", qaxis)
    mreels.plot_qeels_data(eels_stack, qmap, qaxis, "1px_step_2brzones_")
    qmapdiff, qaxisnew, eaxisnew = mreels.scat_prob_differential(qmap, qaxis, eels_stack.axis0)
    test = 1
