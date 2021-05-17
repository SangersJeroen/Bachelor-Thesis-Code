import mreels
import cProfile
import matplotlib.pyplot as plt


if __name__ == '__main__':

    """
    eels_stack = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-003 [-3,36] eV.dm4')
    eels_stack.correct_drift(25)

    qmap, qaxis = None, None
    """
    eels_stack = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4')
    #eels_stack.correct_drift(25)
    #qmap, qaxis = mreels.get_qeels_data(eels_stack, 850, 2, 25, threads=4)
    #mreels.plot_qeels_data(eels_stack, qmap, qaxis, "25meV_res_radial")
    #qmap, qaxis = None, None
    qmap, qaxis = mreels.get_qeels_data(eels_stack, 850, 2, 25, (471,533), method='line', threads=4)
    mreels.plot_qeels_data(eels_stack, qmap, qaxis, "25meV_res_line")

