import mreels
import cProfile
import matplotlib.pyplot as plt


if __name__ == '__main__':
    eels_stack = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-003 [-3,36] eV.dm4')
    qmap, qaxis = mreels.get_qeels_data(eels_stack, 0, 5, 25, (537,467), method='line', threads=12)
    mreels.plot_qeels_data(eels_stack, qmap, qaxis, "low")

    eels_stack, qmap, qaxis = None, None, None
    eels_stack = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4')
    qmap, qaxis = mreels.get_qeels_data(eels_stack, 0, 5, 25, (533, 471), method='line', threads=12)
    mreels.plot_qeels_data(eels_stack, qmap, qaxis, "high")

    test = 1