import mreels
import cProfile
import matplotlib.pyplot as plt


if __name__ == '__main__':
    eels_stack = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-002 [-3,45] eV.dm4')
    eels_stack.correct_drift(25)
    qmap, qaxis = mreels.get_qeels_data(eels_stack, 500, 50, 25, threads=4)
    mreels.plot_qeels_data(eels_stack, qmap, qaxis, "lowradial")


    test = 1