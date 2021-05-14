import mreels
import cProfile

if __name__ == '__main__':
    eels_stack = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-001 [-3,36] eV.dm4')
    qmap, qaxis = mreels.get_qeels_data(eels_stack, 400, 40, 25)

    mreels.plot_qeels_data(eels_stack, qmap, qaxis)

    test = 1