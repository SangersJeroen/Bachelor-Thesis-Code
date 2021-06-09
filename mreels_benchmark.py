import numpy as np
import mreels
import cProfile
import matplotlib.pyplot as plt

string = '000-+K-+M '

if __name__ == '__main__':
    eels_stack = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4', 25)
    eels_stack.build_axes()
    #eels_stack.rem_neg_el()
    #eels_stack.remove_zlp(threads=12)
    #eels_stack.remove_neg_val()
    eels_stack.correct_drift()
    #qmap, qaxis = mreels.get_qeels_data(eels_stack, 1000, 1, 25, (171,288), 'line', threads=12)
    #qmap, qaxis = mreels.get_qeels_data(eels_stack, 1000, 1, 25, (771,778), 'line', threads=12)
    qmap0, qaxis0 = mreels.get_qeels_slice(eels_stack, (992, 812))
    qmap1, qaxis1 = mreels.get_qeels_slice(eels_stack, (1113, 766), starting_point=(992, 812))

    print(qmap0.shape)
    print(qmap1.shape)

    qmap = np.append(qmap0[:-1,], qmap1, axis=0)
    qaxis = np.append(qaxis0[:-1], qaxis1)

    qmap = qmap + np.abs(qmap.min())
    #qmaps, qaxiss = mreels.get_qeels_slice(eels_stack, (771, 778))
    np.save(string+'qmap.npy', qmap)
    np.save(string+'qaxis.npy', qaxis)


    mreels.plot_qeels_data(eels_stack, mreels.sigmoid(qmap, (40, 100)), qaxis, string+" ")