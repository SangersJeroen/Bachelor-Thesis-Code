from mreels import MomentumResolvedDataStack


import mreels
import cProfile

eels_stack = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-001 [-3,36] eV.dm4')
qmap, qaxis = mreels.get_qeels_data(eels_stack, 0, 10, 25, (1509, 991), method='line')

mreels.plot_qeels_data(eels_stack, qmap, qaxis)

test = 1