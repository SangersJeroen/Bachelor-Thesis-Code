import mreels
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    datastack = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-003 [-3,36] eV.dm4')
    im_setup = mreels.ImagingSetup('n-inse_C1_EFTEM-SI-003 [-3,36] eV.dm4')
    datastack.correct_drift(25)
    datastack.build_axes()
    omega, ky, kx = mreels.transform_axis(datastack, im_setup)
    #scat_prob, omega, ky, kx = mreels.scat_prob_differential()

    test = True