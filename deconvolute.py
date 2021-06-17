import numpy as np
from scipy.fftpack import next_fast_len


def deconvolute(eels_obj, spec):

    y = spec
    r = 3  # Drude model, can also use estimation from exp. data
    A = y[-1]
    n_times_extra = 2
    sem_inf = next_fast_len(n_times_extra * eels_obj.axis_0_steps)

    l = eels_obj.axis_0_steps
    ddeltaE = eels_obj.axis_0_scale
    deltaE = eels_obj.axis0

    y_extrp = np.zeros(sem_inf)
    y_ZLP_extrp = np.zeros(sem_inf)
    x_extrp = np.linspace(deltaE[0] - l * ddeltaE,
                          sem_inf * ddeltaE + deltaE[0] - l * ddeltaE, sem_inf)

    x_extrp = np.linspace(deltaE[0], sem_inf * ddeltaE + deltaE[0], sem_inf)

    y_ZLP_extrp[:l] = spec
    y_ZLP_extrp[int(2/0.25):l] = 0
    y_extrp[:l] = y
    x_extrp[:l] = deltaE[::-1]

    y_extrp[l:] = A * np.power(1 + x_extrp[l:] - x_extrp[l], -r)

    x = x_extrp
    y = y_extrp
    y_ZLP = y_ZLP_extrp

    z_nu = CFT(x, y_ZLP)
    i_nu = CFT(x, y)
    abs_i_nu = np.absolute(i_nu)
    # scipy.integrate.cumtrapz(y_ZLP, x, initial=0)[-1]#1 #arbitrary units??? np.sum(EELZLP)
    N_ZLP = 1

    s_nu = N_ZLP * np.log(i_nu / z_nu)
    j1_nu = z_nu * s_nu / N_ZLP
    S_E = np.real(iCFT(x, s_nu))
    s_nu_nc = s_nu
    s_nu_nc[500:-500] = 0
    S_E_nc = np.real(iCFT(x, s_nu_nc))
    J1_E = np.real(iCFT(x, j1_nu))

    return J1_E[:l]


def CFT(x, y):
    x_0 = np.min(x)
    N_0 = np.argmin(np.absolute(x))
    N = len(x)
    x_max = np.max(x)
    delta_x = (x_max - x_0) / N
    k = np.linspace(0, N - 1, N)
    # np.exp(-1j*(x_0)*k*delta_omg)*delta_x
    cont_factor = np.exp(2j * np.pi * N_0 * k / N) * delta_x
    F_k = cont_factor * np.fft.fft(y)
    return F_k


def iCFT(x, Y_k):
    x_0 = np.min(x)
    N_0 = np.argmin(np.absolute(x))
    x_max = np.max(x)
    N = len(x)
    delta_x = (x_max - x_0) / N
    k = np.linspace(0, N - 1, N)
    cont_factor = np.exp(-2j * np.pi * N_0 * k / N)
    f_n = np.fft.ifft(cont_factor * Y_k) / delta_x
    return f_n
