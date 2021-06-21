import mreels
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cv

mpl.rcParams['figure.dpi'] = 180
mpl.rcParams["legend.loc"] = "best"
mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ["Computer Modern Roman"]
})

data = mreels.MomentumResolvedDataStack(
	'n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4',
	25
)

#data.rem_neg_el()
data.remove_neg_val()
#data.correct_drift()

gamma = data.get_centre()
m = (992, 812)
k = (1113, 766)

data.build_axes()
eax = data.axis0


qmap_slice0, qaxis_slice0 = mreels.get_qeels_slice(data, m)


qmap_slice1, qaxis_slice1 = mreels.get_qeels_slice(data, k,
	starting_point=m)

qmap_slice2, qaxis_slice2 = mreels.get_qeels_slice(data, gamma,
	starting_point=k)

qaxis_slice2 = qaxis_slice2[::-1] + qaxis_slice1.max()

qmap_slice = np.append(qmap_slice0[:-1,:], qmap_slice1, axis=0)
qmap_slice = np.append(qmap_slice[:-1,:], qmap_slice2, axis=0)

qaxis_slice = np.append(qaxis_slice0[:-1], qaxis_slice1, axis=0)
qaxis_slice = np.append(qaxis_slice[:-1], qaxis_slice2, axis=0)

print("Points")
print("M:{:.2f} at index {}".format(
	qaxis_slice0[-1], len(qaxis_slice0)))

print("K:{:.2f} at index {}".format(
	qaxis_slice1[-1], len(qaxis_slice1)))

print("G:{:.2f} at index {}".format(
	qaxis_slice2[-1], len(qaxis_slice2)))

yc, xc = data.get_centre()

tosum = data.stack[:,700:1400,800:1300]
tosum[:,yc-50:yc+50,xc-50:xc+50] = 0
imspec = np.sum(tosum, axis=(2,1))

qmap_corr = mreels.batson_correct(data, 2, qmap_slice, imspec)

qmap = np.copy(qmap_corr)


qmap[0:28] = qmap_slice[0:28]
qmap[520:] = qmap_slice[520:]

qmpool, qapool = mreels.pool_qmap(qmap, qaxis_slice, 2)
ppos, perr = mreels.find_peak_in_range(qmpool, np.argwhere(eax==15)[0][0], 50)

ppos[60:-60] -= 10

def func(x, a, b, c):
	return -a*np.sin(x*b)+c

msk = np.where((eax[ppos]<=13) & (eax[ppos]>= 20), False, True)


popt, pcov = cv(func, qapool[msk], eax[ppos][msk])

ydata = func(qaxis_slice, *popt)


qmap2plot = np.copy(qmap_slice)-qmap_slice.min()
qmap2plot[28:520] = np.copy(qmap_corr[28:520])-qmap_corr[28:520].min()

a = 100

qmap2plot = mreels.sigmoid(qmap2plot, (a,-a))

fig, ax = plt.subplots(2,1, sharex=True)

plt.gcf().set_size_inches(1.6*8, 8)

ax[0].pcolormesh(qaxis_slice, eax, qmap2plot.T)
ax[0].errorbar(qapool, eax[ppos], perr*0.25, fmt='+', color='red')
ax[0].set_ylim([0,36])
ax[0].set_ylabel(r"energy loss $\Delta E$ [$eV$]")
ax[0].set_xticks([0, qaxis_slice0.max(), qaxis_slice1.max(), qaxis_slice.max()])
ax[0].vlines(qaxis_slice0.max(), 0, 36, linestyle='--', linewidth=2, color='white')
ax[0].vlines(qaxis_slice1.max(), 0, 36, linestyle='--', linewidth=2, color='white')
ax[0].title.set_text('q-EELS map')

ax[1].errorbar(qapool[msk], eax[ppos][msk], perr[msk]*0.25, fmt='+', color='red')
ax[1].plot(qaxis_slice, ydata, linestyle="-.", color='blue')
ax[1].set_ylim([12,22])
ax[1].set_ylabel(r"energy loss $\Delta E$ [$eV$]")
ax[1].set_xlabel(r"momentum transfer $q_{\perp}$")
ax[1].vlines(qaxis_slice0.max(), 0, 36, linestyle='--', linewidth=2, color='black')
ax[1].vlines(qaxis_slice1.max(), 0, 36, linestyle='--', linewidth=2, color='black')
ax[1].title.set_text('Batson corrected q-EELS map')
ax[1].set_xticks([0, qaxis_slice0.max(), qaxis_slice1.max(), qaxis_slice.max()])
ax[1].set_xticklabels([
	r"$\Gamma$",
	r"$M$",
	r"$K$",
	r"$\Gamma$"
	], fontsize=12)

plt.savefig('qmap_peak.eps', format='eps')
plt.show()
