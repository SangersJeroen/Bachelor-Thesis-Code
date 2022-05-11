import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mreels
from deconvolute import *

if __name__ == "__main__":
	data = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4', 25)
	data.rem_neg_el()
	data.remove_neg_val()
	eax = data.axis0

	qmap0, qaxis0 = mreels.get_qeels_slice(data, (777, 1373))
	zlp_spec = 0*qmap0+1
	#np.sum(qmap0[:,:], axis=0)/len(qmap0[:,0])
	#zlp_spec[8:] = zlp_spec.max()/100 /eax[8:]

	for i in range(len(qmap0)):
		qmap0[i] = deconvolute(data, qmap0[i], zlp_spec)
	mreels.plot_qeels_data(data, mreels.sigmoid(qmap0), qaxis0, '')
	peak = 15
	window = 18

	ppos, perr = mreels.find_peak_in_range(qmap0, np.argwhere(eax==peak)[0][0], window)

	plt.errorbar(qaxis0, eax[ppos], yerr=perr*0.25, fmt='2', label=r'a10a1', color='green')

	plt.axhline(y=(int(peak-window/2*0.25)))
	plt.axhline(y=(int(peak+window/2*0.25)))

	plt.xlim([-0.1,1.1])
	plt.ylim([12,18])
	plt.xlabel(r'$q$ [$nm^{-1}$]')
	plt.ylabel(r'$\Delta eV$ [$eV$]')
	plt.title(r'Tracked peaks in $\Gamma \rightarrow [\overline{1} 1 \overline{2}]$')
	plt.legend()
	plt.show()