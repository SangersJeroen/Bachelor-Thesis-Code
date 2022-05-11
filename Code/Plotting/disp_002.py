import numpy as np
import mreels
import matplotlib.pyplot as plt

data002 = mreels.MomentumResolvedDataStack('InSe_002.dm4', 25)
data002.rem_neg_el()
data002.remove_neg_val()
#data002.correct_drift()

qmap, qaxis = mreels.get_qeels_slice(data002, (581, 945))
eax = data002.axis0
print(eax)

mreels.plot_qeels_data(data002, mreels.sigmoid(qmap, (50, 150)), qaxis, "")
qmap, qaxis = mreels.pool_qmap(qmap, qaxis, 4)


centre, window = 14.5, 14

ppos, perr = mreels.find_peak_in_range(qmap, np.argwhere(eax==centre)[0][0], window)

plt.errorbar(qaxis, eax[ppos], perr*0.5, fmt='+')

plt.xlim([-0.1,1.1])
plt.ylim([12,18])
plt.xlabel(r'$q$ [$nm^{-1}$]')
plt.ylabel(r'$\Delta eV$ [$eV$]')
plt.title(r'Tracked peaks in $\Gamma \rightarrow [\overline{1} 1 \overline{2}]$')
plt.legend()
plt.show()