import mreels
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

if __name__ == "__main__":
	data = mreels.MomentumResolvedDataStack(
		'n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4',
		25
	)

	data.rem_neg_el()
	data.remove_neg_val()
	data.correct_drift()

	gamma = data.get_centre()
	m = (992, 812)
	k = (1113, 766)

	r1 = int(np.sqrt((m[0]-gamma[0])**2 + (gamma[1]-m[1])**2))

	qmap_rad, qaxis_rad = mreels.get_qeels_data(data, r1, 1, 25, threads=8)
	np.save('qmap_rad.npy', qmap_rad)

	qmap_rad = None


	qmap_line0, qaxis_line0 = mreels.get_qeels_data(data, 1000, 1, 23,
		forward_peak=m, method='line', threads=8, peak_width=30)

	"""
	qmap_line1, qaxis_line1 = mreels.get_qeels_data(data, 1000, 1, 25,
		forward_peak=k, method='line', threads=8, starting_point=m, peak_width=20)


	qmap_line2, qaxis_line2 = mreels.get_qeels_data(data, 1000, 1, 25,
		forward_peak=gamma, method='line', threads=8, starting_point=k)

	qmap_line = np.append(qmap_line0[:-1,:], qmap_line1, axis=0)
	qmap_line = np.append(qmap_line[:-1,:], qmap_line2, axis=0)
	"""

	print("Points")

	print("M:{:.2f} at index {}".format(qaxis_line0[-1], len(qaxis_line0)))
	#print("K:{:.2f} at index {}".format(qaxis_line1[-1], len(qaxis_line1)))
	#print("G:{:.2f} at index {}".format(qaxis_line2[-1], len(qaxis_line2)))

	np.save('qmap_line.npy', qmap_line0)
	qmap_line = None

	qmap_slice0, qaxis_slice0 = mreels.get_qeels_slice(data, m)

	"""
	qmap_slice1, qaxis_slice1 = mreels.get_qeels_slice(data, k,
		starting_point=m)

	qmap_slice2, qaxis_slice2 = mreels.get_qeels_slice(data, gamma,
		starting_point=k)

	qmap_slice = np.append(qmap_slice0[:-1,:], qmap_slice1, axis=0)
	qmap_slice = np.append(qmap_slice[:-1,:], qmap_slice2, axis=0)

	"""

	print("Points")
	print("M:{:.2f} at index {}".format(qaxis_slice0[-1], len(qaxis_slice0)))
	#print("K:{:.2f} at index {}".format(qaxis_slice1[-1], len(qaxis_slice1)))
	#print("G:{:.2f} at index {}".format(qaxis_slice2[-1], len(qaxis_slice2)))

	np.save('qmap_slice.npy', qmap_slice0)
