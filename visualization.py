#Visualization Fn

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
from network import *
from hamiltonian import *


def Visualize(network):
	nx = np.linspace(0,10,11)
	ny = np.linspace(0,10,11)


	x,y = np.meshgrid(nx,ny)

	z = np.zeros(np.shape(x))

	for i in range(len(nx)):
	    for j in range(len(ny)):
	        #rint(network.psi(np.array([x[i,j],y[i,j]])))
	        #z[i,j] = network.psi(np.array([x[i,j],y[i,j]]))
	        z[i,j] = network.psi(np.array([x[i,j],y[i,j]]))
	
	#z[np.abs(z) > 600] = 0

	# Normalize z
	z = normalize(z)

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_surface(x,y,z)
	plt.savefig('200-50000.png')
	plt.show()

	# fig = plt.figure()
	# ax = fig.add_subplot(1, 1, 1, projection='3d')
	# ax.contour3D(x, y, z, 150, cmap='binary', alpha=1.)
	# ax.view_init(0, 0)  # elevation, azimuth
	# ax.set_xlabel('nx')
	# ax.set_ylabel('ny')
	# ax.set_zlabel('')
	# plt.title("")
	# plt.legend()
	# plt.show()	