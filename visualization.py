#Visualization Fn

import numpy as np
import matplotlib as plt

from radial-basis-network import *
from hamiltonian-2d-harm-oss import *


nx = np.linspace(0,40,10000)
ny = np.linspace(0,40,10000)
x,y = np.meshgrid(kx,ky)

z = 0*x

for i in range(len(nx)):
    for j in range(len(ny)):
        z[i,j] = network.psi(nx[i,j],ny[i,j])
        
        
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.contour3D(x, y, z, 150, cmap='binary', alpha=1.)
ax.view_init(0, 0)  # elevation, azimuth
ax.set_xlabel('nx')
ax.set_ylabel('ny')
ax.set_zlabel('')
plt.title("")
plt.legend()
plt.show()