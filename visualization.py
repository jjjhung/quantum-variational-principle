#Visualization Fn

import numpy as np
import matplotlib.pyplot as plt

from network import *
from hamiltonian import *

network = RadialBasisFunctionNetwork(2,1,10)

nx = np.linspace(0,40,40)
ny = np.linspace(0,40,40)
x,y = np.meshgrid(nx,ny)

z = 0*x

for i in range(len(nx)):
    for j in range(len(ny)):
        print(np.array([x[i,j],y[i,j]]))
        z[i,j] = i*j / 10
        
        
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