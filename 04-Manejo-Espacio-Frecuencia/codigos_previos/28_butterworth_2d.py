
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

fc = 28  # Frecuencia de corte
N = 3   # orden del filtro

x = np.linspace(-2*fc, 2*fc, 4*fc+1)
y = np.linspace(-2*fc, 2*fc, 4*fc+1)
X, Y = np.meshgrid(x, y)

Z = 1 / (1 + (np.power(np.sqrt(np.power(X, 2)+np.power(Y, 2))/fc, 2*N)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.Spectral)
plt.show()
