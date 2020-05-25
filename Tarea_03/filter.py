import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

fc = 20 # Frecuencia de corte
N =9 #ordendelfiltro
x = np.linspace(-2*fc, 2*fc, 4*fc+1)
y = np.linspace(-2*fc, 2*fc, 4*fc+1)
X, Y = np.meshgrid(x, y)
Z = 1 / (1+(np.power(np.sqrt(X**2+Y**2)/fc, 2*N)))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.Spectral)
plt.show()
