import numpy as np
import matplotlib.pyplot as plt
fc=20  # Frecuencia de corte
N= 9   # orden del filtro


m= np.linspace(0, 2*fc, 2*fc)
H= 1/ (1+(np.power(m/fc,2*N)))

plt.plot(H)
plt.show()
