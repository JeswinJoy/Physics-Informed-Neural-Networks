!pip install sciann

import numpy as np
import matplotlib.pyplot as plt
import sciann as sn
import tensorflow as tf
from tensorflow import keras as k

x = sn.Variable('x')
y = sn.Variable('y')
T = sn.Functional('T', [x,y], 6*[200], 'tanh', kernel_initializer = k.initializers.GlorotUniform( seed = 3678))

from sciann.utils.math import diff, sign
L1 = diff(T, x, order = 2) + diff(T, y, order = 2)

c1 = (10 - T)*(10 - T)*(1 - sign(x - 1))*(1 - sign(1 - y))
c2 = (20 - T)*(20 - T)*(1 - sign(y - 1))*(1 - sign(1 - x))
c3 = (30 - T)*(30 - T)*(1 - sign(x - 1))*(1 - sign(y))
c4 = (40 - T)*(40 - T)*(1 - sign(y - 1))*(1 - sign(x))

m = sn.SciModel([x, y], [L1, c1, c2, c3, c4] , optimizer= k.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))

x_data, y_data = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))

h = m.train([x_data, y_data], 5*['zero'] ,batch_size=512 , epochs = 20000, verbose = 2)

m.save_weights('weights20000.h5')

plt.semilogy(h.history['loss'])

x_test, y_test = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
T_pred = T.eval(m, [x_test, y_test])

fig = plt.figure(figsize = (4, 4))
plt.pcolor(x_test, y_test, T_pred, cmap = 'seismic')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
