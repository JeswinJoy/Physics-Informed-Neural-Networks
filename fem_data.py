import scipy.io as sio
from scipy.io import loadmat

FEM_Data = loadmat('/content/T_FEM.mat')

fig = plt.figure(figsize = (4, 4))
plt.pcolor(x_test, y_test, T_fem, cmap = 'seismic')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()

MSE = np.square(np.subtract(T_fem,T_pred)).mean()
print(MSE)
