print('importing csv')
import csv
# import nibabel as nib
print('importing numpy')
import numpy as np
print('importing scipy')
import scipy as sp
from scipy import sparse
# from scipy import ndimage
# import skimage.transform
print('importing matplotlib')
# from matplotlib import pyplot as plt
import matplotlib as mptlb
mptlb.use('TkAgg')
import matplotlib.pyplot as plt

# from keras.layers import InputLayer, Conv2D, MaxPool2D, Flatten, Dense, UpSampling2D, LocallyConnected2D, Activation
# from keras.models import Model, Sequential
# from keras.layers import UpSampling2D
# from keras.layers import BatchNormalization,SpatialDropout2D
# from keras.layers.advanced_activations import LeakyReLU, PReLU
# import keras.backend as K

print('Beginning Code')
nx = 50
ny = 32
n = (nx+1)*(ny+1)
xlim = 1.0
ylim = 2.0
dx = xlim/nx
dy = ylim/ny
xvec = np.linspace(0.0, xlim, nx+1)
yvec = np.linspace(0.0, ylim, ny+1)
[X, Y] = np.meshgrid(xvec, yvec, indexing='xy')
T = 1.0
dt = min(dx/16.0, dy/16.0) # CFL condition

def fx_atom(x,y,t):
    return x + y

def fy_atom(x,y,t):
    return 2.0

fx = np.vectorize(fx_atom)
fy = np.vectorize(fy_atom)

phi_func = np.vectorize(lambda x,y : 0.5 if (x > 0.5 and y > 0.8) else -0.5)
phi = phi_func(X,Y).reshape((n,1))

# Forward Euler in time, Upwind 1st order in space, periodic boundary conditions (for testing purposes)
kk = lambda i,j : j*(nx+1) + i
U1N = sp.sparse.lil_matrix((n,n))
U1P = sp.sparse.lil_matrix((n,n))
U2N = sp.sparse.lil_matrix((n,n))
U2P = sp.sparse.lil_matrix((n,n))

for i in range(nx+1):
    for j in range(ny+1):
        k = kk(i,j)
        U1N[k,k] = 1
        U1P[k,k] = -1
        U2N[k,k] = 1
        U2P[k,k] = -1
        if i > 0:
            U1N[k,kk(i-1,j)] = -1
        else:
            U1N[k,kk(nx,j)] = -1
        if i < nx:
            U1P[k,kk(i+1,j)] = 1
        else:
            U1P[k,kk(0,j)] =  1
        if j > 0:
            U2N[k,kk(i,j-1)] = -1
        else:
            U2N[k,kk(i,ny)] = -1
        if j < ny:
            U2P[k,kk(i,j+1)] = 1
        else:
            U2P[k,kk(i,0)]  =  1
U1P = U1P.tocsr()
U1N = U1N.tocsr()
U2P = U2P.tocsr()
U2N = U2N.tocsr()

zeros = np.zeros((ny+1,nx+1))

def pos(m_in):
    mtrx = np.copy(m_in)
    mtrx[mtrx <= 0] = 0
    return mtrx

def neg(m_in):
    mtrx = np.copy(m_in)
    mtrx[mtrx >= 0] = 0
    return mtrx

plt.contourf(X,Y,phi.reshape((ny+1,nx+1)))
plt.show()

t = 0;
while t < T:

    ffx = fx(X,Y,t);
    ffy = fy(X,Y,t);
    F1P = sp.sparse.diags([pos(ffx).reshape((n))], [0])
    F1N = sp.sparse.diags([neg(ffx).reshape((n))], [0])
    F2P = sp.sparse.diags([pos(ffy).reshape((n))], [0])
    F2N = sp.sparse.diags([neg(ffy).reshape((n))], [0])

    B = sp.sparse.eye(n) - dt* ((1.0/dx)* F1P*U1N + (1.0/dx)* F1N*U1P + (1.0/dy)* F2P*U2N + (1.0/dy)*F2N*U2P)
    phi = B*phi;

    plt.clf()
    plt.contourf(X,Y,phi.reshape((ny+1, nx+1)))
    plt.colorbar()
    plt.title('t = %.3f' % t)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.draw()
    plt.pause(0.000001)

    t = t+dt

plt.show()
